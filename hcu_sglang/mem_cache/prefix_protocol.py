from __future__ import annotations

import dataclasses
import types
from typing import Any, Optional, Callable
import torch
from sglang.srt.mem_cache import base_prefix_cache as B
from sglang.srt.mem_cache import unified_radix_cache as U
from sglang.srt.session import streaming_session as S
from sglang.srt.mem_cache.unified_cache.components import swa as W
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from hcu_sglang.utils.detection import is_hcu

@dataclasses.dataclass
class _MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None
    # Match only the full-attention component. Decode-side DSV4 prompt
    # donation intentionally leaves SWA tombstones while retaining reusable
    # full-attention pages.
    return_full_match: bool = False

@dataclasses.dataclass
class _InsertParams:
    """Unified parameters for insert across different cache types"""

    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None
    # The ping-pong slot prepare picked; cleanup keeps that same slot.
    mamba_keep_idx: Optional[int] = None

    # DSV4 NPU C128 sidecar pages, one page id per physical C128 page group.
    c128_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_branching_seqlen: Optional[int] = None
    force_leaf_creation: bool = False

    # General
    component_evicted_seqlens: dict[ComponentType, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    # The inserting request already inserted [0, here) (req.kv.cache_inserted_len);
    # only the nodes past it count a hit, so a request counts each node once.
    inserted_len: int = 0
    priority: int = 0
    session_id: Optional[str] = None
    track_adopted_ranges: bool = False

    # Logical-page KV sharding: rotation base of the chain the inserted
    # values belong to (stamped onto new tree nodes; None when sharding is
    # off). See UnifiedTreeNode.rotation_base.
    rotation_base: Optional[int] = None

    def get_evicted_seqlen(self, component_type: ComponentType) -> int:
        return self.component_evicted_seqlens.get(component_type, 0)

    def set_evicted_seqlen(self, component_type: ComponentType, length: int) -> None:
        self.component_evicted_seqlens[component_type] = length

class _BaseMethods:
    def evict_for_alloc(
        self,
        params: EvictParams,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> EvictResult:
        """Evict for allocation using shortfalls or joint-reclaim quotas.

        Without ``allocation_reclaim_satisfied``, callers pass component
        shortfalls in ``params``. The default implementation forwards these
        counts to :meth:`evict`; multi-component caches may stop earlier when
        collateral frees satisfy the requested allocation.

        With ``allocation_reclaim_satisfied``, ``params`` instead contains
        cumulative reclaim quotas. Supporting caches may stop at the pure
        predicate before reaching those quotas. The allocator still owns
        preparation and the actual capacity check after the walk.
        """
        if allocation_reclaim_satisfied is not None:
            raise NotImplementedError("This cache does not support joint reclaim")
        return self.evict(params)

    def evict_sliding_windows(
        self, req: Req, pre_len: int, *, eviction_interval: int = 1
    ) -> None:
        """Slide request-owned windows at the scheduler's safe eviction frontier."""
        try:
            from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
        except ImportError:
            from hcu_sglang.mem_cache.common import free_swa_out_of_window_slots

        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.swa_retain_window,
            page_size=self.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            supports_prefix_sharing=self.supports_prefix_sharing(),
            retain_floor=self.swa_retain_floor(req),
            eviction_interval=eviction_interval,
        )

    def swa_reprefill_tail_tokens(self, req=None) -> int:
        # Caches whose matched SWA is not reusable as is override this.
        return 0


class _UnifiedMethods:
    def evict_for_alloc(
        self,
        params: EvictParams,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> EvictResult:
        """Evict using component shortfalls or cumulative joint-reclaim quotas.

        Without ``allocation_reclaim_satisfied``, ``params`` contains component
        shortfalls. These become availability targets, allowing collateral frees
        to satisfy an allocation before its requested eviction count is reached.

        With ``allocation_reclaim_satisfied``, ``params`` instead contains
        cumulative reclaim quotas and component-local targets are bypassed.
        The pure predicate may stop the walk before those quotas are reached;
        the allocator must prepare and recheck actual capacity after the walk.
        """
        if self.disable:
            return EvictResult()
        if allocation_reclaim_satisfied is not None:
            return self._evict(
                params, allocation_reclaim_satisfied=allocation_reclaim_satisfied
            )

        request_by_type = self._evict_request_by_type(params)
        available_size_targets = {
            ct: (ct, self._component_available_size(ct) + request_cnt)
            for ct, request_cnt in request_by_type.items()
            if request_cnt > 0
        }
        allocator = self.token_to_kv_pool_allocator
        mamba_full_donor = allocator.mamba_full_cache_donor()
        mamba_target = available_size_targets.get(ComponentType.MAMBA)
        initial_params = params
        if mamba_target is not None and mamba_full_donor is not None:
            # Full KV can supply bytes but cannot recycle Mamba virtual IDs.
            mamba_id_shortfall = max(
                0,
                mamba_target[1]
                - self.req_to_token_pool.mamba_allocator.available_size(),
            )
            initial_params = EvictParams(
                num_tokens=params.num_tokens,
                swa_num_tokens=params.swa_num_tokens,
                mamba_num=mamba_id_shortfall,
            )
        result = self._evict(initial_params, available_size_targets)

        if mamba_target is not None and mamba_full_donor is not None:
            mamba_full_donor.prepare_mamba_allocation(mamba_target[1])
            mamba_free_ids = self.req_to_token_pool.mamba_allocator.available_size()
            mamba_capacity = self._component_available_size(ComponentType.MAMBA)

            if mamba_free_ids >= mamba_target[1] and mamba_capacity < mamba_target[1]:
                full_evictable = self.full_evictable_size()
                if full_evictable > 0:
                    donor_result = self._evict(
                        EvictParams(num_tokens=full_evictable),
                        {ComponentType.FULL: mamba_target},
                    )
                    result.num_tokens_evicted += donor_result.num_tokens_evicted
                    result.swa_num_tokens_evicted += donor_result.swa_num_tokens_evicted
                    result.mamba_num_evicted += donor_result.mamba_num_evicted

                # Preserve Mamba-victim recovery if Full cannot fund the target.
                if (
                    self._component_available_size(ComponentType.MAMBA)
                    < mamba_target[1]
                ):
                    mamba_evictable = self.mamba_evictable_size()
                    if mamba_evictable > 0:
                        fallback_result = self._evict(
                            EvictParams(mamba_num=mamba_evictable),
                            {ComponentType.MAMBA: mamba_target},
                        )
                        result.num_tokens_evicted += fallback_result.num_tokens_evicted
                        result.swa_num_tokens_evicted += (
                            fallback_result.swa_num_tokens_evicted
                        )
                        result.mamba_num_evicted += fallback_result.mamba_num_evicted

        return result

    def _evict(
        self,
        params: EvictParams,
        available_size_targets: Optional[
            dict[ComponentType, tuple[ComponentType, int]]
        ] = None,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        request_by_type = self._evict_request_by_type(params)
        self._evict_components(
            request_by_type,
            tracker,
            available_size_targets=available_size_targets,
            allocation_reclaim_satisfied=allocation_reclaim_satisfied,
        )

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        # Report full-layer tokens only
        self.update_eviction_metrics(tracker[BASE_COMPONENT_TYPE], start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def _evict_components(
        self,
        request_by_type: dict[ComponentType, int],
        tracker: dict[ComponentType, int],
        available_size_targets: Optional[
            dict[ComponentType, tuple[ComponentType, int]]
        ] = None,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> None:
        # Buffer mode: eviction always wins over queued backup intents — a
        # destroyed victim's intent is stale-swept and the content rewrites
        # after its recompute.
        last_mamba_donor_check = 0
        mamba_donor_prepared = False

        def target_reached(component_type: ComponentType) -> bool:
            nonlocal last_mamba_donor_check, mamba_donor_prepared
            if allocation_reclaim_satisfied is not None:
                return allocation_reclaim_satisfied()
            if available_size_targets is None:
                return False
            target = available_size_targets.get(component_type)
            if target is None:
                return False
            target_component, target_size = target
            # A Full-leaf cascade can release Mamba or SWA state directly.
            if self._component_available_size(target_component) >= target_size:
                return True
            if (
                component_type == ComponentType.FULL
                and target_component == ComponentType.MAMBA
            ):
                donor = self.token_to_kv_pool_allocator.mamba_full_cache_donor()
                assert donor is not None, "Mamba target requires a Full donor"
                recheck_after = (
                    1
                    if mamba_donor_prepared
                    else donor.full_tokens_before_mamba_recheck(target_size)
                )
                if tracker[component_type] - last_mamba_donor_check < recheck_after:
                    return False
                donor.prepare_mamba_allocation(target_size)
                last_mamba_donor_check = tracker[component_type]
                mamba_donor_prepared = True
            # Schedulable capacity includes donor holes that allocation can compact.
            return self._component_available_size(target_component) >= target_size

        for ct in self.tree_components:
            request_cnt = request_by_type[ct]
            # An allocation-driven FULL request names the allocator's
            # shortfall, not what the walk must free: an allocator can hide
            # headroom behind a clamp (a reserve it refills before raw
            # allocation sees a page again) or page rounding, so freeing
            # exactly the shortfall can leave its target unmet. Such a walk
            # runs to the target; the check after every victim still stops it
            # at the first sufficient one.
            to_target = (
                ct == ComponentType.FULL
                and request_cnt > 0
                and available_size_targets is not None
                and ct in available_size_targets
            )
            # A preceding component may have cascade-evicted this component or,
            # on a shared pool, released enough bytes to satisfy its allocation.
            if target_reached(ct) or (not to_target and tracker[ct] >= request_cnt):
                continue
            walk_quota = request_cnt
            if to_target:
                walk_quota = max(
                    request_cnt, tracker[ct] + self.tree_core.full_evictable_size()
                )
            self.tree_core.evict_device_start(ct, walk_quota)
            try:
                while True:
                    node_id, made_progress = self._evict_device_next_node(ct, tracker)
                    if node_id is None:
                        if not made_progress:
                            break
                    else:
                        backup_kv = self._evict_device_leaf(node_id, tracker)
                    if node_id is not None and backup_kv is not None:
                        # Deferred demote: run the D->H backup, demote only on success.
                        written = self._execute_and_commit_kv_backup(
                            backup_kv, write_back=True
                        )
                        if written > 0:
                            self.writing_check(write_back=True)
                            self._demote(node_id, tracker)
                        elif self._drop_subtree_no_host(node_id, tracker):
                            logger.warning(
                                "write_back: KV subtree dropped without backup "
                                "due to host memory pressure, root node %d",
                                node_id,
                            )
                        else:
                            logger.warning(
                                "write_back: backup failed under host memory "
                                "pressure but subtree drop declined (node "
                                "locked); root node %d stays device-resident "
                                "until host space frees",
                                node_id,
                            )
                    if target_reached(ct):
                        break
            finally:
                self.tree_core.evict_device_end(ct)

    def swa_reprefill_tail_tokens(self, req=None) -> int:
        """Tokens at the end of a matched prefix to re-prefill because their SWA
        cannot be reused as is.

        unified_kv keeps SWA in a per-request ring (state_slot/pos) the tree never
        stores, so a reused window would read another request's stale slots; a
        host pool restores exact contents instead. Decoder SWA bounded replay
        writes late-layer window KV only for an extend's tail rows, so a matched
        prefix may lack it; a streaming turn borrowing its own record has it.
        """
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        swa = self.components.get(ComponentType.SWA)
        if swa is None or not swa.sliding_window_size:
            return 0
        if self.decoder_swa_bounded_replay and not self.session.will_lend_record(req):
            return swa.sliding_window_size
        if not is_unified_kv_triton():
            return 0
        if self.tree_core.has_swa_host_pool:
            return 0
        return swa.sliding_window_size


class _SessionMethods:
    def will_lend_record(self, req: Optional[Req]) -> bool:
        """Whether req has a slot record to borrow. Unlike find_active_slot, it
        leaves a pre-aborted req attached."""
        if not _is_streaming(req) or req.to_finish is not None:
            return False
        slot = self.slots.get(req.session.session_id)
        return slot is not None and slot.kv.holds_kv


class _SwaMethods:
    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        # Unfinished requests can already have an SWA-evicted prefix; preserve
        # that boundary so insertion creates a tombstone instead of live SWA KV.
        insert_params.set_evicted_seqlen(
            self.component_type, req.kv.get_evicted_seqlen(self.component_type)
        )
        insert_params.force_leaf_creation = getattr(
            req, "force_radix_leaf_creation", False
        )

        # A recurrent checkpoint must stay attached to its exact token prefix.
        # Let MambaComponent select the insertion length for hybrid caches.
        if self.cache.is_mamba_enabled:
            return None

        branching_seqlen = req.swa_branching_seqlen
        if branching_seqlen is None or branching_seqlen <= req.kv.cache_protected_len:
            return None

        # An EAGLE key with N bigrams spans N + 1 raw tokens.
        effective_cache_len = branching_seqlen + int(self.tree_core.is_eagle)
        if effective_cache_len > token_ids_len:
            return None

        # Record the logical SWA branch boundary for insertion.
        insert_params.swa_branching_seqlen = branching_seqlen
        return effective_cache_len

_REGISTERED = False


def _bind(source, module, owner, name):
    fn = types.FunctionType(source.__code__, vars(module), name,
                            source.__defaults__, source.__closure__)
    fn.__kwdefaults__ = source.__kwdefaults__
    fn.__annotations__ = source.__annotations__
    fn.__qualname__ = owner.__qualname__ + "." + name
    return fn


def _extend_schema(native, source):
    for field in dataclasses.fields(source):
        if field.name not in native.__dataclass_fields__ and field.default is not dataclasses.MISSING:
            setattr(native, field.name, field.default)
    for name in ("__init__", "__repr__", "__eq__"):
        setattr(native, name, source.__dict__[name])
    native.__dataclass_fields__ = source.__dataclass_fields__.copy()
    native.__annotations__ = source.__annotations__.copy()
    native.__match_args__ = source.__match_args__


def register_hooks():
    global _REGISTERED
    if _REGISTERED or not is_hcu():
        return
    _extend_schema(B.MatchPrefixParams, _MatchPrefixParams)
    _extend_schema(B.InsertParams, _InsertParams)
    for module, owner, source in (
        (B, B.BasePrefixCache, _BaseMethods),
        (U, U.UnifiedRadixCache, _UnifiedMethods),
        (S, S.StreamingSession, _SessionMethods),
        (W, W.SWAComponent, _SwaMethods),
    ):
        for name, raw in vars(source).items():
            if not isinstance(raw, types.FunctionType):
                continue
            fn = _bind(raw, module, owner, name)
            if module is U and name == "evict_for_alloc":
                fn = U.rank_consensus(
                    same_params=["params", "allocation_reclaim_satisfied is not None"],
                    same_results=True,
                )(fn)
            setattr(owner, name, fn)
    from hcu_sglang.mem_cache import tree_protocol
    tree_protocol.register_hooks()
    _REGISTERED = True
