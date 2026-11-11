# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from sglang.srt.managers import schedule_policy as N
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from hcu_sglang import encoder_swa_replay
from hcu_sglang.utils.detection import is_hcu


@dataclass(frozen=True, slots=True)
class _PrefillAdmission:
    prefix_len: int
    extend_len: int
    max_new_tokens: int
    is_chunked: bool
    replay_len: int


class PrefillAdder(N.PrefillAdder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fork's admission reserved unified-pool mamba state slots; 1be97a0's
        # PrefillAdder dropped the members. Rebuild them in their neutral
        # fallback shape (no reservation) unless the allocator exposes the
        # fork's cost ABI.
        self._mamba_slot_cost = 0
        self.rem_mamba_slots = None
        _cost_fn = getattr(self.token_to_kv_pool_allocator, "mamba_slot_full_token_cost", None)
        if _cost_fn is not None:
            try:
                self._mamba_slot_cost = _cost_fn()
            except Exception:
                self._mamba_slot_cost = 0
        if self._mamba_slot_cost:
            _mamba_alloc = getattr(self.token_to_kv_pool_allocator, "mamba_allocator", None)
            _available = (
                _mamba_alloc.schedulable_available_size()
                if _mamba_alloc is not None
                else 0
            )
            if self.tree_cache.supports_mamba():
                _available += self.tree_cache.mamba_evictable_size()
            self.rem_mamba_slots = _available

    def _mamba_gap_budget_for_req(
        self, req: Req, *, include_host_hit: bool = False
    ) -> int:
        """Reserve full-token equivalents for shared-pool Mamba state slots.

        Neutral (0) while no unified-mamba allocator is attached, matching the
        fork's allocator-absent fallback.
        """
        if not self._mamba_slot_cost:
            return 0
        slots = int(not req.kv.holds_mamba)
        if include_host_hit:
            slots += req.mamba_host_hit_length
        return slots * self._mamba_slot_cost

    def _check_prefill_budget(
        self,
        req: Req,
        *,
        extend_input_len: int,
        total_tokens: int,
        swa_host_hit_length: int,
        chunk_limit: Optional[int],
    ) -> tuple[bool, Optional[int]]:
        return self.memory_budget.check_prefill(
            extend_input_len=extend_input_len,
            total_tokens=total_tokens,
            max_new_tokens=self._swa_new_tokens(req),
            input_tokens=len(req.full_untruncated_fill_ids),
            swa_host_hit_length=swa_host_hit_length,
            chunk_limit=chunk_limit,
            auxiliary_swa_host_hit_length=req.auxiliary_swa_host_hit_length,
        )

    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
        mamba_gap_reserve: int = 0,
        request_slot_reserve: int = 0,
        is_chunked_continuation: bool = False,
        replay_len: int = 0,
    ):
        """Charge one admitted request against the prefill budgets.

        The compute budgets (`rem_chunk_tokens`, `rem_input_tokens`,
        `rem_dllm_tokens`) take the rows the step runs: the extend, plus
        `replay_len` cached tokens that encoder replay recomputes. The KV
        budgets take only the page-ceiled extend. Exact-chunk-fill bills the
        compute budgets the raw extend instead, all alike, so none runs out
        first with rounding slack unspent.
        """
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        raw_extend_input_len = extend_input_len
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        compute_charge = (
            raw_extend_input_len if self.exact_chunk_fill else extend_input_len
        ) + replay_len

        self.memory_budget.reserve(
            extend_input_len,
            max_new_tokens,
            # reserve() already charges one page; add the remaining shard pages.
            extra_tokens=(
                mamba_gap_reserve
                + request_slot_reserve
                + self.per_req_token_overhead
                - self.page_size
            ),
            chunk_limit=self.rem_chunk_tokens,
            is_chunked_continuation=is_chunked_continuation,
        )
        # Debit every reserved state slot, including the load-back checkpoint.
        if mamba_gap_reserve and self.rem_mamba_slots is not None:
            self.rem_mamba_slots -= mamba_gap_reserve // self._mamba_slot_cost
        self.rem_input_tokens -= compute_charge

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= compute_charge
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= compute_charge

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += raw_extend_input_len
        self.log_replay_tokens += replay_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += raw_extend_input_len

    def _replay_len(self, req: Req, prefix_len: int) -> int:
        if (
            N.get_exec().features.enable_encoder_swa_bounded_replay
            and encoder_swa_replay.resets_swa_window(req)
        ):
            return encoder_swa_replay.replay_len(prefix_len)
        return 0

    def _account_prefill_cache_admission(self, req: Req, prefix_len: int) -> None:
        if req.retracted_stain:
            # Retraction attribution is intentionally omitted for now; discard
            # its lifecycle state so a later abort cannot report it as a drop.
            self.tree_cache.discard_storage_prefetch_accounting(
                req.cache_request_handle
            )
            self.tree_cache.finish_storage_prefetch_admission(
                req.cache_request_handle, fulfilled_tokens=0, reason=None
            )
            return

        if prefix_len > 0:
            device_hit, host_hit, storage_hit = N.split_cached_prefix_by_tier(
                prefix_len=prefix_len,
                host_hit_len=req.materialized_host_hit_len(),
                storage_hit_len=req.storage_hit_length,
                storage_hit_start=req.storage_hit_start,
                host_hit_is_storage=req.host_hit_is_storage,
            )
            self.log_device_hit_tokens += device_hit
            self.log_host_hit_tokens += host_hit
            self.log_storage_hit_tokens += storage_hit

        fulfilled_storage_hit = req.fulfilled_storage_hit_len(prefix_len)
        reason = None
        if fulfilled_storage_hit < req.storage_hit_length:
            reason = (
                "device_capacity"
                if req.needs_host_load_back()
                and req.host_loaded_length < req.host_hit_length
                else "cache_admission_shortfall"
            )
        self.tree_cache.finish_storage_prefetch_admission(
            req.cache_request_handle,
            fulfilled_tokens=fulfilled_storage_hit,
            reason=reason,
        )

    def _add_dllm_req(self, req: Req):
        prefix_len = req.prefix_len
        trunc_len = self._get_dllm_extend_len(req, prefix_len)
        req.extend_end = prefix_len + trunc_len

        self.can_run_list.append(req)

        self._update_prefill_budget(
            prefix_len,
            trunc_len,
            0,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            replay_len=self._replay_len(req, prefix_len),
        )
        self._account_prefill_cache_admission(req, prefix_len)

    def add_chunked_req(self, req: Req):
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens(req)
        else:
            _rem_tokens = self.memory_budget.available_chunk_tokens(
                self.rem_chunk_tokens
            )
            if _rem_tokens is None:
                return req
            if self.kv_shard_granule:
                # Align the absolute boundary to keep pages within one scratch region.
                # If no budget remains, use a full chunk to make progress.
                prefix_len = req.prefix_len
                floored = (
                    prefix_len + _rem_tokens
                ) // self.kv_shard_granule * self.kv_shard_granule - prefix_len
                _rem_tokens = floored if floored > 0 else self.rem_chunk_tokens

        # A mid-chunk rank prefills this pass regardless of the delayer
        # verdict, so report prefillable=True and ignore the result.
        if self.prefill_delayer_single_pass is not None:
            self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )

        if self.chunked_req_limit is not None:
            assert self.chunked_req_limit > 0
            _rem_tokens = min(_rem_tokens, self.chunked_req_limit)

        cand_extend_input_len = len(req.full_untruncated_fill_ids) - req.prefix_len
        _rem_tokens = self.memory_budget.fit_chunk(
            extend_input_len=cand_extend_input_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=_rem_tokens,
        )
        if _rem_tokens is None:
            return req
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        request_slot_reserve = self._request_slot_reserve_for_req(
            req, extend_input_len=new_len
        )
        if request_slot_reserve and not self.memory_budget.can_allocate_prefill(
            paged_input=(
                self.ceil_paged_tokens(new_len)
                + self.per_req_token_overhead
                + request_slot_reserve
            ),
            extend_input_len=new_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=self.rem_chunk_tokens,
        ):
            return req
        # The continuing chunk must fit. Keep reservation outside assert for -O.
        reserved = self._kv_shard_reserve_scratch(
            prefix_len=req.prefix_len, extend_len=new_len
        )
        if not reserved:
            raise RuntimeError("chunked request exceeds the sharded assembly scratch")
        req.extend_end = req.prefix_len + new_len
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_len,
            (
                min(req.sampling_params.max_new_tokens, N.CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            request_slot_reserve=request_slot_reserve,
            is_chunked_continuation=True,
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    def add_one_req_ignore_eos(self, req: Req):
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - req.prefix_len
        paged_input = (
            self.ceil_paged_tokens(cand_extend_input_len) + self.per_req_token_overhead
        )
        # Shared Mamba pool: fold the new mamba state's shared-gap cost into the
        # budget gate so admission can't over-commit (0 for baseline / non-Mamba).
        paged_input += self._mamba_gap_budget_for_req(req)
        paged_input += self._request_slot_reserve_for_req(req)
        fits = self.memory_budget.can_allocate_prefill(
            paged_input=paged_input,
            extend_input_len=cand_extend_input_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=self.rem_chunk_tokens,
        )
        if not fits:
            return N.AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - paged_input
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= N.IGNORE_EOS_RESERVE_TOKENS * bs:
                    return N.AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return N.AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return N.AddReqResult.OTHER

            if (
                tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
            ) is not None:
                return tile_stop

            self._add_dllm_req(req)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or cand_extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            if (
                tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
            ) is not None:
                return tile_stop

            # Non-chunked prefill — the whole sequence is committed this iter.
            if not self._kv_shard_reserve_scratch(
                prefix_len=req.prefix_len,
                extend_len=cand_extend_input_len,
            ):
                return N.AddReqResult.OTHER
            req.extend_end = len(req.full_untruncated_fill_ids)
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_len,
                min(req.sampling_params.max_new_tokens, N.CLIP_MAX_NEW_TOKENS),
                req.retracted_stain,
                mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                request_slot_reserve=self._request_slot_reserve_for_req(req),
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return N.AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            if (tile_stop := self._check_prefill_tile_budget(trunc_len)) is not None:
                return tile_stop

            # Reserve after the tile gate so rejected requests consume no scratch.
            if not self._kv_shard_reserve_scratch(prefix_len=0, extend_len=trunc_len):
                return N.AddReqResult.OTHER

            assert req.prefix_len == 0
            req.extend_end = req.prefix_len + trunc_len
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(
                0,
                trunc_len,
                0,
                req.retracted_stain,
                mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                request_slot_reserve=self._request_slot_reserve_for_req(
                    req, extend_input_len=trunc_len
                ),
            )

        return self.budget_state()

    def _defer_staged_for_capacity_if_idle(self, req: Req, *, pool: str) -> None:
        # Running work and admitted prefills (including continuing chunks) can
        # release capacity. Only idle budget failures should force recomputation.
        if self.can_run_list or (
            self.running_batch is not None and self.running_batch.reqs
        ):
            return
        if self.tree_cache.buffer_pipeline is not None:
            self.tree_cache.buffer_pipeline.defer_staged_admission(req, pool)

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return N.AddReqResult.OTHER

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # Reserve page_size for page-alignment overhead: the paged allocator may
        # consume one extra page per request (see alloc_extend), which
        # _update_prefill_budget also deducts.
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            N.CLIP_MAX_NEW_TOKENS,
        )
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - req.prefix_len
        total_tokens = cand_extend_input_len + max_new + self.per_req_token_overhead
        # Reserve request and load-back state slots before init_load_back binds
        # the request slot. Reuse this reservation when committing admission.
        mamba_gap_reserve = self._mamba_gap_budget_for_req(req, include_host_hit=True)
        total_tokens += mamba_gap_reserve
        if self.rem_mamba_slots is not None:
            required_mamba_slots = mamba_gap_reserve // self._mamba_slot_cost
            if required_mamba_slots > self.rem_mamba_slots:
                if (
                    isinstance(self.tree_cache, N.UnifiedRadixCache)
                    and self.tree_cache.buffer_pipeline is not None
                ):
                    self.tree_cache.buffer_pipeline.defer_staged_admission(req, "mamba")
                return N.AddReqResult.NO_TOKEN
        request_slot_reserve = self._request_slot_reserve_for_req(req)
        total_tokens += request_slot_reserve

        # The temporary pin excludes this prefix from the evictable budget.
        # Selection itself neither allocates slots nor materializes host hits.
        with self._lock_node(req.last_node):
            admission = self._select_prefill_admission(
                req,
                total_tokens=total_tokens,
                host_hit_length=req.host_hit_length,
                swa_host_hit_length=req.swa_host_hit_length,
                truncation_align_size=truncation_align_size,
                has_chunked_req=has_chunked_req,
            )
            if isinstance(admission, N.AddReqResult):
                if admission is N.AddReqResult.NO_TOKEN:
                    self._defer_staged_for_capacity_if_idle(req, pool="prefill_budget")
                return admission

            # A rejected candidate must not report prefillable or queue H2D.
            if (self.prefill_delayer_single_pass is not None) and (
                not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=self.running_batch.batch_size(),
                    max_prefill_bs=self.max_prefill_bs,
                    max_running_requests=self.max_running_requests,
                    waiting_queue_len=self.waiting_queue_len,
                )
            ):
                return N.AddReqResult.OTHER

            if req.needs_host_load_back():
                load_max_new = min(max_new, admission.max_new_tokens)
                # Reclaim can write back device victims and evict host leaves.
                # Pin the selected host/aux match until load-back owns its locks.
                with (
                    self._lock_node(req.best_match_node, lock_host=True)
                    if isinstance(self.tree_cache, N.UnifiedRadixCache)
                    else N.nullcontext()
                ):
                    full_load_tokens = req.host_hit_length
                    if (
                        self.memory_budget.load_back_needs_full_tokens
                        and isinstance(self.tree_cache, N.UnifiedRadixCache)
                        and self.tree_cache.buffer_pipeline is None
                        and not (
                            self.tree_cache.linker is not None
                            and self.tree_cache.linker.has_hit(req.rid)
                        )
                    ):
                        # Host hits can include resident FULL behind host-only aux.
                        # Reuse the FULL transfer spec to count only new slots.
                        full_transfer = (
                            self.tree_cache.tree_core.build_hicache_transfers(
                                N.ComponentType.FULL,
                                req.best_match_node,
                                N.CacheTransferPhase.LOAD_BACK,
                            )[0]
                        )
                        full_load_tokens = len(full_transfer.host_indices)
                    if not self.memory_budget.prepare_load_back(
                        full_tokens=(
                            full_load_tokens
                            + admission.extend_len
                            + load_max_new
                            + self.page_size
                            + mamba_gap_reserve
                        ),
                        extend_input_len=admission.extend_len,
                        max_new_tokens=load_max_new,
                        swa_host_hit_length=req.swa_host_hit_length,
                        chunk_limit=self.rem_chunk_tokens,
                    ):
                        self._defer_staged_for_capacity_if_idle(
                            req, pool="load_back_budget"
                        )
                        return N.AddReqResult.NO_TOKEN
                    promised_host_hit = req.host_hit_length
                    loaded = self.tree_cache.init_load_back(
                        N.InitLoadBackParams(
                            best_match_node=req.best_match_node,
                            host_hit_length=req.host_hit_length,
                            req=req,
                        )
                    )
                if loaded is None:
                    return N.AddReqResult.OTHER
                req.host_loaded_length, req.last_node = loaded
                if 0 < req.host_loaded_length < promised_host_hit:
                    raise RuntimeError(
                        "HiCache load-back must commit all promised FULL tokens or none: "
                        f"req={req.rid} promised={promised_host_hit} "
                        f"loaded={req.host_loaded_length}"
                    )
                if req.host_loaded_length > promised_host_hit:
                    # A load can expose resident FULL behind host-only aux state; its
                    # H2D is queued, so keep the approved budget and shrink the work.
                    prefix_len = req.prefix_len + req.host_loaded_length
                    extend_len = admission.extend_len
                    if self.dllm_config is None:
                        extend_len = min(
                            extend_len, len(req.full_untruncated_fill_ids) - prefix_len
                        )
                    is_chunked = admission.is_chunked and (
                        prefix_len + extend_len < len(req.full_untruncated_fill_ids)
                    )
                    max_new_tokens = admission.max_new_tokens
                    if admission.is_chunked and not is_chunked:
                        max_new_tokens = min(
                            req.sampling_params.max_new_tokens, N.CLIP_MAX_NEW_TOKENS
                        )
                    admission = _PrefillAdmission(
                        prefix_len,
                        extend_len,
                        max_new_tokens,
                        is_chunked,
                        self._replay_len(req, prefix_len),
                    )
                elif req.host_loaded_length < promised_host_hit:
                    # No FULL was loaded; recomputation may no longer fit.
                    admission = self._select_prefill_admission(
                        req,
                        total_tokens=total_tokens,
                        host_hit_length=0,
                        swa_host_hit_length=0,
                        truncation_align_size=truncation_align_size,
                        has_chunked_req=has_chunked_req,
                    )
                    if isinstance(admission, N.AddReqResult):
                        return admission
                req.prefix_len += req.host_loaded_length
                if self.tree_cache.load_back_is_cache_owned():
                    req.kv.cache_protected_len = req.prefix_len

            # Sharded pools cannot load host KV; reserve scratch after all other gates.
            if not self._kv_shard_reserve_scratch(
                prefix_len=admission.prefix_len, extend_len=admission.extend_len
            ):
                return N.AddReqResult.OTHER

            self._commit_prefill_admission(req, admission, mamba_gap_reserve)

        # This verdict controls the next candidate, not the committed request.
        return self.budget_state()

    def _select_prefill_admission(
        self,
        req: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: Optional[int],
        has_chunked_req: bool = False,
    ) -> _PrefillAdmission | N.AddReqResult:
        """Select a prefill shape without allocating or publishing cached KV."""
        prefix_len = req.prefix_len + host_hit_length
        extend_len = len(req.full_untruncated_fill_ids) - prefix_len
        input_tokens = self.ceil_paged_tokens(extend_len)
        # Whether the request fits whole. Against the raw length under
        # exact-chunk-fill, so a request whose ceiled length would spill is
        # not needlessly split into a second chunk.
        chunk_fit_tokens = extend_len if self.exact_chunk_fill else input_tokens
        replay_len = self._replay_len(req, prefix_len)
        chunk_limit = self.rem_chunk_tokens
        if chunk_limit is not None and replay_len:
            # The replay rows share the chunk with the extend.
            chunk_limit -= replay_len
            if chunk_limit <= 0:
                return N.AddReqResult.OTHER
        can_admit, chunk_tokens_limit = self._check_prefill_budget(
            req,
            extend_input_len=extend_len,
            total_tokens=total_tokens,
            swa_host_hit_length=swa_host_hit_length,
            chunk_limit=chunk_limit,
        )
        if not can_admit:
            return N.AddReqResult.NO_TOKEN

        # Without chunking, allow the first request even above the input cap.
        if (
            self.rem_chunk_tokens is None
            and self.can_run_list
            and input_tokens >= self.rem_input_tokens
        ):
            return N.AddReqResult.OTHER

        is_chunked = False
        max_new_tokens = min(req.sampling_params.max_new_tokens, N.CLIP_MAX_NEW_TOKENS)
        tile_tokens = input_tokens
        if self.dllm_config is not None:
            assert truncation_align_size is None, (
                "truncation_align_size is not supported for dllm prefill"
            )
            extend_len = self._get_dllm_extend_len(req, prefix_len)
            if extend_len <= 0:
                return N.AddReqResult.OTHER
            max_new_tokens = 0
        elif chunk_tokens_limit is not None and chunk_fit_tokens > chunk_tokens_limit:
            if (
                has_chunked_req
                and N.get_schedule().schedule_policy == "shortest-prefill-first"
            ):
                # Only one unfinished chunked request can be tracked.
                return N.AddReqResult.OTHER
            if self.exact_chunk_fill:
                # Take the remainder verbatim so the batch hits exactly
                # chunked_prefill_size. `chunk_fit_tokens > chunk_tokens_limit`
                # here, so this never runs past the end of the prompt. Uses the
                # limit rather than rem_chunk_tokens so an SWA-capped chunk stays
                # capped.
                extend_len = chunk_tokens_limit
                if truncation_align_size is not None:
                    extend_len = (
                        extend_len // truncation_align_size * truncation_align_size
                    )
            else:
                extend_len = chunk_tokens_limit // self.page_size * self.page_size
                if truncation_align_size is not None:
                    extend_len = (
                        extend_len // truncation_align_size * truncation_align_size
                    )
                end = (prefix_len + extend_len) // self.page_size * self.page_size
                extend_len = end - prefix_len
            if extend_len <= 0:
                return N.AddReqResult.OTHER
            is_chunked = True
            max_new_tokens = 0
            tile_tokens = extend_len

        if (verdict := self._check_prefill_tile_budget(tile_tokens)) is not None:
            return verdict

        return _PrefillAdmission(
            prefix_len, extend_len, max_new_tokens, is_chunked, replay_len
        )

    def _commit_prefill_admission(
        self, req: Req, admission: _PrefillAdmission, mamba_gap_reserve: int
    ) -> None:
        assert req.prefix_len == admission.prefix_len
        req.extend_end = admission.prefix_len + admission.extend_len
        req.lock = self.tree_cache.lock(req.last_node)
        self.can_run_list.append(req)
        if admission.is_chunked:
            self.new_chunked_req = req
        self._update_prefill_budget(
            admission.prefix_len,
            admission.extend_len,
            admission.max_new_tokens,
            req.retracted_stain,
            mamba_gap_reserve=mamba_gap_reserve,
            request_slot_reserve=self._request_slot_reserve_for_req(
                req, extend_input_len=admission.extend_len
            ),
            replay_len=admission.replay_len,
        )
        self._account_prefill_cache_admission(req, admission.prefix_len)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.schedule_policy.PrefillAdder", PrefillAdder, HookType.REPLACE)
