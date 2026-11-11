"""Current HCU DSV4 capacity model; shares official sizing infrastructure."""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch
from sglang.srt.model_executor.pool_configurator import (
    DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO, MemoryPoolConfig, MemoryPoolConfigurator,
    _DSV4PoolSizes, _get_dsv4_compress_state_dtype_sizes,
    compute_swa_request_cap, check_dsv4_unified_fp8_pd_supported,
)
from sglang.srt.runtime_context import (get_disagg, get_exec, get_memory,
    get_parallel, get_schedule, get_spec, max_speculative_num_draft_tokens)
from sglang.srt.environ import envs
from sglang.srt.utils.common import ceil_align, ceil_div
try:
    from sglang.srt.utils.common import is_hip
except ImportError:
    from sglang.srt.utils.device import is_hip
try:
    from sglang.srt.utils.common import is_npu
except ImportError:
    from sglang.srt.utils.device import is_npu
from hcu_sglang.config import cp_cache_layer_split_enabled
from hcu_sglang.mem_cache.hcu_v4_pool import (
    collect_sources_by_ratio, get_compress_state_write_pad,
    get_draft_swa_ring_size, get_dsv4_indexer_bytes_per_token,
    get_swa_ring_size, resolve_compress_state_ring_size, uses_draft_swa_ring,
)
import logging
logger = logging.getLogger(__name__)
_is_hip = is_hip()
_is_npu = is_npu()

def _operator_swa_full_tokens_ratio() -> Optional[float]:
    """The operator's --swa-full-tokens-ratio, or None when it was not given."""
    schedule = get_schedule()
    if not schedule._swa_full_tokens_ratio_explicitly_set:
        return None
    return schedule.swa_full_tokens_ratio


class DSV4PoolConfigurator(MemoryPoolConfigurator):
    """DSV4 compressed attention: coeff is bytes_per_full_token, inflated by (T+D)/T
    for a draft worker; bias is the request-scoped pools that do not scale with it."""

    # object.__new__ stubs (SWA floor tests) skip __init__
    _dspark_draft_on_bf16 = False

    def __init__(self, kvc: KVCacheConfigurator):
        self.kv_cache_dtype_str = kvc.kv_cache_dtype_str
        cfg = kvc.model_config
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.indexer_head_dim = cfg.index_head_dim
        self.attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        from hcu_sglang.utils.env_gate_compat import (
            is_unified_kv_fp8,
            is_unified_kv_triton,
        )
        from hcu_sglang.mem_cache.hcu_v4_pool import (
            dsv4_unified_row_bytes,
        )

        # Resolve the unified-kv gate before any sizing so the two cannot drift.
        self._unified = is_unified_kv_triton()
        self._unified_fp8 = is_unified_kv_fp8()
        # DSpark draft still allocates a bf16 ring; target fp8 * (T+1)/T would
        # under-count that ring (640 vs 1024). MTP keeps the old inflation.
        self._dspark_draft_on_bf16 = bool(
            self._unified_fp8 and kvc.spec_algorithm.is_dspark()
        )
        # Row width across both unified pools: 1024 B bf16, 640 B fp8.
        self._unified_row_bytes = dsv4_unified_row_bytes(
            self.qk_nope_head_dim, self.qk_rope_head_dim, self._unified_fp8
        )
        if self._unified:
            # Unified_kv stores the whole latent: one bf16 row, or fp8 nope + bf16 rope.
            self.kv_bytes = self._unified_row_bytes
        elif get_exec().kernel.dsv4_attn_backend == "trtllm":
            self.kv_bytes = self.attn_head_dim
        else:
            # One FlashMLA-layout latent slot, in bytes.
            self.kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8
        # HIP takes the FP4-accurate byte count here. The NVIDIA FP4 path
        # keeps the FP8 estimate.
        self.indexer_bytes_per_token = get_dsv4_indexer_bytes_per_token(
            self.indexer_head_dim,
            _is_hip and get_exec().kernel.enable_deepseek_v4_fp4_indexer,
        )
        self.context_len = kvc.model_config.context_len
        # PP-local slice; matches DeepSeekV4TokenToKVPool's stage_ratios.
        stage = range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
        self.stage_compress_ratios = cfg.compress_ratios[stage.start : stage.stop]
        if kvc.pp_size > 1:
            logger.info(
                f"DSV4 pool PP slice: rank={kvc.pp_group.rank_in_group} "
                f"layers=[{stage.start},{stage.stop}) "
                f"local={len(self.stage_compress_ratios)}/{len(cfg.compress_ratios)}"
            )
        # Layers of this stage that own compressed storage, per ratio. Same rule
        # as the pool, so a ratio budgeted here is one it allocates.
        self.stage_owner_layers = collect_sources_by_ratio(
            cfg.compress_ratios, cfg.hf_config.kv_source_layer_ids, stage
        )
        self.operator_swa_ratio = _operator_swa_full_tokens_ratio()
        self.swa_ratio = (
            self.operator_swa_ratio
            if self.operator_swa_ratio is not None
            else DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO
        )
        self.sliding_window_size = kvc.sliding_window_size
        self.page_size = kvc.page_size
        self.is_speculative = get_spec().speculative_algorithm is not None
        self.online_c128_mtp_max_draft_tokens = max_speculative_num_draft_tokens() or 0
        self.attn_dp_size = kvc.attn_dp_size
        self.requested_max_running_requests_per_worker = (
            max(1, get_schedule().max_running_requests // kvc.attn_dp_size)
            if get_schedule().max_running_requests is not None
            else None
        )
        self.disaggregation_mode = get_disagg().disaggregation_mode
        self.disaggregation_decode_extra_slots = (
            get_disagg().disaggregation_decode_extra_slots or 0
        )
        if get_memory().enable_hisparse:
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            self.c4_shrink_factor = parse_hisparse_config().host_to_device_ratio
        else:
            self.c4_shrink_factor = 1
        assert self.c4_shrink_factor >= 1
        if self.c4_shrink_factor > 1:
            logger.info(f"HiSparse c4 host-to-device ratio = {self.c4_shrink_factor}")

        # Ratio 1 keeps no compress state, so it has no ring.
        self.ring_sizes = {
            ratio: resolve_compress_state_ring_size(ratio)
            for ratio in self.stage_owner_layers
            if ratio != 1
        }

        self.num_layers_total = len(self.stage_compress_ratios)
        self.use_cp_cache_layer_split = bool(
            cp_cache_layer_split_enabled()
            and not kvc.is_draft_worker
        )
        self.cp_draft_layers = 0
        if self.use_cp_cache_layer_split and kvc.spec_algorithm.is_dspark():
            # Draft KV remains replicated on every CP rank, as in the V4
            # LayerSplit implementation; only target KV is layer-sharded.
            self.cp_draft_layers = len(cfg.hf_text_config.dspark_target_layer_ids)
            assert self.cp_draft_layers > 0, "DSPARK draft layer count is required"
        self.swa_priced_layers = self.num_layers_total
        # The low-ratio indexer pools are built with force_fp4=True
        # (deepseek_v4_memory_pool), so they are fp4 whatever dtype c4 uses.
        self.low_ratio_index_bytes = get_dsv4_indexer_bytes_per_token(
            self.indexer_head_dim, use_fp4_indexer=True
        )
        if self.use_cp_cache_layer_split:
            from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_layout import (
                build_cp_cache_layer_split_deepseek_v4_pool_layout,
            )

            cp_size = get_parallel().attn_cp_size
            layouts = [
                build_cp_cache_layer_split_deepseek_v4_pool_layout(
                    rank,
                    cp_size,
                    kvc.layer_info.start_layer,
                    kvc.layer_info.end_layer,
                    cfg.compress_ratios,
                    cfg.hf_config.kv_source_layer_ids,
                )
                for rank in range(cp_size)
            ]
            self.swa_priced_layers = max(l.swa_layer_num for l in layouts) + 1
            owner_kv_bytes = max(
                (l.c1_layer_num + l.c2_layer_num / 2) * self.kv_bytes
                for l in layouts
            )
            replicated_indexer_bytes = sum(
                self.low_ratio_index_bytes / cfg.compress_ratios[l]
                for l in cfg.hf_config.kv_source_layer_ids
                if kvc.layer_info.start_layer <= l < kvc.layer_info.end_layer
                and cfg.compress_ratios[l] in (1, 2)
            )
            staging_kv_bytes = max(
                (
                    self.kv_bytes / cfg.compress_ratios[l]
                    for l in cfg.hf_config.kv_source_layer_ids
                    if kvc.layer_info.start_layer <= l < kvc.layer_info.end_layer
                    and cfg.compress_ratios[l] in (1, 2)
                ),
                default=0,
            )
            self.low_ratio_bytes_per_full_token = (
                owner_kv_bytes + replicated_indexer_bytes + staging_kv_bytes
            )

        # kvc.sliding_window_size is None on a runner without SWA layers.
        self._swa_ring_size = get_swa_ring_size(cfg.window_size, self.is_speculative)
        self._spec_infl = 1.0

        # The unified pool ignores --kv-cache-dtype; V4 resolves "auto" to fp8_e4m3
        # (overrides.py _deepseek_v4_kv_cache_dtype), so only bfloat16 is explicit.
        if self._unified_fp8 and self.kv_cache_dtype_str == "bfloat16":
            logger.warning(
                "--kv-cache-dtype=bfloat16 is ignored on the unified_kv path; "
                "SGLANG_DSV4_UNIFIED_KV_FP8=1 stores the latent as fp8. Unset the "
                "env switch to get a bf16 unified pool."
            )

        check_dsv4_unified_fp8_pd_supported(
            unified_fp8=self._unified_fp8,
            disaggregation_mode=self.disaggregation_mode,
            pp_size=kvc.pp_size,
            enable_hisparse=get_memory().enable_hisparse,
        )

        if self.is_speculative:
            # Ring is sized once here, so it must serve the largest adaptive tier.
            self._assert_ring_serves_draft_tokens(
                max_speculative_num_draft_tokens() or 0
            )

        self.encoder_replay = get_exec().features.enable_encoder_swa_bounded_replay
        draft_layers = 0
        if self.encoder_replay and kvc.spec_algorithm.is_dspark():
            aux = kvc.spec_aux_config
            draft_layers = int(
                aux.dspark_num_stages or aux.dflash_draft_num_layers or 0
            )
            assert draft_layers > 0, "DSpark draft layer count is required"
        # A draft SWA ring is budgeted with the request windows, not per SWA token.
        draft_ring_layers = draft_layers if uses_draft_swa_ring() else 0
        self.paged_draft_layers = draft_layers - draft_ring_layers
        self.request_window_bytes = 0
        if self.encoder_replay:
            slots = self.requested_max_running_requests_per_worker + 1
            capacity = ceil_align(
                self.sliding_window_size + self.online_c128_mtp_max_draft_tokens,
                self.page_size,
            )
            layers = self.num_layers_total
            scratch = (
                max(
                    get_schedule().chunked_prefill_size,
                    slots * max(128, self.online_c128_mtp_max_draft_tokens),
                )
                + slots * 128
                + self.page_size
            )
            draft_ring_size = get_draft_swa_ring_size(
                self.sliding_window_size,
                self.online_c128_mtp_max_draft_tokens,
                self.page_size,
            )
            self.request_window_bytes = (
                (slots * capacity + self.page_size) * layers * (self.kv_bytes + 16)
                + 4 * scratch * (self.kv_bytes + 16)
                + slots * 3 * 16 * self.attn_head_dim * 8
                + (slots * draft_ring_size + self.page_size)
                * draft_ring_layers
                * self.kv_bytes
            )
            if not self.paged_draft_layers:
                self.swa_ratio = 0
        self.swa_prefix_tails = self._resolve_swa_prefix_tails()
        self.swa_cap_tokens = (
            0
            if self.encoder_replay and not self.paged_draft_layers
            else self._resolve_swa_cap_tokens()
        )
        self.bytes_per_swa_token = self._get_bytes_per_swa_token()
        self.bytes_per_full_token = self._get_bytes_per_full_token()
        if self.use_cp_cache_layer_split:
            logger.info(
                "CP Cache LayerSplit KV pricing: cp_size=%d, target_layers=%d, "
                "swa_layers=%d, draft_layers=%d, kv_bytes=%d, swa_cap=%s, "
                "bytes_per_swa_token=%.2f",
                get_parallel().attn_cp_size,
                self.num_layers_total,
                self.swa_priced_layers,
                self.cp_draft_layers,
                self.kv_bytes,
                self.swa_cap_tokens,
                self.bytes_per_swa_token,
            )
        if (
            self.is_speculative
            and not self.encoder_replay
            and not self.use_cp_cache_layer_split
        ):
            # Reserve memory for the speculative draft worker by inflating
            # per-token bytes by (target+draft)/target. Equivalent to dflash's
            # scale_kv_cell_size_per_token_for_dflash but applied to
            # bytes_per_full_token: tokens = avail / (bpft * (T+D)/T).
            draft_layers = 1
            target_layers = self.num_layers_total
            self._spec_infl = (target_layers + draft_layers) / target_layers
            self.bytes_per_full_token *= self._spec_infl
            self.bytes_per_swa_token *= self._spec_infl

        # Online c128 keeps one in-progress (max, sum, kv) state per index and
        # assumes forward-only; MTP would need rollback across draft and verify.
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            allow_experimental_online_c128_mtp = (
                envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
                and kvc.spec_algorithm.is_eagle()
            )
            assert kvc.spec_algorithm.is_none() or allow_experimental_online_c128_mtp, (
                "SGLANG_OPT_USE_ONLINE_COMPRESS does not support speculative decode "
                "(MTP) yet, except the experimental EAGLE topk=1 path gated by "
                "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1"
            )
            if allow_experimental_online_c128_mtp:
                assert self.online_c128_mtp_max_draft_tokens > 0, (
                    "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP requires "
                    "speculative_num_draft_tokens to be set."
                )
                logger.warning(
                    "DSV4 compressed attention: experimental online c128 + MTP enabled "
                    f"(EAGLE topk=1 only, "
                    f"draft_banks={self.online_c128_mtp_max_draft_tokens}). "
                    "Validate correctness carefully."
                )
            else:
                logger.info(
                    "DSV4 compressed attention: online c128 enabled (ring_size=1)"
                )

    def num_layers(self, ratio: int) -> int:
        """Layers of this stage owning ratio's compressed storage."""
        return len(self.stage_owner_layers.get(ratio, ()))

    def _assert_ring_serves_draft_tokens(self, num_draft_tokens: int) -> None:
        """A verify batch writes its whole optimistic tail into the ring, so ring
        capacity bounds the draft count."""
        for compress_ratio in (4, 128):
            if self.num_layers(compress_ratio) == 0:
                continue
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                # Online c128 keeps per-draft state instead of a ring; sized separately.
                continue
            ring_size = self.ring_sizes[compress_ratio]
            max_draft_tokens = get_compress_state_write_pad(compress_ratio, ring_size)
            assert num_draft_tokens <= max_draft_tokens, (
                f"speculative_num_draft_tokens={num_draft_tokens} exceeds what the c{compress_ratio} "
                f"compress state ring can keep resident (ring_size={ring_size} serves at most "
                f"{max_draft_tokens} draft tokens). Lower the draft count, or grow the ring in "
                f"get_compress_state_ring_size()."
            )

    def _resolve_swa_prefix_tails(self) -> int:
        """Cached prefix tails cap mode keeps addressable: a prefix is reusable only
        while its last sliding_window tokens still hold SWA slots."""
        prefix_tails = get_schedule().swa_prefix_tails
        if prefix_tails is not None:
            return prefix_tails
        if get_memory().disable_radix_cache:
            # Nothing is kept for reuse, so the request cap alone bounds the pool.
            return 0
        max_running_requests = self.requested_max_running_requests_per_worker
        if max_running_requests is None:
            return 0
        tails = 4 * max_running_requests
        if self.encoder_replay and self.paged_draft_layers:
            # A replay tail pays for draft layers only; spend the non-replay
            # default's bytes on proportionally more tails.
            tails *= max(1, self.num_layers_total // self.paged_draft_layers)
        return tails

    def _resolve_swa_cap_tokens(self) -> Optional[int]:
        """SWA slots to reserve in cap mode, None to keep ratio sizing. Cap mode
        budgets from the request cap plus radix headroom, not full_tokens."""
        if self.operator_swa_ratio is not None:
            return None
        if self._unified:
            # Ring mode: SWA is a fixed per-request ring, with no paged pool to size.
            return None
        max_running_requests = self.requested_max_running_requests_per_worker
        if max_running_requests is None or self.sliding_window_size is None:
            return None
        chunked_prefill_size = get_schedule().chunked_prefill_size
        if self.disaggregation_mode != "decode" and (
            chunked_prefill_size is None or chunked_prefill_size <= 0
        ):
            return None

        cap = compute_swa_request_cap(
            page_size=self.page_size,
            window=self.sliding_window_size,
            attn_dp_size=self.attn_dp_size,
        )
        headroom = self.swa_prefix_tails * (self.sliding_window_size + self.page_size)
        return ceil_align(cap + headroom, self.page_size)

    def _get_paged_kv_bytes_per_token(self, compress_ratio: int = 0) -> float:
        # Unified rings, the NPU pool and the trtllm uniform-FP8 pool do not go
        # through DeepSeekV4SingleKVPool.create_buffer, so they carry no page pad.
        if self._unified or _is_npu or get_exec().kernel.dsv4_attn_backend == "trtllm":
            return self.kv_bytes
        from hcu_sglang.mem_cache.hcu_v4_pool import (
            resolve_compressed_kv_layout,
            select_dsv4_kv_layout,
        )

        layout, compressed_option = select_dsv4_kv_layout()
        page_size = self.page_size
        if compress_ratio:
            layout = resolve_compressed_kv_layout(
                layout, compress_ratio, compressed_option
            )
            page_size = self.page_size // compress_ratio
        return layout.page_bytes(page_size) / page_size

    def _get_bytes_per_swa_token(self) -> float:
        """Bytes one SWA slot costs across the stage. c4_state_pool_size = swa_tokens
        // page_size * ring, so c4 compress state is priced per SWA slot too."""
        if self.encoder_replay:
            # Target SWA lives in the request window; only the draft owns paged SWA
            # bytes, and its layers carry no compressed state.
            return self.kv_bytes * self.paged_draft_layers
        c4_state_dtype_size, _ = _get_dsv4_compress_state_dtype_sizes()
        c4_state_bytes = 2 * 2 * self.attn_head_dim * c4_state_dtype_size
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * c4_state_dtype_size

        c4_state_ratio = self.ring_sizes.get(4, 0) / self.page_size
        return (
            self._get_paged_kv_bytes_per_token()
            * (self.swa_priced_layers + self.cp_draft_layers)
            + c4_state_ratio
            * (c4_state_bytes + c4_indexer_state_bytes)
            * self.num_layers(4)
        )

    def _compressed_bytes_per_full_token(self, ratio: int) -> float:
        """Compressed KV (+ indexer) bytes one full token adds per layer of `ratio`;
        compress state is priced per SWA slot (c4) or as fixed bytes (c128, ratio 2)."""
        if ratio in (1, 2):
            return (self.kv_bytes + self.low_ratio_index_bytes) / ratio
        if ratio == 4:
            c4_frac = 1 / (4 * self.c4_shrink_factor)
            return (
                c4_frac * self._get_paged_kv_bytes_per_token(4)
                + 1 / 4 * self.indexer_bytes_per_token
            )
        assert ratio == 128, f"unsupported compression ratio: {ratio}"
        return 1 / 128 * self._get_paged_kv_bytes_per_token(128)

    def _get_bytes_per_full_token(self) -> float:
        # Cap mode and ring mode both move the SWA pool and the c4 state that
        # follows it out of the coefficient and into fixed bytes.
        swa_ratio = (
            0 if self._unified or self.swa_cap_tokens is not None else self.swa_ratio
        )
        compressed_bytes = sum(
            self._compressed_bytes_per_full_token(ratio) * len(layers)
            for ratio, layers in self.stage_owner_layers.items()
            if not (self.use_cp_cache_layer_split and ratio in (1, 2))
        )
        if self.use_cp_cache_layer_split:
            compressed_bytes += self.low_ratio_bytes_per_full_token
        return swa_ratio * self.bytes_per_swa_token + compressed_bytes

    def _get_swa_fixed_bytes(self) -> float:
        """Bias bytes of the encoder-replay request window plus the cap-mode pool."""
        paged_bytes = (
            0
            if self.swa_cap_tokens is None
            else self.swa_cap_tokens * self.bytes_per_swa_token
        )
        return self.request_window_bytes + paged_bytes

    def _get_swa_tokens(self, full_token: int, page_size: int) -> int:
        # swa_cap_tokens was already page-aligned at resolve time.
        if self.swa_cap_tokens is None:
            return int(full_token * self.swa_ratio) // page_size * page_size
        return self.swa_cap_tokens

    def _compute_dsv4_sizes(self, full_token: int, page_size: int) -> _DSV4PoolSizes:
        full_token = full_token // page_size * page_size
        swa_tokens = self._get_swa_tokens(full_token, page_size)
        if self.swa_cap_tokens is None:
            # Only ratio sizing can under-size a request: cap mode sizes from the
            # request floor, and encoder replay deliberately runs swa_tokens == 0.
            if not self._unified:
                self.validate_swa_pool_size(
                    swa_tokens, self.sliding_window_size, page_size
                )
            source = "explicit" if self.operator_swa_ratio is not None else "default"
            mode = (
                "ring (paged swa_tokens vestigial)"
                if self._unified
                else f"ratio ({source})"
            )
            logger.info(
                f"DSV4 SWA sizing: mode={mode}, swa_tokens={swa_tokens}, "
                f"swa_full_tokens_ratio={self.swa_ratio}"
            )
        else:
            logger.info(
                f"DSV4 SWA sizing: mode=cap, swa_tokens={swa_tokens}, "
                f"request_cap+headroom={self.swa_cap_tokens}, "
                f"prefix_tails={self.swa_prefix_tails}"
            )
        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            # Unified_kv: request-scoped, finalized once concurrency is known.
            # Otherwise the ring is addressed per SWA page (swa_loc // page_size).
            c4_state_pool_size=(
                0
                if self._unified
                else swa_tokens // page_size * self.ring_sizes.get(4, 0)
            ),
            c128_state_pool_size=0,
        )

    def _get_num_req_slots(self, max_running_requests: int) -> int:
        if self.disaggregation_mode == "decode":
            return max_running_requests + self.disaggregation_decode_extra_slots + 1
        return max_running_requests + 1

    def _get_c128_state_fixed_bytes(self, max_running_requests: int) -> int:
        num_layers = self.num_layers(128)
        if num_layers == 0:
            return 0

        _, c128_state_dtype_size = _get_dsv4_compress_state_dtype_sizes()
        num_req_slots = self._get_num_req_slots(max_running_requests)
        ring_size = self.ring_sizes[128]

        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            state_rows = num_req_slots + ring_size + 1
            state_rows *= 1 + self.online_c128_mtp_max_draft_tokens
            state_last_dim = 3 * self.attn_head_dim
        else:
            state_pool_size = num_req_slots * ring_size
            state_rows = state_pool_size + ring_size + 1
            state_rows = ceil_div(state_rows, 128) * 128
            state_last_dim = 2 * self.attn_head_dim

        return state_rows * state_last_dim * c128_state_dtype_size * num_layers

    def _get_c2_state_fixed_bytes(self, max_running_requests: int) -> int:
        """Ratio-2 pending-pair state, one fp32 (kv, score) ring per request slot
        on each ratio-2 kv_source layer; mirrors _make_pair_state_pool."""
        num_layers = self.num_layers(2)
        if num_layers == 0:
            return 0

        num_req_slots = self._get_num_req_slots(max_running_requests)
        ring_size = self.ring_sizes[2]
        # CompressStatePool allocates `size + ring_size + 1` rows, padded to the ratio.
        state_rows = num_req_slots * ring_size + ring_size + 1
        state_rows = ceil_div(state_rows, 2) * 2
        state_last_dim = 2 * self.attn_head_dim
        return state_rows * state_last_dim * torch.float32.itemsize * num_layers

    def _unified_c4_state_pool_size(self, max_running_requests: int) -> int:
        # Unified C4 state loc is req_pool_idx * c4_ring_size + pos % c4_ring_size.
        num_req_slots = self._get_num_req_slots(max_running_requests)
        return num_req_slots * self.ring_sizes[4]

    def _fixed_c4_state_bytes(self, max_running_requests: int) -> int:
        num_layers = self.num_layers(4)
        if not self._unified or num_layers == 0:
            return 0

        c4_state_dtype_size, _ = _get_dsv4_compress_state_dtype_sizes()
        # Mirror CompressStatePool.__init__: it allocates `size + ring_size + 1`
        # rows, padded to the compress ratio.
        state_rows = self._unified_c4_state_pool_size(max_running_requests)
        state_rows = ceil_div(state_rows + self.ring_sizes[4] + 1, 4) * 4
        # overlap c4: last_dim = 2 * (1 + overlap) * head_dim = 4 * head_dim.
        core_bytes = 4 * self.attn_head_dim * c4_state_dtype_size
        indexer_bytes = 4 * self.indexer_head_dim * c4_state_dtype_size
        return state_rows * (core_bytes + indexer_bytes) * num_layers

    def _resolve_max_running_requests_per_worker(self, available_bytes: int) -> int:
        # Approximates ModelRunner._resolve_max_num_reqs. Over-estimating is safe:
        # a larger fixed bias yields a smaller full_token.
        if self.requested_max_running_requests_per_worker is not None:
            return self.requested_max_running_requests_per_worker

        full_token = int(available_bytes / self.bytes_per_full_token)
        estimated = int(full_token / self.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)
        return min(estimated, full_token // 2)

    def _fixed_swa_bytes(self, max_running_requests: int) -> int:
        """Unified_kv SWA ring bytes, sized by req slots; 0 off unified, where SWA is
        priced per token. DSpark+fp8 adds a bf16 draft ring instead of _spec_infl."""
        if not self._unified:
            return 0
        num_req_slots = self._get_num_req_slots(max_running_requests)
        target_ring = (
            num_req_slots
            * self._swa_ring_size
            * self._unified_row_bytes
            * self.num_layers_total
        )
        if self._dspark_draft_on_bf16:
            from hcu_sglang.mem_cache.hcu_v4_pool import (
                dsv4_unified_row_bytes,
            )

            draft_row = dsv4_unified_row_bytes(
                self.qk_nope_head_dim, self.qk_rope_head_dim, fp8=False
            )
            # 1 layer is what shipped DSpark drafts allocate; a multi-layer draft
            # under-counts ~9 MB/layer, covered by the (T+1)/T inflation.
            draft_ring = num_req_slots * self._swa_ring_size * draft_row
            return int(target_ring + draft_ring)
        return int(target_ring * self._spec_infl)

    def _to_config(self, sizes: _DSV4PoolSizes) -> MemoryPoolConfig:
        full = sizes.full_max_total_num_tokens
        swa = sizes.swa_max_total_num_tokens
        logger.info(
            f"DSV4 pool sizes: full={full}, swa={swa}, "
            f"c4={sizes.c4_max_total_num_tokens}, "
            f"c128={sizes.c128_max_total_num_tokens}, "
            f"c4_state={sizes.c4_state_pool_size}, "
            f"c128_state={sizes.c128_state_pool_size}"
        )
        return MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            c4_max_total_num_tokens=sizes.c4_max_total_num_tokens,
            c128_max_total_num_tokens=sizes.c128_max_total_num_tokens,
            c4_state_pool_size=sizes.c4_state_pool_size,
            c128_state_pool_size=sizes.c128_state_pool_size,
        )

    def finalize_with_max_running_requests(
        self, config: MemoryPoolConfig
    ) -> MemoryPoolConfig:
        assert config.max_running_requests is not None
        num_req_slots = self._get_num_req_slots(config.max_running_requests)
        if envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            config.c128_state_pool_size = num_req_slots
        else:
            config.c128_state_pool_size = num_req_slots * self.ring_sizes.get(128, 0)
        # Ring mode: C4 state is request-scoped, so size it from the known concurrency.
        if self._unified and self.num_layers(4) > 0:
            config.c4_state_pool_size = self._unified_c4_state_pool_size(
                config.max_running_requests
            )
        return config

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        assert page_size % 128 == 0, (
            "page_size must be multiple of 128 for compressed attention"
        )

        max_running_requests_per_worker = self._resolve_max_running_requests_per_worker(
            available_bytes
        )
        c128_state_fixed_bytes = self._get_c128_state_fixed_bytes(
            max_running_requests_per_worker
        )
        c2_state_fixed_bytes = self._get_c2_state_fixed_bytes(
            max_running_requests_per_worker
        )
        swa_ring_fixed_bytes = self._fixed_swa_bytes(max_running_requests_per_worker)
        c4_state_fixed_bytes = self._fixed_c4_state_bytes(
            max_running_requests_per_worker
        )

        swa_fixed_bytes = self._get_swa_fixed_bytes()
        fixed_bytes = (
            c128_state_fixed_bytes
            + c2_state_fixed_bytes
            + swa_fixed_bytes
            + swa_ring_fixed_bytes
            + c4_state_fixed_bytes
        )
        available_bytes_for_tokens = max(available_bytes - fixed_bytes, 0)
        full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)
        if full_token <= 0 and self.swa_cap_tokens is not None:
            raise RuntimeError(
                f"The DSV4 SWA pool cap ({self.swa_cap_tokens} tokens, "
                f"{swa_fixed_bytes / (1 << 30):.2f} GB) leaves no room for the full "
                f"KV pool within the available {available_bytes / (1 << 30):.2f} GB. "
                f"Reduce --max-running-requests, lower --swa-prefix-tails "
                f"or SGLANG_SWA_EVICTION_INTERVAL, or increase --mem-fraction-static."
            )

        sizes = self._compute_dsv4_sizes(full_token, page_size)
        logger.info(
            f"DSV4 memory calculation: unified={self._unified}, "
            f"unified_fp8={self._unified_fp8}, "
            f"dspark_draft_bf16={self._dspark_draft_on_bf16}, "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"c128_state_fixed={c128_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"c2_state_fixed={c2_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"swa_fixed={swa_fixed_bytes / (1 << 30):.2f} GB, "
            f"swa_ring_fixed={swa_ring_fixed_bytes / (1 << 30):.2f} GB, "
            f"c4_state_fixed={c4_state_fixed_bytes / (1 << 30):.2f} GB, "
            f"full_token={sizes.full_max_total_num_tokens}"
        )
        return self._to_config(sizes)

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        # Token count, not a byte budget: the fixed pools are not re-subtracted, so
        # the input must not exceed what calculate_pool_sizes derived for it.
        assert page_size % 128 == 0, (
            "page_size must be multiple of 128 for compressed attention"
        )
        sizes = self._compute_dsv4_sizes(max_total_num_tokens, page_size)
        return self._to_config(sizes)

