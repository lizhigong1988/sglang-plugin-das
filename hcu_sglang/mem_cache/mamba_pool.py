"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

from __future__ import annotations

def __init__(
    self,
    *,
    size: int,
    spec_state_size: int,
    cache_params: BaseLinearStateParams,
    mamba_layer_ids: List[int],
    device: str,
    enable_memory_saver: bool = False,
    speculative_num_draft_tokens: Optional[int] = None,
    speculative_eagle_topk: Optional[int] = None,
    enable_linear_replayssm: bool = False,
    linear_replayssm_cache_len: int = 16,
    envelope_layout: bool = False,
    enable_linear_replayssm_spec: bool = False,
    replayssm_request_ring: Optional[object] = None,  # 1be97a0; unused while replayssm is off
    **_upstream_extra,  # tolerate kwargs added by newer upstream pins
):
    conv_state_shape = cache_params.shape.conv
    temporal_state_shape = cache_params.shape.temporal
    conv_dtype = cache_params.dtype.conv
    ssm_dtype = cache_params.dtype.temporal
    self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=enable_memory_saver
    )
    num_mamba_layers = len(mamba_layer_ids)
    self.mamba_layer_ids = list(mamba_layer_ids)

    self.size = size
    self.device = device
    self.debug_memory_pool = envs.SGLANG_DEBUG_MEMORY_POOL.get()
    self.enable_linear_replayssm = enable_linear_replayssm
    # 1be97a0: native hybrid backend's _replayssm_request_ring() reads this
    # attribute on the state pool; mirror the upstream default when absent.
    self.replayssm_request_ring = bool(
        replayssm_request_ring if replayssm_request_ring is not None else False
    )
    self.linear_replayssm_cache_len = linear_replayssm_cache_len
    # ReplaySSM: the decode ring (--enable-linear-replayssm) allocates the
    # chunked (d, k) records + write_pos; the spec-verify flag
    # (--enable-linear-replayssm-spec) uses compact replay for GDN and raw
    # fold-every-commit for KDA. The shared g allocation gates on
    # `_replayssm_on`.
    self.enable_linear_replayssm_spec = enable_linear_replayssm_spec
    self.use_hcu_kda = cache_params.use_hcu_kda
    self.use_hcu_kda_replayssm = bool(
        enable_linear_replayssm_spec and self.use_hcu_kda
    )
    if self.use_hcu_kda_replayssm and ssm_dtype != torch.float32:
        raise ValueError("GLM5-Next ReplaySSM requires float32 Mamba state")
    self.replayssm_spec_fold = bool(
        enable_linear_replayssm_spec and cache_params.is_kda
    )
    _replayssm_on = enable_linear_replayssm or enable_linear_replayssm_spec

    # for disagg with nvlink
    self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
        maybe_init_custom_mem_pool(device=self.device)
    )

    with (
        self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
        (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ),
    ):
        if envelope_layout:
            # Page-granularity envelope layout (page_size==1 for state): all
            # mamba layers/slots share one contiguous byte buffer; conv and
            # temporal are strided views into it (see mem_cache/layout/
            # token_major.py). Only the standard CUDA Triton path is supported.
            assert not _is_npu and not (_is_cpu and _cpu_has_amx_support), (
                "envelope_layout mamba is only supported on the CUDA path"
            )
            max_slots = size + 1
            entry_bytes = mamba_entry_bytes(
                layer_num=num_mamba_layers,
                conv_state_shapes=conv_state_shape,
                conv_dtype=conv_dtype,
                temporal_state_shape=temporal_state_shape,
                temporal_dtype=ssm_dtype,
            )
            self._raw = torch.zeros(
                max_slots * entry_bytes, dtype=torch.uint8, device=device
            )
            conv_state, temporal_state = build_mamba_entry_views(
                self._raw,
                layer_num=num_mamba_layers,
                conv_state_shapes=conv_state_shape,
                conv_dtype=conv_dtype,
                temporal_state_shape=temporal_state_shape,
                temporal_dtype=ssm_dtype,
                max_slots=max_slots,
            )
        else:
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    _init_npu_conv_state,
                )

                conv_state = _init_npu_conv_state(
                    conv_state[0],
                    conv_state_shape,
                    speculative_num_draft_tokens,
                    is_kda=cache_params.is_kda,
                )

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            temporal_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )

        # GDN ReplaySSM ring buffers (slice 1a). Allocated only when the
        # flag is on; otherwise left as None so the legacy State is
        # byte-identical. temporal_state_shape == (HV, V, K). Either the decode
        # ring (--enable-linear-replayssm) or the spec-verify ring
        # (--enable-linear-replayssm-spec) shares this allocation.
        replayssm_d = replayssm_k = replayssm_g = None
        replayssm_rawv = replayssm_rawk = replayssm_beta = None
        if _replayssm_on:
            hv, v_dim, k_dim = temporal_state_shape
            h_k = getattr(cache_params.shape, "num_k_heads_per_tp", hv)
            L = linear_replayssm_cache_len
            # GDN speculative replay is request-lifetime scratch. Size it by
            # active requests instead of every persistent radix-cache slot.
            num_slots = (
                spec_state_size + 1
                if enable_linear_replayssm_spec and not cache_params.is_kda
                else size + 1
            )
            # Decode records follow the SSM dtype. Spec-verify compact d/k
            # records follow the activation dtype; g stays fp32.
            ring_dtype = conv_dtype if enable_linear_replayssm_spec else ssm_dtype
            # Fold-every-commit: one verify window, no chunked (d, k)
            # records. KDA is the exception on both counts: its window
            # stays L-sized (the fused verify ring-write drops
            # absorb-inflated rows past L), and d/k stay allocated --
            # forward_decode routes on `replayssm_d is None` (fused vs
            # decode-ring), so skipping them would flip KDA decode to the
            # fused path, a behavior change needing its own validation
            # (memory follow-up).
            if self.use_hcu_kda_replayssm or (
                self.replayssm_spec_fold and not cache_params.is_kda
            ):
                record_len = (
                    speculative_num_draft_tokens
                    if speculative_num_draft_tokens is not None
                    else L
                )
            else:
                record_len = L
            if not self.replayssm_spec_fold or (
                cache_params.is_kda and not self.use_hcu_kda_replayssm
            ):
                replayssm_d = torch.zeros(
                    size=(num_mamba_layers, num_slots, hv, L, v_dim),
                    dtype=ring_dtype,
                    device=device,
                )
                replayssm_k = torch.zeros(
                    size=(num_mamba_layers, num_slots, h_k, L, k_dim),
                    dtype=ring_dtype,
                    device=device,
                )
            # The log-decay gate ring (fp32): per-head SCALAR for the GDN
            # gate -> [.., record_len]; per-K VECTOR for the KDA gate ->
            # [.., record_len, K] (k_dim == temporal_state_shape[-1] for both).
            g_shape = (
                (num_mamba_layers, num_slots, hv, record_len, k_dim)
                if cache_params.is_kda
                else (num_mamba_layers, num_slots, hv, record_len)
            )
            replayssm_g = torch.zeros(
                size=g_shape,
                dtype=torch.float32,
                device=device,
            )
            # KDA still uses raw-input fold-every-commit. GDN materializes
            # its compact d/k/g history directly and needs no duplicate ring.
            if enable_linear_replayssm_spec and cache_params.is_kda:
                if not self.use_hcu_kda_replayssm and (
                    cache_params.is_kda or not self.replayssm_spec_fold
                ):
                    # Backstop for the KDA ring invariants; this pool is
                    # sized with the final adaptive-aware draft maximum.
                    if L & (L - 1) != 0:
                        raise ValueError(
                            f"spec-verify ring length must be a power of two, got {L}"
                        )
                    if (
                        speculative_num_draft_tokens is not None
                        and L < 2 * speculative_num_draft_tokens
                    ):
                        raise ValueError(
                            f"spec-verify ring too small: {L} < "
                            f"2 * {speculative_num_draft_tokens} (early-flush margin)"
                        )
                replayssm_rawv = torch.zeros(
                    size=(num_mamba_layers, num_slots, hv, record_len, v_dim),
                    dtype=conv_dtype,
                    device=device,
                )
                replayssm_rawk = torch.zeros(
                    size=(num_mamba_layers, num_slots, h_k, record_len, k_dim),
                    dtype=conv_dtype,
                    device=device,
                )
                replayssm_beta = torch.zeros(
                    size=(num_mamba_layers, num_slots, hv, record_len),
                    dtype=torch.float32,
                    device=device,
                )
            elif enable_linear_replayssm_spec and ring_dtype != torch.float32:
                # Low parts of compact D and normalized K. The rings follow
                # the activation dtype regardless of checkpoint dtype, so
                # materialization always needs both parts.
                replayssm_rawv = torch.zeros(
                    size=(num_mamba_layers, num_slots, hv, record_len, v_dim),
                    dtype=conv_dtype,
                    device=device,
                )
                replayssm_rawk = torch.zeros(
                    size=(num_mamba_layers, num_slots, h_k, record_len, k_dim),
                    dtype=conv_dtype,
                    device=device,
                )

        if speculative_num_draft_tokens is not None:
            # Cache intermediate SSM states per draft token during target verify
            # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, V, K]
            #
            # ReplaySSM spec-verify owns rollback via the ring + cursors (the
            # verify kernel never writes per-draft snapshots; the commit never
            # reads them), so this buffer -- the dominant spec scratch, ~46x
            # the conv state -- is dead weight there and is skipped. The conv
            # intermediate windows below STAY (conv rollback consumes them).
            # The recurrent-verify fallback cannot be reached under the flag
            # (GDN + linear chain + triton enforced in server_args; the
            # backend asserts loudly if it ever is).
            # ReplaySSM skips this dominant scratch (~9GB @ K3 dspark γ=7): the
            # KDA verify kernel takes intermediate_states_buffer=None (skips the
            # per-step write, CACHE_INTERMEDIATE_STATES=False) and the commit
            # replays the ring into the checkpoint instead. This is the memory win.
            if enable_linear_replayssm_spec:
                intermediate_ssm_state_cache = None
            else:
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=device,
                )
            # Cache intermediate conv windows (last K-1 inputs) per draft token
            # during target verify.
            #
            # On CUDA (Triton conv kernel + Triton scatter) we use a
            # *deduplicated sliding-window* layout: consecutive draft tokens'
            # (K-1)-wide windows overlap by (K-2), so instead of D separate
            # [dim, K-1] windows we store one shared [dim, D+K-2] buffer per
            # (layer, slot) and expose an overlapping `as_strided` view of
            # logical shape [num_layers, size+1, draft_tokens, dim, K-1] where
            # step `t`'s window is the slice shared[..., :, t:t+K-1]. This
            # halves the conv-intermediate footprint (D*(K-1) -> D+K-2 columns)
            # with no numerical change: both the conv kernel write (idempotent
            # overlapping stores) and `fused_conv_window_scatter_with_mask`
            # consume the view through its strides.
            #
            # Dedup the sliding-window conv-intermediate only when it is safe:
            # CUDA + a linear draft chain (topk <= 1). NPU/CPU and EAGLE tree
            # verify (topk > 1) keep the dense layout -- see
            # `conv_window_dedup_enabled` for the full rationale. The
            # `fused_conv_window_scatter_with_mask` scatter is layout-agnostic,
            # so the dense fallback reads correctly through the same code path.
            dedup_conv_window = (
                not cache_params.shape.disable_conv_window_dedup
                and conv_window_dedup_enabled(
                    _is_npu, _is_cpu, speculative_eagle_topk, cache_params.is_kda
                )
            )
            self._intermediate_conv_window_phys = []
            if dedup_conv_window:
                win_len = cache_params.shape.conv_kernel - 1
                self.conv_window_axis = self._detect_conv_window_axis(
                    conv_state_shape, win_len
                )
                intermediate_conv_window_cache = []
                for conv_shape in conv_state_shape:
                    phys, view = self._allocate_deduplicated_conv_window(
                        conv_shape=conv_shape,
                        num_mamba_layers=num_mamba_layers,
                        spec_state_size=spec_state_size,
                        speculative_num_draft_tokens=speculative_num_draft_tokens,
                        conv_dtype=conv_dtype,
                    )
                    self._intermediate_conv_window_phys.append(phys)
                    intermediate_conv_window_cache.append(view)
            else:
                # Original dense layout (NPU/CPU, or EAGLE tree verify): one
                # [dim, K-1] window per draft token.
                # Shape: [num_layers, size+1, draft_tokens, dim, K-1]
                dense_conv_shapes = [
                    (
                        (conv_shape[1], conv_shape[0])
                        if _is_npu and cache_params.is_kda
                        else conv_shape
                    )
                    for conv_shape in conv_state_shape
                ]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device=device,
                    )
                    for conv_shape in dense_conv_shapes
                ]
                self._intermediate_conv_window_phys = intermediate_conv_window_cache
            self.mamba_cache = self.SpeculativeState(
                conv=conv_state,
                temporal=temporal_state,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_rawv=replayssm_rawv,
                replayssm_rawk=replayssm_rawk,
                replayssm_beta=replayssm_beta,
            )
            intermediate_ssm_gb = (
                get_tensor_size_bytes(intermediate_ssm_state_cache) / GB
                if intermediate_ssm_state_cache is not None
                else 0.0
            )
            logger.info(
                f"Mamba Cache is allocated. "
                f"max_mamba_cache_size: {size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                f"intermediate_ssm_state_cache size: {intermediate_ssm_gb:.2f}GB "
                # Report the deduplicated PHYSICAL conv-window buffers (the view
                # over-reports its logical, un-deduplicated size).
                f"intermediate_conv_window_cache size: {get_tensor_size_bytes(self._intermediate_conv_window_phys) / GB:.2f}GB "
            )
        else:
            self.mamba_cache = self.State(
                conv=conv_state,
                temporal=temporal_state,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_rawv=replayssm_rawv,
                replayssm_rawk=replayssm_rawk,
                replayssm_beta=replayssm_beta,
            )
            logger.info(
                f"Mamba Cache is allocated. "
                f"max_mamba_cache_size: {size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
            )
        if _replayssm_on:
            logger.info(
                f"GDN ReplaySSM ring buffers allocated "
                f"(record_len={record_len}, fold={self.replayssm_spec_fold}): "
                f"d={get_tensor_size_bytes(replayssm_d) / GB if replayssm_d is not None else 0.0:.3f}GB, "
                f"k={get_tensor_size_bytes(replayssm_k) / GB if replayssm_k is not None else 0.0:.3f}GB, "
                f"g={get_tensor_size_bytes(replayssm_g) / GB:.3f}GB "
                + (
                    f"rawv={get_tensor_size_bytes(replayssm_rawv) / GB:.3f}GB, "
                    f"rawk={get_tensor_size_bytes(replayssm_rawk) / GB:.3f}GB, "
                    + (
                        f"beta={get_tensor_size_bytes(replayssm_beta) / GB:.3f}GB "
                        if replayssm_beta is not None
                        else ""
                    )
                    if replayssm_rawv is not None
                    else ""
                )
            )
        # Gate granularity of the linear-attn layers (drives the kernel's
        # IS_KDA path + the g_cache layout). Read by the backend metadata to
        # decide the per-K (KDA) vs scalar (GDN) flush/advance handling.
        self.replayssm_is_kda = bool(_replayssm_on and cache_params.is_kda)
        # Decode ReplaySSM remains keyed by persistent mamba slots.
        self.replayssm_write_pos = (
            torch.zeros((size + 1,), dtype=torch.int32, device=device)
            if enable_linear_replayssm
            else None
        )
        self.replayssm_spec_write_pos = (
            torch.zeros((spec_state_size + 1,), dtype=torch.int32, device=device)
            if enable_linear_replayssm_spec and not self.replayssm_spec_fold
            else None
        )
        self.replayssm_cache_base = (
            torch.zeros((spec_state_size + 1,), dtype=torch.int32, device=device)
            if enable_linear_replayssm_spec and not self.replayssm_spec_fold
            else None
        )
        self.replayssm_is_flush = (
            torch.zeros((spec_state_size + 1,), dtype=torch.int8, device=device)
            if enable_linear_replayssm_spec and not self.replayssm_spec_fold
            else None
        )
        mem_usage_bytes = self.mamba_cache.mem_usage_bytes()
        if isinstance(self.mamba_cache, self.SpeculativeState):
            # `intermediate_conv_window` is an as_strided view whose logical
            # shape over-reports its real footprint; charge the physical buffers
            # instead. No-op for the dense layout, where the view and the
            # physical tensors coincide.
            mem_usage_bytes -= get_tensor_size_bytes(
                self.mamba_cache.intermediate_conv_window
            )
            mem_usage_bytes += get_tensor_size_bytes(
                self._intermediate_conv_window_phys
            )
        self.mem_usage = mem_usage_bytes / GB
        self.num_mamba_layers = num_mamba_layers
    # Full (unsharded) conv sub-block dims for PD transfer across different
    # attn_tp_size (GDN: [key_dim, key_dim, value_dim]); None otherwise.
    self.conv_shard_groups = getattr(cache_params.shape, "conv_shard_groups", None)
    self.conv_slice_axis = getattr(cache_params.shape, "conv_slice_axis", 0)
    self._warmup_fused_copy_slot_kernel()


def _should_fuse_slot_ops(self) -> bool:
    return (
        not _is_hcu
        and self._conv_fuse_ok
        and not envs.SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS.get()
    )



def conv_window_dedup_enabled(
    is_npu: bool, is_cpu: bool, speculative_eagle_topk: Optional[int], is_kda: bool
) -> bool:
    """Whether the deduplicated sliding-window conv-intermediate layout is safe.

    It is safe for CUDA linear draft chains whose kernels consume the window raw.
    Tree verify, NPU/CPU, and KDA keep dense windows: tree ancestors need independent
    windows, platform kernels expect contiguous steps, and KDA transposes the window
    before conv so the overlapping ``as_strided`` layout would corrupt stores.

    HCU's Triton causal_conv1d_update writes the per-step window with those
    overlapping strides and also corrupts the conv *output* (not only the saved
    window). Target verify then disagrees with the HCU decode conv, so MTP
    bonus tokens are garbage even when every draft is rejected. Keep a dense
    window on HCU.
    """
    if _is_hcu:
        return False
    return (
        not is_npu
        and not is_cpu
        and not is_kda
        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
    )


_METHODS = {}


def around_init(original, pool, *args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original(pool, *args, **kwargs)
    return _METHODS['__init__'](pool, *args, **kwargs)


def around_should_fuse_slot_ops(original, pool):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original(pool)
    return _METHODS['_should_fuse_slot_ops'](pool)


def around_conv_window_dedup(original, *args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original(*args, **kwargs)
    return _METHODS['conv_window_dedup_enabled'](*args, **kwargs)


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return
    import types
    from sglang.srt.mem_cache import memory_pool as native
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    native._is_hcu = True
    for name in ('__init__', '_should_fuse_slot_ops', 'conv_window_dedup_enabled'):
        fn = globals()[name]
        _METHODS[name] = types.FunctionType(fn.__code__, vars(native), name, fn.__defaults__)
        _METHODS[name].__kwdefaults__ = fn.__kwdefaults__
    HookRegistry.register('sglang.srt.mem_cache.memory_pool.MambaPool.__init__', around_init, HookType.AROUND)
    HookRegistry.register('sglang.srt.mem_cache.memory_pool.MambaPool._should_fuse_slot_ops', around_should_fuse_slot_ops, HookType.AROUND)

    HookRegistry.register('sglang.srt.mem_cache.memory_pool.conv_window_dedup_enabled', around_conv_window_dedup, HookType.AROUND)
