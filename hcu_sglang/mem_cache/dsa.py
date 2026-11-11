"""HCU DSA index-key cache compatibility."""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import (
    is_hcu,
    is_hcu_native_fp8_supported,
)


def _uses_bf16_index_cache(pool: Any) -> bool:
    return is_hcu() and not getattr(pool, "use_fp8_index_k_cache", True)


def _should_use_bf16_index_cache(dtype: torch.dtype) -> bool:
    return is_hcu() and (
        dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or not is_hcu_native_fp8_supported()
    )


def _index_cache_bytes_per_token(dtype: torch.dtype, index_head_dim: int) -> int:
    if _should_use_bf16_index_cache(dtype):
        return index_head_dim * torch.bfloat16.itemsize
    quant_block_size = 128
    return index_head_dim + index_head_dim // quant_block_size * 4


def _create_bf16_index_k_buffer(pool: Any) -> None:
    num_pages = (pool.index_buf_size + pool.page_size + 1) // pool.page_size
    with (
        torch.cuda.use_mem_pool(pool.custom_mem_pool)
        if pool.custom_mem_pool
        else nullcontext()
    ):
        pool.index_k_buffer = [
            torch.zeros(
                (
                    0 if pool.skip_topk_layers[layer_idx] else num_pages,
                    pool.page_size,
                    1,
                    pool.index_head_dim,
                ),
                dtype=pool.index_k_buffer_dtype,
                device=pool.device,
            )
            for layer_idx in range(pool.layer_num)
        ]


def around_init(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.__init__(pool, *args, **kwargs)


def around_create_index_key_cache(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P._create_index_key_cache(pool, *args, **kwargs)


def get_index_k_with_scale_buffer_property(pool: Any):
    index_key_cache = getattr(pool, "index_key_cache", None)
    return None if index_key_cache is None else index_key_cache.buffer


def around_clear_buffers(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P._clear_buffers(pool, *args, **kwargs)


def around_move_kv_cache(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.move_kv_cache(pool, *args, **kwargs)


def get_index_k_buffer(pool: Any, layer_id: int) -> torch.Tensor:
    assert pool.index_k_buffer is not None, "BF16 index K cache is not enabled"
    if pool.layer_transfer_counter is not None:
        pool.layer_transfer_counter.wait_until(layer_id - pool.start_layer)
    return pool.index_k_buffer[layer_id - pool.start_layer]


def around_get_index_k_with_scale_buffer(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_index_k_with_scale_buffer(pool, *args, **kwargs)


def around_get_index_k_continuous(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_index_k_continuous(pool, *args, **kwargs)


def _assert_fp8_index_cache(pool: Any) -> None:
    assert not _uses_bf16_index_cache(pool), "FP8 index K cache is not enabled"


def around_fp8_only(original: Callable[..., Any], pool: Any, *args, **kwargs):
    _assert_fp8_index_cache(pool)
    return original(pool, *args, **kwargs)


def set_index_k_buffer(
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    index_k: torch.Tensor,
) -> None:
    assert pool.index_k_buffer is not None, "BF16 index K cache is not enabled"
    if index_k.dtype != pool.index_k_buffer_dtype:
        index_k = index_k.to(pool.index_k_buffer_dtype)
    pool.index_k_buffer[layer_id - pool.start_layer][
        loc // pool.page_size, loc % pool.page_size
    ] = index_k


def _copy_bf16_index_cache_to_cpu(pool: Any, indices: torch.Tensor):
    page_indices = indices[:: pool.page_size] // pool.page_size
    torch.cuda.synchronize()
    index_k_cpu = []
    page_chunk_size = max(1, pool.cpu_offloading_chunk_size // pool.page_size)
    for index_k in pool.index_k_buffer:
        layer_chunks = []
        index_k_cpu.append(layer_chunks)
        if index_k.shape[0] == 0:
            continue
        for offset in range(0, len(page_indices), page_chunk_size):
            chunk_page_indices = page_indices[offset : offset + page_chunk_size]
            layer_chunks.append(
                index_k[chunk_page_indices].to("cpu", non_blocking=True)
            )
    torch.cuda.synchronize()
    return index_k_cpu


def around_get_cpu_copy(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_cpu_copy(pool, *args, **kwargs)


def _load_bf16_index_cache_from_cpu(
    pool: Any, index_k_cpu: list[list[torch.Tensor]], indices: torch.Tensor
) -> None:
    page_indices = indices[:: pool.page_size] // pool.page_size
    torch.cuda.synchronize()
    page_chunk_size = max(1, pool.cpu_offloading_chunk_size // pool.page_size)
    for layer_id, index_k in enumerate(pool.index_k_buffer):
        if index_k.shape[0] == 0:
            continue
        for offset in range(0, len(page_indices), page_chunk_size):
            chunk_page_indices = page_indices[offset : offset + page_chunk_size]
            idx_cpu = index_k_cpu[layer_id][offset // page_chunk_size]
            assert idx_cpu.shape[0] == len(chunk_page_indices)
            index_k[chunk_page_indices] = idx_cpu.to(index_k.device, non_blocking=True)
    torch.cuda.synchronize()


def around_load_cpu_copy(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.load_cpu_copy(pool, *args, **kwargs)


def around_get_state_buf_infos(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_state_buf_infos(pool, *args, **kwargs)


def around_get_kv_size_bytes(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_kv_size_bytes(pool, *args, **kwargs)


def around_compute_dsa_indexer_cell_size(
    original: Callable[..., int],
    configurator: Any,
    *,
    kvc: Any,
    num_layers: int,
    allocate_all_layers: bool = False,
) -> int:
    workspace_bytes = 0.0
    use_bf16_index_cache = _should_use_bf16_index_cache(kvc.kv_cache_dtype)
    if use_bf16_index_cache:
        from sglang.srt.configs.model_config import get_dsa_index_head_dim
        from hcu_sglang.mem_cache.dsa_index_cache_mode import (
            IndexKCacheMode,
            index_k_workspace_bytes_per_token,
            resolve_index_k_cache_mode,
        )

        cache_mode = resolve_index_k_cache_mode(
            kvc.kv_cache_dtype,
            kvc.page_size,
            get_dsa_index_head_dim(kvc.model_config.hf_config),
        )
        workspace_bytes = index_k_workspace_bytes_per_token(cache_mode)
        use_bf16_index_cache = cache_mode is IndexKCacheMode.BF16
    if not use_bf16_index_cache:
        persistent_bytes = original(
            configurator,
            kvc=kvc,
            num_layers=num_layers,
            allocate_all_layers=allocate_all_layers,
        )
        return math.ceil(persistent_bytes + workspace_bytes)

    from sglang.srt.configs.model_config import (
        dsa_layer_skips_topk,
        get_dsa_index_head_dim,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _should_elide_dsa_index_k,
    )
    from sglang.srt.runtime_context import get_memory

    memory_config = get_memory()
    indexer_ratio = 1
    if memory_config.enable_hisparse:
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        indexer_ratio = parse_hisparse_config(kvc.server_args).host_to_device_ratio

    if allocate_all_layers or not _should_elide_dsa_index_k(
        is_draft_worker=kvc.is_draft_worker
    ):
        num_indexer_layers = num_layers
    else:
        active_indexer_layers = [
            layer_id
            for layer_id in range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
            if not dsa_layer_skips_topk(kvc.model_config.hf_config, layer_id)
        ]
        from sglang.srt.layers.cp.utils import (
            get_glm_dsa_cp_layer_shard_info,
            get_layer_shard_range,
        )

        _, shard_size = get_glm_dsa_cp_layer_shard_info(kvc)
        if shard_size > 1:
            active_set = set(active_indexer_layers)
            max_owned = 0
            for rank in range(shard_size):
                start, end = get_layer_shard_range(rank, shard_size, num_layers)
                max_owned = max(
                    max_owned,
                    sum(
                        kvc.layer_info.start_layer + offset in active_set
                        for offset in range(start, end)
                    ),
                )
            num_indexer_layers = max_owned + 1
        else:
            num_indexer_layers = len(active_indexer_layers)

    index_head_dim = get_dsa_index_head_dim(kvc.model_config.hf_config)
    return int(
        _index_cache_bytes_per_token(kvc.kv_cache_dtype, index_head_dim)
        * num_indexer_layers
        * indexer_ratio
    )


def after_hybrid_swa_init(result: Any, configurator: Any, kvc: Any):
    if not _should_use_bf16_index_cache(kvc.kv_cache_dtype):
        return result

    from sglang.srt.configs.model_config import (
        get_dsa_index_head_dim,
        is_deepseek_dsa,
    )

    if not is_deepseek_dsa(kvc.model_config.hf_config):
        return result
    index_head_dim = get_dsa_index_head_dim(kvc.model_config.hf_config)
    quantized_bytes = index_head_dim + index_head_dim // 128 * 4
    bf16_bytes = _index_cache_bytes_per_token(kvc.kv_cache_dtype, index_head_dim)
    configurator._full_per_token += bf16_bytes - quantized_bytes
    return result


def _fp8_contiguous(pool, layer_id, seq_len, page_indices):
    lengths = torch.tensor([seq_len], device=page_indices.device, dtype=torch.int32)
    return pool.index_key_cache.get_k_and_scale(
        layer_id, lengths, page_indices.reshape(1, -1), seq_len, seq_len
    )


def _compat_index_k_continuous(pool, layer_id, seq_len, page_indices):
    key, _scale = _fp8_contiguous(pool, layer_id, seq_len, page_indices)
    return key.reshape(seq_len, 1, pool.index_head_dim)


def _compat_index_scale_continuous(pool, layer_id, seq_len, page_indices):
    _key, scale = _fp8_contiguous(pool, layer_id, seq_len, page_indices)
    return scale


def _install_compat_members() -> None:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    if not hasattr(DSATokenToKVPool, "get_index_k_continuous"):
        DSATokenToKVPool.get_index_k_continuous = _compat_index_k_continuous
    if not hasattr(DSATokenToKVPool, "get_index_k_scale_continuous"):
        DSATokenToKVPool.get_index_k_scale_continuous = _compat_index_scale_continuous
    DSATokenToKVPool.index_k_with_scale_buffer = P.index_k_with_scale_buffer
    for name in ('_initialize_int8_index_k_workspace', '_create_index_k_buffer', '_get_indexer_cache_index', '_clear_index_k_buffers', 'get_index_k_with_scale_write_buffer', 'get_index_k_buffer', 'dequantize_index_k_int8_paged', 'set_index_k_int8_buffer', 'set_index_k_buffer', 'get_kv_layer_ids', 'get_state_layer_ids', 'get_index_k_cache_transfer_abi'):
        setattr(DSATokenToKVPool, name, getattr(P, name))

    # 1be97a0: hybrid linear models wrap the DSA pool in HybridLinearKVPool,
    # which upstream does not give the plugin's index-K members (mode flag,
    # set/write accessors). The indexer reads them off the outer pool, so
    # expose delegating pass-throughs onto the wrapper.
    try:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    except ImportError:
        HybridLinearKVPool = None
    if HybridLinearKVPool is not None:
        if "index_k_cache_mode" not in HybridLinearKVPool.__dict__:

            @property
            def _hcu_index_k_cache_mode(self):
                return getattr(self.full_kv_pool, "index_k_cache_mode", None)

            HybridLinearKVPool.index_k_cache_mode = _hcu_index_k_cache_mode

        def _make_delegate(name):
            def _delegate(self, *args, **kwargs):
                # Model layer ids live in a different space than the wrapped
                # full-attention pool's; mirror the native read delegation's
                # wait + transfer before forwarding.
                if "layer_id" in kwargs:
                    self._wait_for_layer(kwargs["layer_id"])
                    kwargs["layer_id"] = self._transfer_full_attention_id(
                        kwargs["layer_id"]
                    )
                return getattr(self.full_kv_pool, name)(*args, **kwargs)

            return _delegate

        for name in (
            "set_index_k_int8_buffer",
            "set_index_k_buffer",
            "get_index_k_buffer",
            "dequantize_index_k_int8_paged",
            "get_kv_layer_ids",
            "get_state_layer_ids",
            "invalidate_index_buffer_for_layer",
            "_is_layer_owned",
        ):
            if name in HybridLinearKVPool.__dict__ or not hasattr(
                DSATokenToKVPool, name
            ):
                continue
            setattr(HybridLinearKVPool, name, _make_delegate(name))

        if "get_index_k_with_scale_write_buffer" not in HybridLinearKVPool.__dict__:

            def _write_buffer_delegate(self, layer_id):
                self._wait_for_layer(layer_id)
                layer_id = self._transfer_full_attention_id(layer_id)
                return self.full_kv_pool.get_index_k_with_scale_write_buffer(
                    layer_id=layer_id
                )

            HybridLinearKVPool.get_index_k_with_scale_write_buffer = (
                _write_buffer_delegate
            )

        # fork-only contiguous index-K views (bf16/int8 gather); the indexer
        # calls them with a positional layer_id, so mirror the native
        # accessors' wait + transfer instead of the kwargs-only delegate.
        def _make_layer_first_delegate(name):
            def _delegate(self, layer_id, *args, **kwargs):
                self._wait_for_layer(layer_id)
                layer_id = self._transfer_full_attention_id(layer_id)
                return getattr(self.full_kv_pool, name)(layer_id, *args, **kwargs)

            return _delegate

        for name in ("get_index_k_continuous", "get_index_k_scale_continuous"):
            if name in HybridLinearKVPool.__dict__ or not hasattr(
                DSATokenToKVPool, name
            ):
                continue
            setattr(HybridLinearKVPool, name, _make_layer_first_delegate(name))


def register_hooks() -> None:
    if not is_hcu():
        return
    from hcu_sglang.mem_cache import dsa_cache_layer_split
    dsa_cache_layer_split.register_hooks()
    from hcu_sglang.mem_cache import dsa_layer_split_integration
    dsa_layer_split_integration.register_hooks()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _install_compat_members()
    target = "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool"
    hooks = {
        "host_pool_decls": around_host_pool_decls,
        "__init__": around_init,
        "_create_index_key_cache": around_create_index_key_cache,
        "_clear_buffers": around_clear_buffers,
        "move_kv_cache": around_move_kv_cache,
        "get_index_k_with_scale_buffer": around_get_index_k_with_scale_buffer,
        "get_index_k_continuous": around_get_index_k_continuous,
        "get_index_k_scale_continuous": around_get_index_k_scale_continuous,
        "get_index_k_scale_buffer": around_get_index_k_scale_buffer,
        "set_index_k_scale_buffer": around_set_index_k_scale_buffer,
        "get_cpu_copy": around_get_cpu_copy,
        "load_cpu_copy": around_load_cpu_copy,
        "get_state_buf_infos": around_get_state_buf_infos,
        "get_kv_size_bytes": around_get_kv_size_bytes,
    }
    for method_name, hook in hooks.items():
        HookRegistry.register(f"{target}.{method_name}", hook, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_dsa_indexer_cell_size",
        around_compute_dsa_indexer_cell_size,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator.__init__",
        after_hybrid_swa_init,
        HookType.AFTER,
    )


def around_get_index_k_scale_continuous(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_index_k_scale_continuous(pool, *args, **kwargs)


def around_get_index_k_scale_buffer(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.get_index_k_scale_buffer(pool, *args, **kwargs)


def around_set_index_k_scale_buffer(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.set_index_k_scale_buffer(pool, *args, **kwargs)


def around_host_pool_decls(original, pool, *args, **kwargs):
    if not is_hcu():
        return original(pool, *args, **kwargs)
    from hcu_sglang.mem_cache import dsa_pool_methods as P

    return P.host_pool_decls(pool, *args, **kwargs)
