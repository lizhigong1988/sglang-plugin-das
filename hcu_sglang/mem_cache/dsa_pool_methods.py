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

import typing
import sglang.srt.mem_cache.memory_pool as N
from hcu_sglang.mem_cache import dsa_index_cache_mode as K
from hcu_sglang.mem_cache.dsa_index_key_cache import IndexKeyCache
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()


def __init__(
    self,
    size: int,
    page_size: int,
    kv_lora_rank: int,
    dtype: N.torch.dtype,
    qk_rope_head_dim: int,
    layer_num: int,
    device: str,
    index_head_dim: int,
    enable_memory_saver: bool,
    kv_cache_dim: int,
    start_layer: N.Optional[int] = None,
    end_layer: N.Optional[int] = None,
    index_buf_size: N.Optional[int] = None,
    index_kpool: int = 1,
    index_kpool_compress: bool = False,
    tail_extra_slots: int = 0,
    max_running_requests: N.Optional[int] = None,
    skip_topk_layers: N.Optional[N.List[bool]] = None,
    indexer_layer_ids: N.Optional[typing.Sequence[int]] = None,
    dcp_replicated: bool = False,
    kv_scale_ue8m0: N.Optional[object] = None,  # 1be97a0; native path handles ue8m0 scales
    **_upstream_extra,  # tolerate kwargs added by newer upstream pins
):
    override_dim = (
        kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
    )

    super(N.DSATokenToKVPool, self).__init__(
        size,
        page_size,
        dtype,
        kv_lora_rank,
        qk_rope_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        start_layer,
        end_layer,
        use_dsa=True,
        override_kv_cache_dim=override_dim,
        dcp_replicated=dcp_replicated,
    )
    # self.index_k_dtype = torch.float8_e4m3fn
    # self.index_k_scale_dtype = torch.float32
    self.index_head_dim = index_head_dim
    self.index_kpool = index_kpool
    self.index_kpool_compress = index_kpool_compress
    self.tail_extra_slots = tail_extra_slots
    assert self.page_size % index_kpool == 0, (
        f"page_size {self.page_size} must be a multiple of index_kpool {index_kpool}"
    )
    self.index_page_size = (
        self.page_size if _is_hcu else self.page_size // index_kpool
    )
    layer_range = range(self.start_layer, self.start_layer + self.layer_num)
    if skip_topk_layers is not None and len(skip_topk_layers) != self.layer_num:
        raise ValueError("skip_topk_layers must have one entry per local layer")
    if indexer_layer_ids is None:
        resolved_indexer_layer_ids = tuple(
            layer_id
            for layer_id in layer_range
            if skip_topk_layers is None
            or not skip_topk_layers[layer_id - self.start_layer]
        )
    else:
        if len(indexer_layer_ids) != len(set(indexer_layer_ids)):
            raise ValueError("indexer_layer_ids must not contain duplicates")
        resolved_indexer_layer_ids = tuple(sorted(indexer_layer_ids))
        invalid_layer_ids = [
            layer_id
            for layer_id in resolved_indexer_layer_ids
            if layer_id not in layer_range
        ]
        if invalid_layer_ids:
            raise ValueError(
                f"indexer_layer_ids {invalid_layer_ids} are outside local layer "
                f"range [{self.start_layer}, "
                f"{self.start_layer + self.layer_num})"
            )
    self.indexer_layer_ids = resolved_indexer_layer_ids
    self.indexer_layer_num = len(self.indexer_layer_ids)
    self.indexer_layer_id_to_index = {
        layer_id: index for index, layer_id in enumerate(self.indexer_layer_ids)
    }
    self.skip_topk_layers = (
        list(skip_topk_layers)
        if skip_topk_layers is not None
        else [
            layer_id not in self.indexer_layer_id_to_index
            for layer_id in layer_range
        ]
    )
    if len(self.skip_topk_layers) != self.layer_num:
        raise ValueError("skip_topk_layers must have one entry per local layer")
    if dcp_replicated and not _is_hcu:
        self.index_page_size //= N.get_parallel().attn_dcp_size
    if index_buf_size is None:
        dcp_size = N.get_parallel().attn_dcp_size
        index_buf_size = (
            size
            if dcp_replicated
            else (size + self.page_size) * dcp_size - self.page_size
        )
    self.index_buf_size = index_buf_size
    # num head == 1 and head dim == 128 for index_k in DSA
    assert index_head_dim == 128
    self.index_k_cache_mode = K.resolve_index_k_cache_mode(
        dtype, page_size, index_head_dim
    )
    self.use_fp8_index_k_cache = (
        self.index_k_cache_mode is K.IndexKCacheMode.FP8_SCALED
    )
    self.use_int8_index_k_cache = (
        self.index_k_cache_mode is K.IndexKCacheMode.INT8_SCALED
    )
    self.use_scaled_index_k_cache = (
        self.index_k_cache_mode is not K.IndexKCacheMode.BF16
    )
    self.index_k_buffer_dtype = (
        N.torch.bfloat16
        if self.index_k_cache_mode is K.IndexKCacheMode.BF16
        else self.dtype
    )

    physical_page_size = self.index_page_size
    if N._is_hip and not _is_hcu:
        if N.aiter_can_use_preshuffle_paged_mqa():
            assert physical_page_size % 16 == 0, (
                f"HIP preshuffle requires page_size to be a multiple of 16, got {physical_page_size}"
            )
        else:
            assert physical_page_size == 1, (
                f"HIP legacy DSA path requires page_size == 1, got {physical_page_size}"
            )
    elif N.is_xpu():
        assert physical_page_size in (
            64,
            128,
        ), f"XPU DSA requires page_size 64 or 128, got {physical_page_size}"
    else:
        assert physical_page_size == 64, (
            f"DSA requires 64-token physical pages, got page_size={self.page_size} "
            f"with index_kpool={index_kpool}"
        )
    self.index_key_cache = self._create_index_key_cache()
    self._init_kpool_compress_tail_buffers(
        index_kpool=index_kpool,
        index_kpool_compress=index_kpool_compress,
        tail_extra_slots=tail_extra_slots,
        index_head_dim=index_head_dim,
        layer_num=layer_num,
        device=device,
        max_running_requests=max_running_requests,
    )
    self._initialize_int8_index_k_workspace()
    self._finalize_allocation_log(size)

    if self.use_int8_index_k_cache:
        persistent_bytes = sum(
            buffer.nbytes for buffer in self.index_key_cache.buffer
        ) + getattr(
            getattr(self.index_key_cache, "remote_buffer", None), "nbytes", 0
        )
        N.logger.info(
            "DSA index-K cache mode=%s, persistent bytes/token/layer=%d, "
            "workspace_count=1, workspace_bytes=%d, page_claim_bytes=%d, "
            "persistent_bytes=%d",
            self.index_k_cache_mode.value,
            K.index_k_cache_bytes_per_token(self.index_k_cache_mode),
            self.index_k_dequant_workspace.nbytes,
            self.index_k_page_claims.nbytes,
            persistent_bytes,
        )

def _create_index_key_cache(self) -> N.Optional[IndexKeyCache]:
    if self.index_k_cache_mode is K.IndexKCacheMode.BF16:
        # The official facade only implements the quantized (FP8 data +
        # FP32 scale) layout. HCU parts without a scaled cache fall back to
        # a plain bf16 index-K cache, which the pool owns directly.
        self._create_index_k_buffer()
        return None
    self.index_k_buffer = None
    return IndexKeyCache(self, self.index_buf_size)

def _initialize_int8_index_k_workspace(self) -> None:
    self.index_k_int8_aliases = None
    self.index_k_int8_remote_aliases = None
    self.index_k_dequant_workspace = None
    self.index_k_page_claims = None
    if not self.use_int8_index_k_cache:
        return

    self.index_k_int8_aliases = [
        K.create_index_k_int8_aliases(buffer)
        for buffer in self.index_key_cache.buffer
    ]
    remote_buffer = getattr(self.index_key_cache, "remote_buffer", None)
    if remote_buffer is not None:
        self.index_k_int8_remote_aliases = K.create_index_k_int8_aliases(
            remote_buffer
        )

    num_pages = (self.index_buf_size + self.page_size + 1) // self.page_size
    with (
        self.memory_saver_adapter.region(N.GPU_MEMORY_TYPE_KV_CACHE),
        (
            N.torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else N.nullcontext()
        ),
    ):
        self.index_k_dequant_workspace = N.torch.empty(
            (
                num_pages,
                self.page_size,
                1,
                self.index_head_dim,
            ),
            dtype=N.torch.bfloat16,
            device=self.device,
        )
        self.index_k_page_claims = N.torch.empty(
            (num_pages,), dtype=N.torch.int32, device=self.device
        )

def _create_index_k_buffer(self):
    num_pages = (self.index_buf_size + self.page_size + 1) // self.page_size
    with (
        N.torch.cuda.use_mem_pool(self.custom_mem_pool)
        if self.custom_mem_pool
        else N.nullcontext()
    ):
        self.index_k_buffer = [
            N.torch.zeros(
                (
                    num_pages
                    if self._should_allocate_index_layer(layer_id - self.start_layer)
                    else 0,
                    self.page_size,
                    1,
                    self.index_head_dim,
                ),
                dtype=self.index_k_buffer_dtype,
                device=self.device,
            )
            for layer_id in self.indexer_layer_ids
        ]

def _get_indexer_cache_index(self, layer_id: int) -> int:
    try:
        return self.indexer_layer_id_to_index[layer_id]
    except KeyError as exc:
        raise ValueError(
            f"Layer {layer_id} does not own an Index-K cache; active layers are "
            f"{self.indexer_layer_ids}"
        ) from exc

def host_pool_decls(self):
    # pool_host imports this module. Resolve the mirror side lazily.
    from sglang.srt.mem_cache.pool_host.dsa import make_dsa_indexer_pool_decl

    kv_decls = super(N.DSATokenToKVPool, self).host_pool_decls()
    # Shared-topk layers own a 0-row placeholder, so a non-empty buffer list
    # is not enough: some layer must actually hold index keys.
    if (
        not self.index_k_with_scale_buffer
        and not (_is_hcu and getattr(self, "index_k_buffer", None))
    ) or all(self.skip_topk_layers):
        return kv_decls
    return (*kv_decls, make_dsa_indexer_pool_decl(self))

@property
def index_k_with_scale_buffer(self):
    # Preserve direct HiCache access while storage lives behind the facade.
    # ``None`` on the HCU bf16 fallback, where no quantized buffer exists.
    if self.index_key_cache is None:
        return None
    return self.index_key_cache.buffer

def _init_kpool_compress_tail_buffers(
    self,
    index_kpool: int,
    index_kpool_compress: bool,
    tail_extra_slots: int,
    index_head_dim: int,
    layer_num: int,
    device: str,
    max_running_requests: N.Optional[int],
) -> None:
    """Keep request tails on the pool so they follow the index-cache lifecycle."""
    self.kpool_use_compress = index_kpool > 1 and index_kpool_compress

    if not self.kpool_use_compress:
        self._compress_tail_k = None
        self._compress_tail_score = None
        return

    assert max_running_requests is not None, (
        "DSATokenToKVPool with kpool compress requires max_running_requests"
    )
    # +1 mirrors req_to_token_pool.size + 1 used by the indexer to
    # provide an extra slot for invalid / sentinel req indices.
    req_pool_size = max_running_requests + 1
    tail_dtype = N.torch.bfloat16
    tail_width = index_kpool + tail_extra_slots
    with (
        N.torch.cuda.use_mem_pool(self.custom_mem_pool)
        if self.custom_mem_pool
        else N.nullcontext()
    ):
        self._compress_tail_k: N.Optional[N.List[N.torch.Tensor]] = [
            N.torch.zeros(
                req_pool_size if self._should_allocate_index_layer(i) else 0,
                tail_width,
                index_head_dim,
                dtype=tail_dtype,
                device=device,
            )
            for i in range(layer_num)
        ]
        self._compress_tail_score: N.Optional[N.List[N.torch.Tensor]] = [
            N.torch.zeros(
                req_pool_size if self._should_allocate_index_layer(i) else 0,
                tail_width,
                index_head_dim,
                dtype=tail_dtype,
                device=device,
            )
            for i in range(layer_num)
        ]

def get_compress_tail_buffers(
    self, layer_id: int
) -> N.Tuple[N.torch.Tensor, N.torch.Tensor]:
    assert self.kpool_use_compress, (
        "get_compress_tail_buffers called when kpool compress is disabled"
    )
    idx = layer_id - self.start_layer
    return (
        self._compress_tail_k[idx],
        self._compress_tail_score[idx],
    )

def get_compress_tail_buf_infos(self):
    if not self.kpool_use_compress:
        return [], [], []
    transfer_layer_ids = list(range(self.layer_num))
    # Keep zero-row indexShare entries in the pointer list so layer offsets
    # stay aligned across PD peers; item_len=0 makes transfer backends skip them.
    tail_buffers = [self._compress_tail_k[i] for i in transfer_layer_ids] + [
        self._compress_tail_score[i] for i in transfer_layer_ids
    ]
    data_ptrs = [buf.data_ptr() for buf in tail_buffers]
    data_lens = [buf.nbytes for buf in tail_buffers]
    item_lens = [buf[0].nbytes if buf.shape[0] > 0 else 0 for buf in tail_buffers]
    return data_ptrs, data_lens, item_lens

def kpool_decode_update_index_cache(
    self,
    layer_id: int,
    key: N.torch.Tensor,
    slot_score: N.torch.Tensor,
    ape: N.torch.Tensor,
    block_tables: N.torch.Tensor,
    req_pool_indices: N.torch.Tensor,
    positions: N.torch.Tensor,
    seq_lens: N.torch.Tensor,
    out_cache_loc: N.torch.Tensor,
    round_scale: bool = False,
) -> None:
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        kpool_decode_update_and_maybe_write_cache,
    )

    assert self.kpool_use_compress, (
        "kpool_decode_update_index_cache called when kpool compress is disabled"
    )
    idx = layer_id - self.start_layer
    buf = self.get_index_k_with_scale_buffer(layer_id)
    kpool_decode_update_and_maybe_write_cache(
        pool=self,
        buf=buf,
        tail_k=self._compress_tail_k[idx],
        tail_score=self._compress_tail_score[idx],
        key=key,
        slot_score=slot_score,
        ape=ape,
        block_tables=block_tables,
        req_pool_indices=req_pool_indices,
        positions=positions,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        round_scale=round_scale,
    )

def set_compress_tail_for_request(
    self,
    layer_id: int,
    req_pool_idx: N.torch.Tensor,
    key_tail: N.torch.Tensor,
    score_tail: N.torch.Tensor,
    n_remain: int,
    dst_logical_start: int,
) -> None:
    """Leave the ring untouched at a pool boundary; no tail carries over."""
    assert self.kpool_use_compress, (
        "set_compress_tail_for_request called when kpool compress is disabled"
    )
    idx = layer_id - self.start_layer
    if n_remain > 0:
        slots = (
            N.torch.arange(n_remain, device=key_tail.device, dtype=N.torch.long)
            + int(dst_logical_start)
        ) % self._compress_tail_k[idx].shape[1]
        self._compress_tail_k[idx][req_pool_idx, slots] = key_tail
        self._compress_tail_score[idx][req_pool_idx, slots] = score_tail

def _clear_index_k_buffers(self) -> None:
    self.index_k_int8_aliases = None
    self.index_k_int8_remote_aliases = None
    self.index_k_dequant_workspace = None
    self.index_k_page_claims = None
    if self.index_key_cache is not None:
        self.index_key_cache.clear()
    if getattr(self, "index_k_buffer", None) is not None:
        del self.index_k_buffer
    if hasattr(self, "_compress_tail_k") and self._compress_tail_k is not None:
        del self._compress_tail_k
        del self._compress_tail_score

def _clear_buffers(self):
    super(N.DSATokenToKVPool, self)._clear_buffers()
    self._clear_index_k_buffers()

def move_kv_cache(self, tgt_loc: N.torch.Tensor, src_loc: N.torch.Tensor):
    """Move latent KV and the DSA indexer cache (key + scale) in lockstep."""
    super(N.DSATokenToKVPool, self).move_kv_cache(tgt_loc, src_loc)
    if self.index_key_cache is not None:
        self.index_key_cache.move(tgt_loc, src_loc)
        return

    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for index_k in self.index_k_buffer:
        flat_index_k = index_k.view(-1, 1, self.index_head_dim)
        flat_index_k[tgt_loc_flat] = flat_index_k[src_loc_flat]

def get_index_k_with_scale_buffer(self, layer_id: int) -> N.torch.Tensor:
    assert self.use_scaled_index_k_cache, "Scaled index K cache is not enabled"
    return self.index_key_cache.get_local_buffer(layer_id)

def get_index_k_with_scale_write_buffer(self, layer_id: int) -> N.torch.Tensor:
    assert self.use_scaled_index_k_cache, "Scaled index K cache is not enabled"
    return self.index_key_cache.get_write_buffer(layer_id)

def get_index_k_buffer(self, layer_id: int) -> N.torch.Tensor:
    assert self.index_k_buffer is not None, "BF16 index K cache is not enabled"
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    return self.index_k_buffer[self._get_indexer_cache_index(layer_id)]

def get_index_k_continuous(
    self,
    layer_id: int,
    seq_len: int,
    page_indices: N.torch.Tensor,
):
    if self.use_int8_index_k_cache:
        num_pages = (seq_len + self.page_size - 1) // self.page_size
        return self.index_k_dequant_workspace[page_indices[:num_pages]].view(
            -1, 1, self.index_head_dim
        )[:seq_len]
    if self.index_k_cache_mode is K.IndexKCacheMode.BF16:
        num_pages = (seq_len + self.page_size - 1) // self.page_size
        buf = self.get_index_k_buffer(layer_id)
        return buf[page_indices[:num_pages]].view(-1, 1, self.index_head_dim)[
            :seq_len
        ]
    return self.index_key_cache.get_k_continuous(layer_id, seq_len, page_indices)

def dequantize_index_k_int8_paged(
    self,
    layer_id: int,
    block_tables: N.torch.Tensor,
    context_lens: N.torch.Tensor,
) -> N.torch.Tensor:
    assert self.use_int8_index_k_cache, "INT8 index K cache is not enabled"
    cache_index = self._get_indexer_cache_index(layer_id)
    packed_cache = self.index_key_cache.get_buffer(layer_id)
    local_cache = self.index_key_cache.buffer[cache_index]
    if packed_cache is local_cache:
        int8_k, fp32_scales = self.index_k_int8_aliases[cache_index]
    elif self.index_k_int8_remote_aliases is not None and packed_cache is getattr(
        self.index_key_cache, "remote_buffer", None
    ):
        int8_k, fp32_scales = self.index_k_int8_remote_aliases
    else:
        int8_k, fp32_scales = K.create_index_k_int8_aliases(packed_cache)

    return K.dequantize_index_k_int8_paged(
        packed_cache,
        block_tables,
        context_lens,
        self.index_k_dequant_workspace,
        self.index_k_page_claims,
        int8_k=int8_k,
        fp32_scales=fp32_scales,
    )

def get_index_k_scale_continuous(
    self,
    layer_id: int,
    seq_len: int,
    page_indices: N.torch.Tensor,
):
    assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
    return self.index_key_cache.get_k_scale_continuous(
        layer_id, seq_len, page_indices
    )

def get_index_k_scale_buffer(
    self,
    layer_id: int,
    seq_len_tensor: N.torch.Tensor,
    page_indices: N.torch.Tensor,
    seq_len_sum: int,
    max_seq_len: int,
):
    """
    Fused method to get both index K and scale data in a single call using Triton.
    More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

    :param layer_id: Layer index
    :param seq_len: Sequence length
    :param page_indices: Page indices tensor
    :return: tuple of (k_fp8, k_scale) where
             k_fp8: (seq_len, index_head_dim), uint8
             k_scale: (seq_len, 4), uint8
    """
    assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
    return self.index_key_cache.get_k_and_scale(
        layer_id, seq_len_tensor, page_indices, seq_len_sum, max_seq_len
    )

def set_index_k_scale_buffer(
    self,
    layer_id: int,
    loc: N.torch.Tensor,
    index_k: N.torch.Tensor,
    index_k_scale: N.torch.Tensor,
) -> None:
    assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
    self.index_key_cache.store_quantized(layer_id, loc, index_k, index_k_scale)

def set_index_k_int8_buffer(
    self,
    layer_id: int,
    loc: N.torch.Tensor,
    index_k: N.torch.Tensor,
) -> None:
    assert self.use_int8_index_k_cache, "INT8 index K cache is not enabled"
    cache_index = self._get_indexer_cache_index(layer_id)
    int8_k, fp32_scales = self.index_k_int8_aliases[cache_index]
    K.quantize_and_store_index_k_int8(
        index_k,
        self.index_key_cache.buffer[cache_index],
        loc,
        page_size=self.page_size,
        int8_k=int8_k,
        fp32_scales=fp32_scales,
    )

def _get_compress_tail_cpu_copy(self, req_pool_index):
    if not self.kpool_use_compress or req_pool_index is None:
        return None

    tail_k_cpu = []
    tail_score_cpu = []
    for tail_k, tail_score in zip(self._compress_tail_k, self._compress_tail_score):
        if tail_k.shape[0] == 0:
            tail_k_cpu.append(None)
            tail_score_cpu.append(None)
            continue
        tail_k_cpu.append(tail_k[req_pool_index].to("cpu", non_blocking=True))
        tail_score_cpu.append(
            tail_score[req_pool_index].to("cpu", non_blocking=True)
        )
    return tail_k_cpu, tail_score_cpu

def _load_compress_tail_cpu_copy(self, tail_k_cpu, tail_score_cpu, req_pool_index):
    if (
        not self.kpool_use_compress
        or req_pool_index is None
        or tail_k_cpu is None
        or tail_score_cpu is None
    ):
        return

    for tail_k, tail_score, saved_k, saved_score in zip(
        self._compress_tail_k,
        self._compress_tail_score,
        tail_k_cpu,
        tail_score_cpu,
    ):
        if tail_k.shape[0] == 0 or saved_k is None or saved_score is None:
            continue
        tail_k[req_pool_index] = saved_k.to(tail_k.device, non_blocking=True)
        tail_score[req_pool_index] = saved_score.to(
            tail_score.device, non_blocking=True
        )

def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
    # Retraction reuses index-cache pages; offload index/scale with KV so resume cannot read another request's entries.
    kv_cache_cpu = super(N.DSATokenToKVPool, self).get_cpu_copy(indices, mamba_indices=mamba_indices)
    if self.index_key_cache is None:
        # HCU bf16 index-K fallback: no quantized cache object exists, so
        # offload the plain page-indexed buffer the pool owns directly.
        index_k_cache = self.index_k_buffer
        page_indices = indices[:: self.page_size] // self.page_size
        N.torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.indexer_layer_num):
            index_k_cpu.append([])
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cache[layer_id][chunk_page_indices].to(
                    "cpu", non_blocking=True
                )
                index_k_cpu[-1].append(idx_cpu)
        N.torch.cuda.synchronize()
        return {"kv": kv_cache_cpu, "index_k": index_k_cpu}

    cpu_copy = {
        "kv": kv_cache_cpu,
        "index_k": self.index_key_cache.cpu_copy(indices),
    }
    compress_tail = self._get_compress_tail_cpu_copy(req_pool_index)
    if compress_tail is not None:
        cpu_copy["tail_k"], cpu_copy["tail_score"] = compress_tail
    N.torch.cuda.synchronize()
    return cpu_copy

def load_cpu_copy(
    self,
    kv_cache_cpu_dict,
    indices,
    mamba_indices=None,
    req_pool_index=None,
):
    super(N.DSATokenToKVPool, self).load_cpu_copy(
        kv_cache_cpu_dict["kv"],
        indices,
        mamba_indices=mamba_indices,
        req_pool_index=req_pool_index,
    )
    if self.index_key_cache is None:
        # HCU bf16 index-K fallback, mirrors get_cpu_copy above.
        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        index_k_cache = self.index_k_buffer
        N.torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.indexer_layer_num):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(index_k_cache[0].device, non_blocking=True)
                index_k_cache[layer_id][chunk_page_indices] = idx_chunk
        N.torch.cuda.synchronize()
        return

    self.index_key_cache.load_cpu_copy(kv_cache_cpu_dict["index_k"], indices)
    self._load_compress_tail_cpu_copy(
        kv_cache_cpu_dict.get("tail_k"),
        kv_cache_cpu_dict.get("tail_score"),
        req_pool_index,
    )
    N.torch.cuda.synchronize()

def set_index_k_buffer(
    self,
    layer_id: int,
    loc: N.torch.Tensor,
    index_k: N.torch.Tensor,
) -> None:
    assert self.index_k_buffer is not None, "BF16 index K cache is not enabled"
    if index_k.dtype != self.index_k_buffer_dtype:
        index_k = index_k.to(self.index_k_buffer_dtype)

    self.index_k_buffer[self._get_indexer_cache_index(layer_id)][
        loc // self.page_size, loc % self.page_size
    ] = index_k

def get_state_buf_infos(self):
    if self.index_key_cache is not None:
        return self.index_key_cache.state_buf_infos()

    index_cache = self.index_k_buffer
    data_ptrs = [buffer.data_ptr() for buffer in index_cache]
    data_lens = [buffer.nbytes for buffer in index_cache]
    item_lens = [buffer[0].nbytes for buffer in index_cache]
    return data_ptrs, data_lens, item_lens

def get_kv_layer_ids(self):
    """Global layer ids aligned with the MLA KV transfer buffers."""
    return list(range(self.start_layer, self.start_layer + self.layer_num))

def get_state_layer_ids(self):
    """Global layer ids aligned with the DSA Index-K transfer buffers."""
    return list(self.indexer_layer_ids)

def get_index_k_cache_transfer_abi(self) -> str:
    """Return the persistent index-K page ABI used by PD state transfer."""
    return (
        f"dsa-index-k-page-v1:{self.index_k_cache_mode.value}:"
        f"page_size={self.page_size}:head_dim={self.index_head_dim}"
    )

def get_kv_size_bytes(self):
    kv_size_bytes = super(N.DSATokenToKVPool, self).get_kv_size_bytes()
    index_cache = (
        self.index_k_with_scale_buffer
        if self.use_scaled_index_k_cache
        else self.index_k_buffer
    )
    for index_k_cache in index_cache:
        kv_size_bytes += N.get_tensor_size_bytes(index_k_cache)
    if self.use_int8_index_k_cache:
        kv_size_bytes += N.get_tensor_size_bytes(self.index_k_dequant_workspace)
        kv_size_bytes += N.get_tensor_size_bytes(self.index_k_page_claims)
    return kv_size_bytes
