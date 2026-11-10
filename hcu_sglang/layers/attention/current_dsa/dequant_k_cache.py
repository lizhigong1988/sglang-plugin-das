# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import triton
import triton.language as tl

try:
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
        dequantize_k_cache_paged as _generic_dequantize_k_cache_paged,
    )
except ImportError:
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
        dequantize_k_cache_paged_dsa as _generic_dequantize_k_cache_paged,
    )


_DEQUANT_PAGED_BLOCK_T = int(os.getenv("SGLANG_DEQUANT_K_CACHE_BLOCK_T", "16"))
if _DEQUANT_PAGED_BLOCK_T not in (0, 1, 2, 4, 8, 16):
    raise ValueError("SGLANG_DEQUANT_K_CACHE_BLOCK_T must be one of 0,1,2,4,8,16")


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    if (
        quant_k_cache.shape[-1] != 656
        or group_size != 128
        or _DEQUANT_PAGED_BLOCK_T == 0
    ):
        return _generic_dequantize_k_cache_paged(
            quant_k_cache, page_table_1_flattened, group_size
        )

    quant_k_cache = quant_k_cache.view(-1, 656)
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    num_tokens = page_table_1_flattened.shape[0]
    output = torch.empty(
        (num_tokens, 1, 576),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    input_nope_q = quant_k_cache[:, :512]
    input_nope_s = quant_k_cache[:, 512:528].view(torch.float32)
    input_rope = quant_k_cache[:, 528:].view(torch.bfloat16)
    block_t = _DEQUANT_PAGED_BLOCK_T
    _dequantize_k_cache_paged_tiled_kernel[(triton.cdiv(num_tokens, block_t),)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        num_tokens,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        BLOCK_T=block_t,
        GROUP_SIZE_K=128,
        NUM_GROUPS_K=4,
        DIM_NOPE_K=512,
        DIM_ROPE_K=64,
        num_warps=4,
    )
    return output


@triton.jit
def _dequantize_k_cache_paged_tiled_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_ptr,
    num_tokens,
    output_stride_0,
    input_nope_q_stride_0,
    input_nope_s_stride_0,
    input_rope_stride_0,
    BLOCK_T: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    NUM_GROUPS_K: tl.constexpr,
    DIM_NOPE_K: tl.constexpr,
    DIM_ROPE_K: tl.constexpr,
):
    """Gather and dequantize BLOCK_T complete KV tokens per program.

    The original (N, 5) launch reloads page_table and schedules five programs
    per token. This 1-D launch loads each page index once and handles four
    128-element nope groups plus the 64-element rope tail in one program.
    """
    token_ids = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_ids < num_tokens
    paged = tl.load(page_table_ptr + token_ids, mask=token_mask, other=0).to(tl.int64)

    elem = tl.arange(0, GROUP_SIZE_K)
    for group in tl.static_range(0, NUM_GROUPS_K):
        src = (
            input_nope_q_ptr
            + paged[:, None] * input_nope_q_stride_0
            + group * GROUP_SIZE_K
            + elem[None, :]
        )
        scale = tl.load(
            input_nope_s_ptr + paged * input_nope_s_stride_0 + group,
            mask=token_mask,
            other=0.0,
        )
        q = tl.load(src, mask=token_mask[:, None], other=0.0).to(tl.float32)
        dst = (
            output_ptr
            + token_ids[:, None] * output_stride_0
            + group * GROUP_SIZE_K
            + elem[None, :]
        )
        tl.store(
            dst,
            (q * scale[:, None]).to(output_ptr.dtype.element_ty),
            mask=token_mask[:, None],
        )

    rope_elem = tl.arange(0, DIM_ROPE_K)
    rope_src = (
        input_rope_ptr
        + paged[:, None] * input_rope_stride_0
        + rope_elem[None, :]
    )
    rope_dst = (
        output_ptr
        + token_ids[:, None] * output_stride_0
        + DIM_NOPE_K
        + rope_elem[None, :]
    )
    rope = tl.load(rope_src, mask=token_mask[:, None], other=0.0).to(tl.bfloat16)
    tl.store(rope_dst, rope, mask=token_mask[:, None])
