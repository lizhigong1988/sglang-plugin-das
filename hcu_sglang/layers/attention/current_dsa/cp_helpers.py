# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from typing import List, Tuple
import torch
from sglang.srt.runtime_context import get_parallel
try:
    from sglang.kernels.ops.attention.dsa.cp_split import dsa_cp_interleave_q_seqs_kernel as dsa_cp_round_robin_split_q_seqs_kernel
except ImportError:
    from sglang.kernels.ops.attention.dsa.cp_split import _dsa_cp_interleave_q_seqs_kernel as dsa_cp_round_robin_split_q_seqs_kernel
from hcu_sglang.models.deepseek.current_v4.hcu_backend.compat import is_dsa_prefill_cp_round_robin_split, dsa_cp_round_robin_split_data
from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import effective_forward_mode


def _can_dsa_cp_v2_split_uneven(forward_batch: "ForwardBatch") -> bool:
    # Import lazily: CP strategy initialization loads the DSA metadata helper.
    from sglang.srt.layers.cp.utils import is_cp_v2_active

    return is_cp_v2_active(forward_batch)

def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not effective_forward_mode(forward_batch).is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        # CP-v2 interleave shards uneven batches and the DSA indexer metadata
        # must follow that same layout. The divisibility guard only protects
        # the legacy model-level split, which requires equal rank lengths.
        and (seq_len % cp_size == 0 or _can_dsa_cp_v2_split_uneven(forward_batch))
        and cp_size > 1
    )

def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    extra_seq = 0
    q_seqs = []
    for cur_len in extend_seqs:
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = [i for i, q_len in enumerate(q_seqs) if q_len > 0]
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx

def dsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx
