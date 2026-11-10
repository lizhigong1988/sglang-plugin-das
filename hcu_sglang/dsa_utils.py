from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from functools import lru_cache
from sglang.srt.layers.attention.dsa import utils as U
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import effective_forward_mode
from sglang.srt.utils.common import ceil_align
try:
    from sglang.kernels.ops.attention.dsa.cp_split import (
        dsa_cp_interleave_q_seqs_kernel as dsa_cp_round_robin_split_q_seqs_kernel,
    )
except ImportError:
    from sglang.kernels.ops.attention.dsa.cp_split import (
        _dsa_cp_interleave_q_seqs_kernel as dsa_cp_round_robin_split_q_seqs_kernel,
    )


def should_remap_pd_dsa_seed_to_local_slots() -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    from hcu_sglang.glm5_next_policy import is_glm5_next_hcu
    from hcu_sglang.utils.detection import is_hcu

    if is_hcu():
        config = U.process_model_config()
        if is_glm5_next_hcu(config.hf_config):
            from hcu_sglang.glm5_next_policy import (
                get_glm5_next_runtime_args,
            )
            from hcu_sglang.glm5_next_policy import (
                should_remap_pd_dsa_seed_to_local_slots as hcu_should_remap,
            )

            return hcu_should_remap(get_glm5_next_runtime_args(), config)
    return (
        (U.is_cuda() or U.is_hip())
        and U.envs.SGLANG_DSA_FUSE_TOPK.get()
        and U.get_disagg().disaggregation_mode == "decode"
        and not U.get_memory().enable_hisparse
        and not U.get_parallel().dcp_enabled
    )


def dsa_prefill_has_history(forward_batch: "ForwardBatch") -> bool:
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    return prefix_lens is None or any(int(length) > 0 for length in prefix_lens)


def is_dsa_prefill_cp_round_robin_split():
    if is_hcu():
        return (
            U.is_dsa_enable_prefill_cp()
            and U.get_parallel().dsa_prefill_cp_mode == "round-robin-split"
        )
    return U.is_dsa_prefill_cp_interleave()


def is_graph_dsa_split_op_surface(forward_batch: "ForwardBatch") -> bool:
    return (
        (U.is_cuda() or U.is_hip())
        and (U.is_in_tc_piecewise_cuda_graph() or U.is_in_breakable_cuda_graph())
        and effective_forward_mode(forward_batch).is_extend_without_speculative()
    )


def _can_dsa_cp_v2_split_uneven(forward_batch: "ForwardBatch") -> bool:
    # Import lazily: CP strategy initialization loads the DSA metadata helper.
    from sglang.srt.layers.cp.utils import is_cp_v2_active

    return is_cp_v2_active(forward_batch)


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not effective_forward_mode(forward_batch).is_context_parallel_extend():
        return False
    cp_size = U.get_parallel().attn_cp_size
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


def dsa_cp_round_robin_split_data(input_: Union[U.torch.Tensor, List]):
    cp_size = U.get_parallel().attn_cp_size
    cp_rank = U.get_parallel().attn_cp_rank
    if isinstance(input_, (tuple, list)):
        return input_[cp_rank::cp_size]

    tokens = len(input_)
    if tokens % cp_size != 0:
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = U.torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    shard = input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank]
    if shard.stride(0) != input_.stride(0):
        shard = shard.clone(memory_format=U.torch.contiguous_format)
    return shard


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.layers.cp.utils import enable_cp_v2, is_cp_active

    # CP-v2 already pads each rank-local shard to its physical size
    if is_cp_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            U.get_parallel().attn_cp_rank
        ]

    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = U.get_parallel().attn_cp_size
    if not enable_cp_v2():
        cp_align_size = get_cp_padding_align_size()
        for i in range(sync_group_size):
            global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
    # Reuse the mode selected when the DP buffer was prepared.
    dp_padding_mode = forward_batch.dp_padding_mode
    if dp_padding_mode is None:
        dp_padding_mode = U.DpPaddingMode.get_dp_padding_mode(
            forward_batch.is_extend_in_batch, global_num_tokens
        )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    else:
        tokens = global_num_tokens[U.dp_slot_in(global_num_tokens)]
    if can_dsa_prefill_cp_round_robin_split(forward_batch):
        tokens = U.ceil_div(tokens, attn_cp_size)
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    attn_cp_size = U.get_parallel().attn_cp_size
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_round_robin_split(
        forward_batch
    )
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        return dsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - dsa_cache_seqlens.shape[0]
    if pad_len > 0:
        dsa_cache_seqlens = U.torch.cat(
            [
                dsa_cache_seqlens,
                dsa_cache_seqlens.new_zeros(pad_len, *dsa_cache_seqlens.shape[1:]),
            ]
        )
    return dsa_cache_seqlens


def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):
    if (
        cp_size <= 1
        or not use_dsa
        or not effective_forward_mode(forward_batch).is_context_parallel_extend()
        or not U.is_dsa_enable_prefill_cp()
        or sum(forward_batch.extend_seq_lens_cpu) < cp_size
    ):
        return False
    if is_dsa_prefill_cp_round_robin_split():
        # Physical input tensors are padded by ForwardBatch to the CP alignment.
        # The original per-request token count may be uneven across ranks.
        if seq_len % cp_size != 0:
            return False
        cur_cp_seq_len = seq_len // cp_size
    else:
        cur_cp_seq_len = seq_len // (cp_size * 2)
    return cur_cp_seq_len != 0


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = U.get_parallel().attn_cp_size
    cp_rank = U.get_parallel().attn_cp_rank
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
) -> Tuple[List, U.torch.Tensor, List, U.torch.Tensor]:
    cp_size = U.get_parallel().attn_cp_size
    cp_rank = U.get_parallel().attn_cp_rank
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = U.torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = U.torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=U.torch.int32
    )
    dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = U.is_dsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and effective_forward_mode(forward_batch).is_context_parallel_extend()
    ):
        return True
    else:
        return False


def _dsa_prefill_has_history(forward_batch: "ForwardBatch") -> bool:
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if prefix_lens is None:
        return True
    return any(int(prefix_len) > 0 for prefix_len in prefix_lens)


def maybe_configure_main_kv_page_plan(forward_batch: "ForwardBatch") -> None:
    """Install this batch's compact Main-KV mapping when LayerSplit is active."""

    from hcu_sglang.mem_cache.dsa_cache_layer_split import build_main_kv_page_plan
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool

    token_to_kv_pool = U.get_token_to_kv_pool()
    use_prefill_cp = dsa_use_prefill_cp(forward_batch)
    configure_page_plan = getattr(token_to_kv_pool, "configure_main_kv_page_plan", None)
    if configure_page_plan is not None:
        page_plan = None
        can_compact_main_kv = (
            is_hcu()
            and getattr(token_to_kv_pool, "layer_shard_enabled", False)
            and use_prefill_cp
            and forward_batch.forward_mode.is_extend_without_speculative()
            # Compact mapping is batch-specific; retain the official full-
            # scratch path while TBO alternates between two child batches.
            and not forward_batch.can_run_tbo
            and forward_batch.tbo_parent_token_range is None
            and forward_batch.extend_prefix_lens_cpu is not None
            and forward_batch.out_cache_loc is not None
        )
        if can_compact_main_kv:
            if forward_batch.dsa_layer_split_main_kv_page_plan is None:
                forward_batch.dsa_layer_split_main_kv_page_plan = (
                    build_main_kv_page_plan(
                        req_to_token=get_req_to_token_pool().req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices,
                        prefix_lens=forward_batch.extend_prefix_lens_cpu,
                        current_locs=forward_batch.out_cache_loc,
                        page_size=token_to_kv_pool.page_size,
                    )
                )
            page_plan = forward_batch.dsa_layer_split_main_kv_page_plan
        # Explicitly clear a compact layout left by an earlier ForwardBatch
        # before entering the legacy full-pool path.
        configure_page_plan(page_plan, forward_batch)


def maybe_prefetch_full_attention_kv(
    forward_batch: "ForwardBatch",
    full_attention_layer_id: Optional[int],
) -> None:
    """Configure the batch plan and prefetch one DSA layer's caches."""

    maybe_configure_main_kv_page_plan(forward_batch)

    if full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    token_to_kv_pool = U.get_token_to_kv_pool()
    has_history = _dsa_prefill_has_history(forward_batch)
    prefetch_mla = getattr(token_to_kv_pool, "prefetch_mla_kv_buffer", None)
    if prefetch_mla is not None:
        prefetch_mla(full_attention_layer_id, has_history=has_history)
    else:
        # Index-K and Main-KV share a communicator, so all ranks enqueue them
        # in this order. A skip-topk layer makes the Index-K call a no-op.
        prefetch_index = getattr(token_to_kv_pool, "prefetch_index_buffer", None)
        if is_hcu() and prefetch_index is not None:
            prefetch_index(full_attention_layer_id, has_history=has_history)
        prefetch_kv = getattr(token_to_kv_pool, "prefetch_kv_buffer", None)
        if prefetch_kv is not None:
            prefetch_kv(full_attention_layer_id, has_history=has_history)


def maybe_prefetch_next_full_attention_kv(
    forward_batch: "ForwardBatch",
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch the next layer while the current layer's MLP runs."""

    maybe_prefetch_full_attention_kv(forward_batch, next_full_attention_layer_id)


def _dispatch(implementation):
    def wrapper(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return wrapper


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name in ('_can_dsa_cp_v2_split_uneven', '_dsa_prefill_has_history', 'cal_padded_tokens', 'can_dsa_cp_split', 'can_dsa_prefill_cp_round_robin_split', 'dsa_cp_round_robin_split_data', 'dsa_cp_round_robin_split_q_seqs', 'dsa_cp_round_robin_split_q_seqs_cpu', 'dsa_prefill_has_history', 'dsa_use_prefill_cp', 'is_dsa_prefill_cp_round_robin_split', 'is_graph_dsa_split_op_surface', 'maybe_configure_main_kv_page_plan', 'maybe_prefetch_full_attention_kv', 'maybe_prefetch_next_full_attention_kv', 'pad_dsa_cache_seqlens', 'should_remap_pd_dsa_seed_to_local_slots'):
        implementation = globals()[name]
        if hasattr(U, name):
            HookRegistry.register('sglang.srt.layers.attention.dsa.utils.' + name,
                                  _dispatch(implementation), HookType.AROUND)
        else:
            setattr(U, name, implementation)
