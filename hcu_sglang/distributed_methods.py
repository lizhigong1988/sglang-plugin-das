# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/parallel_state.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` to initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""

from __future__ import annotations

# NOTE: distributed_backend.register_hooks rebinds these functions with
# vars(parallel_state) as globals — keep every referenced name either local
# or present in that module's namespace; no module-level helpers here.

def _reduce_scatter_tensor(
    self,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    if _ATTN_TP_USE_AITER_CUSTOM_COMM:
        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input)
        ):
            if ca_comm._IS_CAPTURING:
                if torch.cuda.is_current_stream_capturing():
                    ca_comm.reduce_scatter(
                        input,
                        output,
                        registered=getattr(
                            ca_comm, "enable_register_for_capturing", False
                        ),
                    )
                    return output
            else:
                ca_comm.reduce_scatter(input, output, registered=False)
                return output
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and (
        not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
    ):
        self.debug_check_symmetric_mempool(
            self, {"output": output, "input": input}, "reduce_scatter_tensor"
        )
        with pynccl_comm.change_state(enable=True):
            pynccl_comm.reduce_scatter(output, input)
    else:
        reduce_scatter_single(output, input, group=self.device_group)
    return output


def _all_to_all_single(
    self,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and not pynccl_comm.disabled:
        with pynccl_comm.change_state(enable=True):
            pynccl_comm.all_to_all_single(output, input)
    else:
        torch.distributed.all_to_all_single(output, input, group=self.device_group)
    return output


def _maybe_aiter_reduce_scatter(
    self, output: torch.Tensor, input: torch.Tensor
) -> bool:
    # Aiter custom reduce-scatter (ROCm). Mirrors `_all_gather_into_tensor`'s
    # custom all-gather path: an equal-chunk (no variable sizes) reduce-scatter
    # using the registered symmetric-memory buffers, which is faster than the
    # generic RCCL kernel for the small, latency-bound decode collective.
    # Gated by SGLANG_DP_USE_REDUCE_SCATTER. Falls back (returns False)
    # for HCU, non-ROCm, or unsupported shape/size/topology so the caller uses RCCL.
    # HCU stays excluded here on purpose; the DP-attention MAX_LEN combine
    # re-enables the aiter IPC kernel narrowly in dp_attention.py's
    # `_aiter_reduce_scatter_tensor` so no other reduce_scatter caller is affected.
    if not (
        is_hip()
        and not _is_hcu
        and envs.SGLANG_DP_USE_REDUCE_SCATTER.get()
        and self._has_aiter_custom_reduce_scatter()
        and input.is_contiguous()
        and output.is_contiguous()
        and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return False
    ca_comm = self.ca_comm
    # input is the full (pre-reduce) buffer; should_custom_ar bounds its size.
    if not ca_comm.should_custom_ar(input):
        return False
    # Equal-chunk only: input rows must split evenly into world_size chunks
    # matching the per-rank output rows.
    if input.shape[0] != output.shape[0] * self.world_size:
        return False
    if getattr(ca_comm, "_IS_CAPTURING", False):
        if torch.cuda.is_current_stream_capturing():
            if (
                envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
                or get_bool_env_var(
                    "SGLANG_AITER_CAPTURE_COPY_IN", default="false"
                )
                or not getattr(ca_comm, "enable_register_for_capturing", True)
            ):
                ca_comm.reduce_scatter(input, output, registered=False)
            else:
                ca_comm.reduce_scatter(input, output, registered=True)
        elif is_in_tc_piecewise_cuda_graph():
            ca_comm.reduce_scatter(input, output, registered=False)
        elif get_bool_env_var("SGLANG_AITER_AR_REAL_GRAPH_WARMUP", default="true"):
            # Registered AITER buffers are valid only during real HIP graph
            # capture. Keep eager warmup on the same collective so consumers
            # never observe a synthetic zero result.
            ca_comm.reduce_scatter(input, output, registered=False)
        else:
            # Compatibility escape hatch for older AITER runtimes.
            output.zero_()
        return True
    ca_comm.reduce_scatter(input, output, registered=False)
    return True


def all_to_all_single(
    self,
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    """All-to-all over dim 0.

    Even splits use the registered op (CUDA-graph friendly). Variable
    per-rank splits (DSV4 compressor RLC repartition) go through
    ``torch.distributed.all_to_all_single`` on the device group.
    """
    if self.world_size == 1:
        output.copy_(input)
        return output
    if output_split_sizes is None and input_split_sizes is None:
        reg_all_to_all_single(output, input, group_name=self.unique_name)
        return output
    torch.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=self.device_group,
    )
    return output


def _all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
    # Aiter custom all-gather (ROCm). Set SGLANG_USE_AITER_AG=0 to disable.
    # Aiter's should_custom_ag still owns shape/layout validation:
    # 16B alignment, weak-contiguous, supported topology, and per-rank
    # size <= max_size/(world*2).
    # On a hit, writes directly into the caller's pre-allocated `output` via
    # all_gather_reg during CUDA-graph capture, and all_gather_unreg
    # under torch_memory_saver and other paths.
    if _ATTN_TP_USE_AITER_CUSTOM_COMM:
        ca_comm = self.ca_comm
        # Only use aiter all_gather for small tensors (decode); prefill
        # performance is worse than NCCL for large tensors.
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ag(input)
            and input.numel() <= 256 * 6144  # ~3 MB, typical decode batch
        ):
            if ca_comm._IS_CAPTURING:
                if torch.cuda.is_current_stream_capturing():
                    (
                        ca_comm.all_gather_reg(input, out=output)
                        if getattr(ca_comm, "enable_register_for_capturing", False)
                        else ca_comm.all_gather_unreg(input, out=output)
                    )
                    return
            else:
                ca_comm.all_gather_unreg(input, out=output)
                return
    ca_comm = self.ca_comm
    if (
        is_hip()
        and envs.SGLANG_USE_AITER_AG.get()
        and self._has_aiter_custom_all_gather()
        and input.is_contiguous()
        and output.is_contiguous()
        and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and ca_comm.should_custom_ag(input)
    ):
        if getattr(ca_comm, "_IS_CAPTURING", False):
            if torch.cuda.is_current_stream_capturing():
                if (
                    envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
                    or get_bool_env_var(
                        "SGLANG_AITER_CAPTURE_COPY_IN", default="false"
                    )
                    or _is_hcu
                    or not getattr(ca_comm, "enable_register_for_capturing", True)
                ):
                    ca_comm.all_gather_unreg(input, out=output, dim=0)
                else:
                    ca_comm.all_gather_reg(input, out=output, dim=0)
            elif is_in_tc_piecewise_cuda_graph():
                ca_comm.all_gather_unreg(input, out=output, dim=0)
            elif get_bool_env_var(
                "SGLANG_AITER_AR_REAL_GRAPH_WARMUP", default="true"
            ):
                ca_comm.all_gather_unreg(input, out=output, dim=0)
            else:
                # Compatibility escape hatch for older AITER runtimes.
                output.zero_()
            return
        else:
            ca_comm.all_gather_unreg(input, out=output, dim=0)
            return

    pymscclpp_comm = self.pymscclpp_comm
    if pymscclpp_comm is not None and pymscclpp_comm.should_mscclpp_allgather(
        output, input
    ):
        pymscclpp_comm.all_gather(output, input)
        return

    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and (
        not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
    ):
        self.debug_check_symmetric_mempool(
            self, {"output": output}, "all_gather_into_tensor"
        )
        with pynccl_comm.change_state(enable=True):
            pynccl_comm.all_gather(output, input)
    else:
        all_gather_single(output, input, group=self.device_group)


def all_gather(
    self,
    input_: torch.Tensor,
    dim: int = -1,
    output_tensor_list: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    world_size = self.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        if output_tensor_list is not None:
            logger.warning(
                "Performing in-place all-gather with a group size of 1. "
                "This may be unnecessary; consider bypassing it for better efficiency."
            )
            output_tensor_list[0].copy_(input_)
            return None
        else:
            return input_

    if output_tensor_list is not None:
        # TODO(ch-wan): support other backends
        return torch.distributed.all_gather(
            output_tensor_list, input_, group=self.device_group
        )

    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
    )

    # For HPUs, use HPU communicator. 1be97a0: upstream ships inert
    # communicator stubs (HPU/NPU) whose ``disabled`` property RAISES on
    # non-matching platforms — treat that as disabled. Kept inline: these
    # functions execute under parallel_state globals when hooked.
    hpu_comm = self.hpu_communicator
    try:
        hpu_disabled = hpu_comm is None or bool(hpu_comm.disabled)
    except Exception:
        hpu_disabled = True
    if not hpu_disabled:
        return hpu_comm.all_gather(input_, dim)

    # For NPUs, use NPU communicator.
    npu_comm = self.npu_communicator
    try:
        npu_disabled = npu_comm is None or bool(npu_comm.disabled)
    except Exception:
        npu_disabled = True
    if not npu_disabled:
        return npu_comm.all_gather(input_, dim)

    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # NOTE: we have to use concat-style all-gather here,
    # stack-style all-gather has compatibility issues with
    # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
    output_size = (input_size[0] * world_size,) + input_size[1:]
    # Allocate output tensor.
    with self.use_symmetric_memory(
        self, disabled=not self.is_allocation_symmetric()
    ):
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )

    # All-gather.
    if input_.is_cpu:
        if is_shm_available(input_.dtype, self.world_size, self.local_size):
            return torch.ops.sgl_kernel.shm_allgather(input_, dim)
        else:
            all_gather_single(output_tensor, input_, group=self.device_group)
    else:
        self.all_gather_into_tensor(output_tensor, input_)

    # Reshape
    output_tensor = output_tensor.reshape((world_size,) + input_size)
    output_tensor = output_tensor.movedim(0, dim)

    if _is_hcu and _use_fused_reshape_to_float:
        from lightop import op

        vocab_size = (
            input_size[:dim]
            + (world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        output_tensor = op.reshape_to_float(output_tensor, vocab_size[1])
    else:
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
    return output_tensor


def _init_attn_cp_overlap_group(
    *,
    world_size: int,
    attn_cp_size: int,
    attn_tp_size: int,
    backend: Optional[str],
    recovered_rank: bool,
    rank_offset: int,
    max_world_size: Optional[int],
) -> None:
    """Second communicator over the attention CP ranks; RCCL deadlocks when one
    communicator is driven from two streams at once."""
    global _ATTN_CP_OVERLAP
    assert _ATTN_CP_OVERLAP is None, (
        "attention context parallel overlap group is already initialized"
    )
    if attn_cp_size <= 1:
        return

    span = attn_tp_size * attn_cp_size
    group_ranks = [
        list(range(base + i, base + i + span, attn_tp_size))
        for base in range(0, world_size, span)
        for i in range(attn_tp_size)
    ]
    rank = torch.distributed.get_rank()
    mine = next(ranks for ranks in group_ranks if rank in ranks)
    assert mine == _ATTN_CP.ranks, (
        f"attn_cp_overlap partition {mine} does not match attn_cp "
        f"{_ATTN_CP.ranks}; the two communicators must span the "
        "same ranks or the overlapped collectives will not pair up"
    )

    _ATTN_CP_OVERLAP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="attn_cp_overlap",
        recovered_rank=recovered_rank,
        rank_offset=rank_offset,
        max_world_size=max_world_size,
    )
