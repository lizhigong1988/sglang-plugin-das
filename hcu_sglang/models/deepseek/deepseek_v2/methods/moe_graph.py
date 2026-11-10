# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
from __future__ import annotations

import sglang.srt.models.deepseek_v2 as N0
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import eager_on_graph
from sglang.srt.utils.custom_op import register_custom_op

def _mega_moe_eager_body(
    moe,
    hidden_states: N0.torch.Tensor,
    forward_batch,
    input_ids_global,
) -> N0.torch.Tensor:
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe

    return forward_mega_moe(
        moe, hidden_states, forward_batch, input_ids_global=input_ids_global
    )

def _mega_moe_capture_stub(
    moe,
    hidden_states: N0.torch.Tensor,
    forward_batch,
    input_ids_global,
) -> N0.torch.Tensor:
    # Capture pass only: record the bridge buffer's address and shape, skip the
    # rank-coupled MegaMoE dispatch. Warmup and replay run the real body, which
    # sees get_is_capture_mode() == False and therefore sizes its output by the
    # live token count -- the shape this stub must match.
    return N0.torch.zeros_like(hidden_states)

try:
    _bcg_forward_mega_moe = eager_on_graph(True, on_capture=_mega_moe_capture_stub)(_mega_moe_eager_body)
except TypeError:
    # sglang v0.5.22 spells this kwarg capture_stub
    _bcg_forward_mega_moe = eager_on_graph(True, capture_stub=_mega_moe_capture_stub)(_mega_moe_eager_body)

class MergedMoEOutput(N0.MoEOutput):
    """Merged routed rows with a replicated shared output still separate."""

    def get_merged(self) -> N0.torch.Tensor:
        return self.routed

def _fake_dsv4_dual_stream_outputs(
    hidden_states, layer_id, mlp_reduce_scatter, shared_replicated
):
    return N0.torch.empty_like(hidden_states), (
        N0.torch.empty_like(hidden_states)
        if shared_replicated
        else hidden_states.new_empty((0,))
    )

@N0.register_custom_op(fake_impl=_fake_dsv4_dual_stream_outputs)
def dsv4_flashinfer_moe_dual_stream_outputs(
    hidden_states: N0.torch.Tensor,
    layer_id: int,
    mlp_reduce_scatter: bool,
    shared_replicated: bool,
) -> N0.Tuple[N0.torch.Tensor, N0.torch.Tensor]:
    return _run_dsv4_dual_stream_outputs(
        hidden_states, layer_id, mlp_reduce_scatter, shared_replicated
    )

def _run_dsv4_dual_stream_outputs(
    hidden_states: N0.torch.Tensor,
    layer_id: int,
    mlp_reduce_scatter: bool,
    shared_replicated: bool,
) -> N0.Tuple[N0.torch.Tensor, N0.torch.Tensor]:
    forward_context = N0.get_tc_piecewise_forward_context()
    assert forward_context is not None and forward_context.moe_fusions is not None
    moe = forward_context.moe_fusions[layer_id]
    assert moe is not None
    with N0.get_forward().scoped(
        mlp_reduce_scatter=mlp_reduce_scatter,
        flashinfer_trtllm_bypass=True,
        lora_batch_layout=N0.LoRABatchLayout.TP_GLOBAL,
        defer_moe_finalize=False,
    ):
        pieces = moe.forward_normal_dual_stream(hidden_states, leave_output_sum=True)
        assert pieces.shared_is_replicated == shared_replicated
        merged = pieces.get_merged()
        return merged, pieces.shared if shared_replicated else merged.new_empty((0,))
