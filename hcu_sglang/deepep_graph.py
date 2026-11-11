# Copyright 2023-2026 SGLang Team
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
"""DeepEP capture/replay adapter — records the dispatch mode used during
capture and re-applies it during replay so DeepEP all-to-all has
consistent expert routing across the captured graph.
"""

from __future__ import annotations

import inspect

from typing import Callable, Optional

from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.model_executor.runner_utils import deepep_adapter as A
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as N
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend


class DeepEPCudaGraphRunnerAdapter(A.DeepEPCudaGraphRunnerAdapter):
    def __init__(self) -> None:
        # Record DeepEP mode used during capture to ensure replay consistency.
        self._captured_deepep_mode = None
        self._captured_low_latency_layout = None

    def capture(self, is_extend_in_batch: bool) -> None:
        if not get_moe_a2a_backend().is_deepep():
            return
        self._captured_deepep_mode = get_deepep_mode().resolve(
            is_extend_in_batch=is_extend_in_batch
        )
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)

    def capture_end(self) -> None:
        if self._captured_deepep_mode is not None:
            self._captured_low_latency_layout = DeepEPBuffer.get_low_latency_layout()

    def replay(self) -> None:
        if not get_moe_a2a_backend().is_deepep():
            return
        assert self._captured_deepep_mode is not None
        # Graph replay bypasses the Python dispatcher. Restore the layout
        # recorded after warmup/capture, not the preceding model's layout.
        if self._captured_low_latency_layout is not None:
            DeepEPBuffer.set_low_latency_layout(*self._captured_low_latency_layout)
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)


def capture_one_shape(
    self,
    size: int,
    forward: Callable,
    a=None,
    b=None,
    c=None,
):
    # The fork called this with a leading multi-stream index —
    # (bs, forward, stream_idx, variant_label, attention_variant) — while
    # 1be97a0 dropped the streams: (bs, forward, variant_label,
    # attention_variant). stream_idx is always an int and the labels are
    # str/None, so disambiguate positionally.
    if isinstance(a, int):
        stream_idx, variant_label, attention_variant = a, b, c
    else:
        stream_idx, variant_label, attention_variant = None, a, b
    num_tokens = size * self.captured_req_width
    bs = self._ragged_capture_slots(num_tokens) if self.ragged_verify_mode else size

    # Sanity-check: --debug-cuda-graph requires breakable backend.
    if N.get_exec().graph.debug_cuda_graph:
        assert isinstance(self.backend, N.BreakableCudaGraphBackend), (
            "Breakable CUDA graph is required for --debug-cuda-graph"
        )

    # 1be97a0: upstream dropped the fork's multi-stream capture; pass
    # stream_idx only when the bound native method still accepts it.
    _prepare_kwargs = dict(num_tokens=num_tokens)
    if stream_idx is not None and "stream_idx" in inspect.signature(
        self.capture_prepare
    ).parameters:
        _prepare_kwargs["stream_idx"] = stream_idx
    forward_batch, attn_backend, pp_proxy_tensors = self.capture_prepare(
        bs, **_prepare_kwargs
    )

    # All setup hooks below read get_attn_backend() (TboForwardBatchPreparer,
    # DeepEP adapter, …) so they must run inside the same ForwardContext
    # that wraps the warmup/capture forward.
    with N.forward_context(N.ForwardContext(attn_backend=attn_backend)):
        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if forward_batch.lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

        def run_once():
            # Backends may gather live pool mappings into their static
            # metadata here; these nodes run before any model KV reads.
            attn_backend.init_forward_metadata_in_graph(forward_batch)
            self._record_in_graph_metadata_prep_done()

            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                None
            )
            N.set_dp_buffer_len(
                forward_batch.global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                forward_batch.global_num_tokens_cpu,
            )
            N.set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in N.inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = N.PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if (
                (
                    self.dllm_uses_input_embeds
                    or (
                        self.model_runner.spec_algorithm.is_dflash_family()
                        and self.model_runner.is_draft_worker
                    )
                )
                and "input_embeds" in N.inspect.signature(forward).parameters
                and not hasattr(self.model_runner.model, "forward_embed")
            ):
                if self.dllm_input_preparation is None:
                    kwargs["input_embeds"] = self.buffers.input_embeds[:num_tokens]
                else:
                    state = self.buffer_registry.get_slot(
                        "dllm_input_preparation_state"
                    ).buffer[:num_tokens]
                    kwargs["input_embeds"] = self.dllm_input_preparation(
                        self.model_runner.model, forward_batch.input_ids, state
                    )

            out = forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            for capture_hook in self.model_runner.capture_tail_hooks:
                capture_hook(self, out, forward_batch, num_tokens)
            return out

        self.deepep_adapter.capture(is_extend_in_batch=False)
        canary_ctx = (
            c.with_active_single_forward_manager(0)
            if (c := self.model_runner.canary_manager) is not None
            else N.contextlib.nullcontext()
        )

        with canary_ctx:
            # 1be97a0: upstream _make_graph_key is (size, variant_label,
            # attention_variant); keep stream_idx in the key only where the
            # fork-style signature still accepts it.
            _key_extra = (
                [stream_idx]
                if stream_idx is not None
                and "stream_idx"
                in inspect.signature(self._make_graph_key).parameters
                else []
            )
            shape_key = self._make_graph_key(
                self._capture_graph_size(bs=bs, num_tokens=num_tokens),
                *_key_extra,
                variant_label,
                attention_variant,
            )
            # Adaptive runners may own a different backend than model_runner.
            post_warmup_hook = getattr(
                attn_backend,
                "on_after_cuda_graph_warmup",
                None,
            )
            N.maybe_flashinfer_autotune_speculative_draft(
                self,
                run_once,
                post_warmup_hook=post_warmup_hook,
                run_lm_head=True,
            )
            self.backend.capture_one(
                shape_key,
                run_once,
                capture_inputs=None,
                post_warmup_hook=post_warmup_hook,
            )
            self.deepep_adapter.capture_end()



def around_capture(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return capture_one_shape(self, *args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.model_executor.runner_utils.deepep_adapter.DeepEPCudaGraphRunnerAdapter", DeepEPCudaGraphRunnerAdapter, HookType.REPLACE)
    HookRegistry.register("sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape", around_capture, HookType.AROUND)
