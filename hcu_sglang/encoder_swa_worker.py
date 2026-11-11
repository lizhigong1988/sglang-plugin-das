# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

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
from __future__ import annotations

from typing import Optional
from sglang.srt.managers import tp_worker as N
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, CaptureHiddenMode, PPProxyTensors
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import GenerationBatchResult
from hcu_sglang.utils.detection import is_hcu


def forward_batch_generation(
    self,
    batch: Optional[ScheduleBatch],
    forward_batch: Optional[ForwardBatch] = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    is_verify: bool = False,
    skip_attn_backend_init: Optional[bool] = None,  # deprecated
    *,
    capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    return_hidden_states_before_norm: bool = False,
    return_kv_loc_plan: bool = False,
) -> GenerationBatchResult:
    # Get forward batch from schedule batch
    folded = None
    if batch is not None:
        # update the consumer index of hicache to the running batch
        self.set_hicache_consumer(batch.hicache_consumer_index)

        if N.get_exec().features.enable_encoder_swa_bounded_replay:
            from hcu_sglang.encoder_swa_replay import (
                apply_folded_extend,
                fold_encoder_swa_replay,
                run_encoder_swa_replay,
            )

            if getattr(self.model_runner.attn_backend, "folds_encoder_swa_replay", False):
                # Hits extend from their replay start, rebuilding the window in-pass.
                folded = fold_encoder_swa_replay(self, batch)
            else:
                # Replay reads restored main/indexer KV before the normal extend.
                run_encoder_swa_replay(self, batch)

        forward_batch = ForwardBatch.init_new(
            batch if folded is None else folded.batch,
            self.model_runner,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=return_hidden_states_before_norm,
        )
        if folded is not None:
            apply_folded_extend(folded, forward_batch)
    else:
        # FIXME(lsyin): unify the interface of forward_batch
        assert forward_batch is not None
        assert capture_hidden_mode is None, (
            "capture_hidden_mode override requires a ScheduleBatch input"
        )

    # Deprecated kwarg: pre-planners mark the batch themselves now.
    forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

    if self.is_dllm():
        return self._forward_batch_generation_dllm(forward_batch, batch)

    if self.pp_group.is_last_rank:
        out = self.model_runner.forward(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if folded is not None:
            from hcu_sglang.encoder_swa_replay import (
                drop_folded_rows,
            )

            tail = self.model_runner.attn_backend.tail_forward_metadata
            drop_folded_rows(
                logits_output=logits_output,
                folded=folded,
                tail_lens=(
                    None
                    if tail is None
                    else tail.late_layer_tail.extend_seq_lens_cpu
                ),
            )
        batch_result_kwargs = dict(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
            routed_experts_output=out.routed_experts_output,
            indexer_topk_output=out.indexer_topk_output,
        )
        if return_kv_loc_plan and "kv_loc_plan" in getattr(
            N.GenerationBatchResult, "__dataclass_fields__", {}
        ):
            # fork-only result field; 1be97a0 dropped it (plans live on
            # forward_batch.iteration_ids now). A folded forward also wrote
            # the replay rows; the caller's plan covers the batch it
            # scheduled.
            _translator = getattr(
                self.model_runner, "kv_index_translator", None
            )
            batch_result_kwargs["kv_loc_plan"] = (
                getattr(forward_batch, "kv_loc_plan", None)
                if folded is None
                else (
                    _translator
                    if _translator is not None
                    else self.model_runner.runner_ids
                ).own_plan(batch)
            )
        batch_result = N.GenerationBatchResult(**batch_result_kwargs)

        N.capture_pre_sample_logits(batch, forward_batch, logits_output)

        if is_verify:
            # Skip sampling; spec_v2 worker fires its own publish post-verify.
            return batch_result

        # Delay sampling only for normal generation requests.
        # Keep the existing grammar behavior unchanged.
        if (
            self.enable_overlap
            and not self.enable_spec
            and (
                forward_batch.sampling_info.grammars is not None
                or (
                    N.envs.SGLANG_ENABLE_DELAY_SAMPLE.get()
                    and not forward_batch.is_prefill_only
                )
            )
        ):

            def sample_batch_func():
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
                self._maybe_finalize_elastic_cuda_graph_scale()
                return batch_result

            batch_result.delay_sample_func = sample_batch_func
            return batch_result

        if not forward_batch.is_prefill_only:
            # For normal requests, sample the next token ids.
            batch_result.next_token_ids = self.model_runner.sample(
                logits_output, forward_batch
            )
        else:
            # For prefill-only requests, create dummy token IDs on CPU
            # The size should match the batch size (number of sequences), not total tokens
            batch_result.next_token_ids = N.torch.zeros(
                len(forward_batch.seq_lens),
                dtype=N.torch.long,
                device=forward_batch.input_ids.device,
            )
            if (
                forward_batch.return_logprob
                and logits_output.next_token_logits is not None
            ):
                # NOTE: Compute logprobs without full sampling
                self.model_runner.compute_logprobs_only(
                    logits_output, forward_batch
                )

        self._maybe_finalize_elastic_cuda_graph_scale()
        return batch_result
    else:
        out = self.model_runner.forward(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
        return N.GenerationBatchResult(
            pp_hidden_states_proxy_tensors=pp_proxy_tensors,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )


def around_generation(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return forward_batch_generation(self, *args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation", around_generation, HookType.AROUND)
