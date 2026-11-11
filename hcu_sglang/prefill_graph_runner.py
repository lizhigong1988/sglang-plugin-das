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
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from typing import Any, Dict, List, Optional, Union
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as N
from sglang.srt.environ import envs
from hcu_sglang.prefill_graph_policy import prefill_graph_scope
from hcu_sglang.utils.detection import is_hcu

_NATIVE = [N.PrefillCudaGraphRunner]


class PrefillCudaGraphRunner(_NATIVE[0]):
    _backend_vote_prefill_cuda_graph = None

    @property
    def _attn_backend_is_dsa(self) -> bool:
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        # DSA sets use_mha=False inside piecewise/breakable graphs, so the MHA
        # companion never runs there and any prefix goes through sparse MLA.
        return isinstance(self.model_runner.attn_backend, DeepseekSparseAttnBackend)

    def __init__(self, model_runner: ModelRunner):
        if N.get_schedule().enable_mixed_chunk:
            backend = N.get_exec().graph.cuda_graph_config.prefill.backend
            assert backend in (N.Backend.BREAKABLE, N.Backend.FULL), (
                "Mixed chunk prefill requires a padded prefill CUDA graph "
                f"backend; got '{backend}'."
            )
        super(_NATIVE[0], self).__init__(model_runner)
        self.dllm_attention = model_runner.attn_backend.dllm_attention
        self._backend_can_run_prefill_cuda_graph = (
            self.dllm_attention.can_run_prefill_cuda_graph
            if self.dllm_attention is not None
            else getattr(model_runner.attn_backend, "can_run_prefill_cuda_graph", None)
        )
        self._backend_vote_prefill_cuda_graph = getattr(
            model_runner.attn_backend, "vote_prefill_cuda_graph", None
        )
        prefill_attn_backend = getattr(
            model_runner.attn_backend, "prefill_backend", model_runner.attn_backend
        )
        self._fa4_prefill = getattr(prefill_attn_backend, "fa_impl_ver", None) == 4
        # --- model flags ----------------------------------------------
        self.quant_config = getattr(model_runner.model, "quant_config", None)
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.is_dsv41_model = (
            model_runner.model_config.hf_config.model_type == "deepseek_v41"
        )
        self.enable_lora = model_runner.lora_manager is not None
        # Classification/reward forwards branch on return_pooled_hidden_states;
        # capture must use the same flag value as replay for those models.
        self.capture_return_pooled_hidden_states = not model_runner.is_generation

        # --- prefill graph config -------------------------------------
        prefill_config = N.get_exec().graph.cuda_graph_config.prefill
        self.prefill_backend_name = prefill_config.backend
        qwen_bcg = self.prefill_backend_name == N.Backend.BREAKABLE and N.is_qwen4_exp(
            model_runner.model_config.hf_config
        )
        self._qwen_bcg_hc_sidechannel = qwen_bcg and not model_runner.is_draft_worker
        self._qwen_bcg_mtp_draft = qwen_bcg and model_runner.is_draft_worker
        self.prefer_eager_mixed_prefill = (
            self.prefill_backend_name == N.Backend.BREAKABLE
            and N.get_parallel().attn_dp_enabled
            and getattr(
                model_runner.attn_backend,
                "prefer_eager_mixed_prefill_under_dp_attention",
                False,
            )
        )
        # bs in prefill carries the captured shape (token count for
        # tc_piecewise) — one shape knob per phase.
        capture_tokens = prefill_config.bs
        assert capture_tokens is not None, "cuda_graph_config[prefill].bs is not set"
        self.capture_num_tokens = N.get_prefill_num_tokens_to_capture(capture_tokens)
        assert self.capture_num_tokens, "cuda_graph_config[prefill].bs is empty"

        # --- runner bounds --------------------------------------------
        self.max_num_tokens = max(self.capture_num_tokens)
        self.max_bs = model_runner.req_to_token_pool.size
        self.max_context_size = prefill_config.max_context_size
        self.max_seq_len = prefill_config.max_seq_len
        if self.max_seq_len is None:
            self.max_seq_len = (
                getattr(model_runner.attn_backend, "prefill_cuda_graph_default_max_seq_len", None)
            )
            if self.max_seq_len is not None:
                N.logger.info(
                    "Prefill CUDA graph replays contexts up to %d tokens, the "
                    "attention backend's default; longer contexts run eager prefill.",
                    self.max_seq_len,
                )
        self._validate_max_context_capacity(
            max_context_size=self.max_context_size,
            table_width=model_runner.req_to_token_pool.req_to_token.shape[1],
        )
        if (
            self.prefill_backend_name == N.Backend.TC_PIECEWISE
            and self.max_context_size is not None
        ):
            # TODO(SYChen123): Plumb max_seq_len_override through TcPiecewise
            # metadata preparation before enabling the fixed context limit here.
            self._ignore_max_context_size("tc_piecewise prefill CUDA graph")

        # --- capture modes --------------------------------------------
        self.capture_forward_mode = N.ForwardMode.EXTEND
        # Hidden-state capture mode cases:
        # - Breakable EAGLE draft: LAST.
        # - EAGLE target: FULL.
        # - Return-hidden-states or DFLASH: FULL.
        # - Otherwise: NULL.
        is_eagle = model_runner.spec_algorithm.is_eagle()
        is_breakable_eagle_draft = (
            self.prefill_backend_name == N.Backend.BREAKABLE
            and is_eagle
            and model_runner.is_draft_worker
        )
        # A multimodal target prepares the EAGLE3 draft inputs as a merged
        # embedding sequence, so capture that path instead of baking
        # embed_tokens(input_ids). EAGLE(MTP) drafts are excluded since
        # they handle embeds in their own forward and reject an explicit
        # input_embeds argument.
        self._use_draft_input_embeds = (
            is_breakable_eagle_draft and model_runner.spec_algorithm.is_eagle3()
        )
        if is_breakable_eagle_draft:
            self.capture_hidden_mode = N.CaptureHiddenMode.LAST
        elif (is_eagle and not model_runner.is_draft_worker) or (
            model_runner.spec_algorithm.is_dflash_family()
        ):
            self.capture_hidden_mode = N.CaptureHiddenMode.FULL
        else:
            self.capture_hidden_mode = self.return_hidden_states_mode

        self.mamba_track_enabled = self._is_mamba_track_enabled()

        # --- buffers ---------------------------------------------------
        # `hidden_size` here sizes only the multimodal `input_embeds` buffer,
        # which `general_mm_embed_routine` copies the merged text+media
        # embeddings into. A model whose merge happens above the embedding width
        # (e.g. a residual-stream merge) writes a wider tensor than
        # `config.hidden_size`, so let it declare that width.
        input_embeds_hidden_size = self._input_embeds_hidden_size()
        self.buffers: N.PrefillInputBuffers = N.PrefillInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_tokens=self.max_num_tokens,
            cache_loc_dtype=self._cache_loc_dtype(),
            is_multimodal=self.is_multimodal,
            hidden_size=input_embeds_hidden_size,
            dtype=self.model_runner.dtype,
            enable_mamba_track=self.mamba_track_enabled,
            enable_input_embeds=(self.is_multimodal or self._use_draft_input_embeds),
            pp_size=self.pp_size,
            is_first_pp_rank=self.model_runner.pp_group.is_first_rank,
            hc_hidden_size=getattr(
                self.model_runner.model_config, "hc_hidden_size", None
            ),
            pp_proxy_topk_size=self.model_runner.get_pp_proxy_topk_size(),
            pp_proxy_residual_num_blocks=(
                self.model_runner.get_pp_proxy_residual_num_blocks()
            ),
            pp_proxy_dspark_hidden_size=(
                self.model_runner.get_pp_proxy_dspark_hidden_size()
            ),
        )
        self.buffers.share_buffers()
        # Token-axis FB-shared slot registry adopting PrefillInputBuffers
        # storage; same physical tensors, stable data_ptr for capture vs replay.
        self.buffer_registry: N.CudaGraphBufferRegistry = N.build_prefill_registry(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_tokens,
            cache_loc_dtype=self._cache_loc_dtype(),
            is_multimodal=self.is_multimodal,
            hidden_size=input_embeds_hidden_size,
            embed_dtype=self.model_runner.dtype,
            enable_mamba_track=self.mamba_track_enabled,
            # FullCG always pads to a capture bucket. Models that mask padded
            # hidden rows need the live boundary even without expert parallelism.
            enable_num_token_non_padded=(
                N.enable_num_token_non_padded()
                or self.prefill_backend_name == N.Backend.FULL
            ),
            require_gathered_buffer=N.require_gathered_buffer(),
            enable_prefill_cp=(N.is_dsa_enable_prefill_cp() or N.is_mla_cp_enabled()),
            attn_tp_sharded_fn=self.model_runner.attn_tp_sequence_sharded,
            register_input_embeds=(self.is_multimodal or self._use_draft_input_embeds),
            source=self.buffers,
        )

        self.attention_layers = self.model_runner.attention_layers
        self.mha_companion_layers = self.model_runner.mha_companion_layers
        self.has_mha_companion_layers = any(
            layer is not None for layer in self.mha_companion_layers
        )
        self.moe_layers = self.model_runner.moe_layers
        self.moe_fusions = self.model_runner.moe_fusions
        self.dsa_indexers = getattr(self.model_runner, "dsa_indexers", None)

        self.require_mlp_tp_gather = N.require_mlp_tp_gather()
        self.require_attn_tp_gather = N.require_attn_tp_gather()

        # --- backend ---------------------------------------------------
        # TcPiecewise resolves by running a compile pass that calls back into
        # capture_prepare / _run_forward, so these fields must exist first.
        self._prefill_static_buffers: Optional[Dict[str, N.torch.Tensor]] = None
        self.static_draft_hidden_states: Optional[N.torch.Tensor] = None
        self.layer_model = None
        self._capture_req_slots = 1
        # Same rationale: _run_compile_pass runs a dummy _run_forward before
        # resolve_prefill_backend returns, and that forward reads
        # self._is_full_backend. The compile-pass backend is never Full, so
        # default False; the assignment below sets the real value once the
        # backend type is known.
        self._is_full_backend = False
        # Same ordering requirement: capture_prepare reads this.
        self._capture_lora = False
        self.enable_cp_bcg_capture = False
        self.prefill_cp_bcg_input: Optional[N.PrefillCPBCGInput] = None
        # Auto mode may initialize AITER during the TC compile pass; prewarm
        # chip metadata before backend construction and capture.
        N.maybe_pre_warm_aiter_chip_info()
        try:
            self.backend = N.resolve_prefill_backend(self)
        except RuntimeError as e:
            if self.prefill_backend_name != N.Backend.TC_PIECEWISE:
                raise
            raise RuntimeError(
                f"Capture prefill CUDA graph failed: {e}\n"
                f"{N.prefill_failure_msg(self.prefill_backend_name)}"
            ) from e

        self._is_full_backend = isinstance(self.backend, N.FullCudaGraphBackend)
        if self._is_full_backend:
            max_req = prefill_config.full_prefill_max_req
            assert max_req is not None, "full_prefill_max_req must be resolved"
            self._capture_req_slots = max_req
            if self.dllm_attention is not None:
                self.dllm_attention.init_prefill_graph_state(
                    max_req, self.capture_num_tokens
                )

        # BCG/Full record LoRA kernels, so the metadata they read must live in
        # static buffers refreshed in place per batch; unsupported LoRA
        # configs were already routed to the eager runner.
        self._capture_lora = self.enable_lora and isinstance(
            self.backend, (N.BreakableCudaGraphBackend, N.FullCudaGraphBackend)
        )
        if self._capture_lora:
            model_runner.lora_manager.init_prefill_cuda_graph_batch_info(
                max_num_tokens=self.max_num_tokens,
                max_num_requests=(
                    self._capture_req_slots
                    if self._is_full_backend
                    else min(self.max_num_tokens, self.max_bs)
                ),
            )
        self._full_cg_seq_lens_cpu = (
            N.torch.zeros((self._capture_req_slots,), dtype=N.torch.int64, device="cpu")
            if self._is_full_backend
            else None
        )
        # This flag controls whether the model dispatches through the distinct
        # chunked-prefix topology; backend capability is validated separately.
        self._capture_chunked_prefix = (
            self._is_full_backend
            and not N.get_schedule().disable_chunked_prefix_cache
            and self.dllm_attention is None
        )
        self._prefix_chunk_len = 0
        self._prefix_chunk_capacity = 0
        self._prefix_max_len = 0
        self._prefix_capture_variants: tuple[int, ...] = ()
        self._prefix_capture_batches: Dict[N.ShapeKey, N.ForwardBatch] = {}
        self._prefix_capture_buffers: Optional[N._ChunkedPrefixCaptureBuffers] = None
        if self._capture_chunked_prefix:
            attn_backend = model_runner.attn_backend
            assert attn_backend.supports_full_cuda_graph_chunked_prefix, (
                f"{type(attn_backend).__name__} does not support chunked-prefix "
                "Full prefill CUDA graphs"
            )
            prefix_config = N.get_exec().graph.cuda_graph_config.prefill
            (
                self._prefix_chunk_len,
                self._prefix_chunk_capacity,
            ) = self._resolve_prefix_chunk_shape(model_runner, self._capture_req_slots)
            self._prefix_max_len = self._max_addressable_prefix_len(model_runner)
            max_real_chunks = N._ceil_div(self._prefix_max_len, self._prefix_chunk_len)
            # Variants are geometric, so n // 2 is the next-smaller variant; keep
            # n only if the smaller variant does not already cover the max prefix.
            self._prefix_capture_variants = tuple(
                n for n in N._CHUNKED_PREFIX_VARIANTS if n // 2 < max_real_chunks
            )
            self._prefix_capture_buffers = self._create_chunked_prefix_buffers()
            N.logger.info(
                "Full prefill CUDA graph cached-prefix chunks: "
                "%d aggregate tokens/chunk (%d/request x %d slots), "
                "variants=%s (%s)",
                self._prefix_chunk_capacity,
                self._prefix_chunk_len,
                self._capture_req_slots,
                self._prefix_capture_variants,
                (
                    "configured"
                    if prefix_config.full_prefill_prefix_chunk_tokens is not None
                    else "auto from chunked_prefill_size"
                ),
            )
        if isinstance(self.backend, (N.BreakableCudaGraphBackend, N.FullCudaGraphBackend)):
            with N.torch.device(self.device):
                self._prefill_static_buffers = {
                    name: N.torch.zeros((self.max_bs,), dtype=N.torch.int64)
                    for name in N._PREFILL_STATIC_FIELDS
                }

        server_args = model_runner.server_args
        self.enable_cp_bcg_capture = isinstance(
            self.backend, N.BreakableCudaGraphBackend
        ) and N.should_enable_cp_bcg_capture(server_args)
        if self.enable_cp_bcg_capture:
            if self.max_context_size is not None:
                # TODO(SYChen123): Preserve max_seq_len_override through CP's padded
                # metadata preparation before enabling the fixed context limit.
                self._ignore_max_context_size("CP breakable prefill CUDA graph")
            self.capture_num_tokens = N.filter_prefill_cp_bcg_capture_num_tokens(
                self.capture_num_tokens, server_args
            )
            self.prefill_cp_bcg_input = N.PrefillCPBCGInput.create(self)
            N.logger.info(
                "Prefill CP breakable CUDA graph enabled: strategy=%s, cp_size=%d, "
                "capture_num_tokens=%s",
                N.get_parallel().cp_strategy,
                N.get_parallel().attn_cp_size,
                self.capture_num_tokens,
            )
        if self.max_context_size is not None and not (
            model_runner.attn_backend.supports_prefill_cuda_graph_max_context_size
        ):
            raise ValueError(
                "--cuda-graph-prefill-max-context is only supported by attention "
                "backends that implement fixed-context prefill graph metadata; "
                f"got {type(model_runner.attn_backend).__name__}"
            )

        # Static hidden_states buffer giving the captured graph a stable
        # address; load_batch refreshes it from live spec_info at replay.
        # Draft consumes aux-concatenated hidden states from the target
        # (e.g. EAGLE3 stacks 3 target layers), so capture the pre-reduction
        # width when the draft model exposes it.
        if (
            isinstance(self.backend, N.BreakableCudaGraphBackend)
            and model_runner.is_draft_worker
            and model_runner.spec_algorithm.is_eagle()
        ):
            hidden_dim = N.get_draft_input_from_target_hidden_dim(model_runner)
            with N.torch.device(self.device):
                self.static_draft_hidden_states = N.torch.zeros(
                    (self.max_num_tokens, hidden_dim),
                    dtype=model_runner.dtype,
                )

        # Some attention backends (e.g. DSV4) opt into a captured-metadata
        # contract under BCG: capture-time builds a per-bucket metadata
        # object the backend then refreshes in place at replay. We honor
        # the contract only when the backend is Breakable; FullCG and
        # TC_PIECEWISE use the eager init_forward_metadata path.
        if isinstance(self.backend, N.BreakableCudaGraphBackend):
            self.use_captured_attn_metadata = model_runner.attn_backend.use_captured_forward_metadata_for_breakable_cuda_graph
        else:
            self.use_captured_attn_metadata = False
        self._captured_attn_metadata_max_bs = (
            model_runner.attn_backend.prefill_cuda_graph_max_batch_size
            if self.use_captured_attn_metadata
            else None
        )
        self.attn_metadata_buffers: Optional[Dict[N.ShapeKey, object]] = (
            {} if self.use_captured_attn_metadata else None
        )

        # BCG and Full CG capture only the transformer body (layer_model.forward),
        # not the LM head + logits_processor — the eager tail keeps the captured
        # graph bs-invariant so req_slots is not bound by an (req_slots, vocab) buffer.
        if isinstance(self.backend, (N.BreakableCudaGraphBackend, N.FullCudaGraphBackend)):
            try:
                self.layer_model = N._resolve_transformer_layer_model(
                    self.model_runner.model
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{type(self.backend).__name__} {exc} This backend is "
                    f"unsupported for this model architecture."
                ) from exc
            params = list(N.inspect.signature(self.layer_model.forward).parameters)
            self._input_embeds_arg_idx = next(
                (
                    params.index(name)
                    for name in ("input_embeds", "inputs_embeds")
                    if name in params
                ),
                None,
            )

        # --- capture --------------------------------------------------
        self.device_module.synchronize()
        N.get_parallel().tp_group.barrier()
        self.capture()

        self.raw_num_tokens = 0
        self.raw_bs = 0

    @N.contextmanager
    def _prefill_forward_context(
        self,
        forward_batch: N.ForwardBatch,
        *,
        num_tokens: Optional[int] = None,
        raw_num_tokens: Optional[int] = None,
    ):
        with (
            N.set_forward_batch(forward_batch),
            prefill_graph_scope(
                full_graph=self._is_full_backend, raw_num_tokens=raw_num_tokens
            ),
            N.forward_context(
                N.ForwardContext(attn_backend=self.model_runner.attn_backend)
            ),
            N.set_tc_piecewise_forward_context(
                forward_batch,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                dsa_indexers=self.dsa_indexers,
                mha_companion_layers=self.mha_companion_layers,
                num_tokens=num_tokens,
                raw_num_tokens=raw_num_tokens,
                full_graph=self._is_full_backend,
            ),
        ):
            yield

    def run_dummy_multimodal_deepstack_forward(
        self, language_model: N.torch.nn.Module, num_tokens: int
    ) -> bool:
        """Warm the tensor-valued deepstack branch before serving requests.

        The regular PCG dummy is text-only. Qwen3-VL only provides
        ``input_deepstack_embeds`` after visual encoding, so leaving this
        branch cold makes the first image request synchronously recompile the
        language model. The model/signature checks keep this a no-op for
        non-deepstack architectures.
        """
        if (
            "input_deepstack_embeds"
            not in N.inspect.signature(language_model.forward).parameters
        ):
            return False

        num_deepstack = getattr(self.model_runner.model, "num_deepstack_embeddings", 0)
        if num_deepstack <= 0:
            return False

        hidden_size = (
            getattr(getattr(language_model, "config", None), "hidden_size", None)
            or self.model_runner.model_config.hidden_size
        )
        fb, attn_backend = self.capture_prepare(num_tokens)
        attn_backend.init_forward_metadata(fb)
        deepstack_embeds = N.torch.zeros(
            (num_tokens, hidden_size * num_deepstack),
            dtype=self.model_runner.dtype,
            device=self.device,
        )
        N.torch._dynamo.maybe_mark_dynamic(deepstack_embeds, 0)

        fb.dp_local_start_pos = fb.dp_local_num_tokens = None
        N.set_dp_buffer_len(
            fb.global_dp_buffer_len,
            num_tokens,
            fb.dp_padding_mode.is_max_len(),
            fb.global_num_tokens_cpu,
        )
        N.set_is_extend_in_batch(False)

        with (
            N.set_forward_batch(fb),
            prefill_graph_scope(full_graph=self._is_full_backend),
            N.forward_context(
                N.ForwardContext(attn_backend=self.model_runner.attn_backend)
            ),
            N.set_tc_piecewise_forward_context(
                fb,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                dsa_indexers=self.dsa_indexers,
            ),
        ):
            language_model.forward(
                fb.input_ids,
                self._get_layer_model_positions(fb),
                fb,
                input_embeds=fb.input_embeds,
                input_deepstack_embeds=deepstack_embeds,
            )
        return True

    def can_replay_locally(
        self,
        *,
        batch_size: int,
        num_tokens: Optional[int],
        input_embeds,
        replace_embeds,
        prefix_lens,
        is_target_verify: bool,
        capture_hidden_mode,
        return_logprob: bool,
        lora_ineligible: bool = False,
        is_mixed: bool = False,
        batch_max_context_len: Optional[int] = None,
        contains_mm_inputs: bool = False,
    ) -> bool:
        """Rank-local replay eligibility: the single source of truth for
        ``can_run_graph`` (ForwardBatch, forward time) and the dp mlp-sync
        vote (ScheduleBatch, schedule time) — all dp ranks must reach the
        same replay-vs-eager decision or their collectives mismatch. Pass
        ``capture_hidden_mode=None`` when unknown at the call site (it is
        rank-uniform; forward-time-only checking cannot split the group).
        """
        if contains_mm_inputs and (
            self._qwen_bcg_hc_sidechannel or self._qwen_bcg_mtp_draft
        ):
            return False
        # Full replay bypasses forward_extend's Python image-mask guard.
        # Keep FA4 multimodal prefill eager until its full-graph path is
        # supported; both forward dispatch and the DP vote use this policy.
        if self._is_full_backend and self._fa4_prefill and contains_mm_inputs:
            return False
        if self._is_full_backend and batch_size > self._capture_req_slots:
            return False
        if (
            self._captured_attn_metadata_max_bs is not None
            and batch_size > self._captured_attn_metadata_max_bs
        ):
            return False
        # LoRA replays need prepare_lora_batch's static metadata. lora_manager
        # keeps LoRA prefill eager on every rank under dp attention, so the
        # schedule-time vote derives this from enable_lora alone.
        if lora_ineligible:
            return False
        if is_mixed and getattr(self, "prefer_eager_mixed_prefill", False):
            return False
        if input_embeds is not None:
            return False
        if replace_embeds is not None:
            return False
        # Off CUDA, BCG takes the MHA companion, whose prefix path is
        # uncapturable. DSA is exempt: it stays on sparse MLA under BCG.
        if (
            self.prefill_backend_name == N.Backend.BREAKABLE
            and self.has_mha_companion_layers
            and not N.is_cuda()
            and prefix_lens is not None
            and any(prefix_lens)
            and not self._attn_backend_is_dsa
        ):
            return False
        # FullCG's chunked-prefix topology covers a bounded prefix. Its capture
        # flag is FullCG-only, so this is inert for the BreakableCG vote path.
        if self._has_uncapturable_chunked_prefix(prefix_lens):
            return False
        # tc_piecewise captures with ForwardMode.EXTEND and spec_info=None.
        if is_target_verify:
            return False
        if (
            capture_hidden_mode is not None
            and self.capture_hidden_mode < capture_hidden_mode
        ):
            return False
        if return_logprob and not self._uses_eager_prefill_tail():
            return False
        # Vote on both bounds so DP ranks agree on graph versus eager.
        for context_limit in (self.max_context_size, self.max_seq_len):
            if context_limit is not None and (
                batch_max_context_len is None or batch_max_context_len > context_limit
            ):
                return False
        if num_tokens is None:
            return True
        if num_tokens > self.max_num_tokens:
            return False
        # No exact-shape check: load_batch bucket-pads; only reject
        # disproportionate padding waste.
        padded_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        if padded_num_tokens > num_tokens * N._MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR:
            return False
        return True

    def can_run_graph(self, forward_batch: N.ForwardBatch) -> bool:
        # DP check: group verdict from the schedule-time all-gather
        # (min-reduced votes; also requires every rank to hold tokens).
        if (
            forward_batch.global_num_tokens_cpu is not None
            and not forward_batch.can_run_dp_prefill_cuda_graph
        ):
            return False

        # Every dp rank must hold tokens this forward (reads the synced
        # table post dp-padding; idle ranks vote permissively upstream).
        if self._has_inactive_dp_rank(forward_batch):
            return False

        # A batch can be too small to activate CP. Its unsharded inputs must
        # not replay a model body captured with CP-local rows.
        if getattr(self, "enable_cp_bcg_capture", False) and not N.is_cp_active(
            forward_batch
        ):
            return False

        # Non-DP local check (sole decision for tp-only).
        batch_max_context_len = (
            self._batch_max_context_len(forward_batch)
            if self.max_context_size is not None or self.max_seq_len is not None
            else None
        )
        if not self.can_replay_locally(
            batch_size=forward_batch.batch_size,
            num_tokens=len(forward_batch.input_ids),
            input_embeds=forward_batch.input_embeds,
            replace_embeds=forward_batch.replace_embeds,
            prefix_lens=forward_batch.extend_prefix_lens_cpu,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            return_logprob=forward_batch.return_logprob,
            contains_mm_inputs=forward_batch.contains_mm_inputs(),
            lora_ineligible=self.enable_lora
            and not (
                self._capture_lora
                and self.model_runner.lora_manager.can_use_prefill_cuda_graph(
                    forward_batch
                )
            ),
            is_mixed=any(
                getattr(forward_batch, field, None) == N.ForwardMode.MIXED
                for field in (
                    "forward_mode",
                    "global_forward_mode",
                    "_original_forward_mode",
                )
            ),
            batch_max_context_len=batch_max_context_len,
        ):
            return False
        if getattr(self, "enable_cp_bcg_capture", False):
            if not N.is_cp_active(forward_batch):
                return False
            assert self.prefill_cp_bcg_input is not None
            if (
                self.prefill_cp_bcg_input.select_replay_bucket_for_batch(
                    num_tokens=len(forward_batch.input_ids),
                    extend_seq_lens=forward_batch.extend_seq_lens_cpu,
                    capture_num_tokens=self.capture_num_tokens,
                    max_padding_factor=N._MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR,
                )
                is None
            ):
                return False
        backend_can_run = self._backend_can_run_prefill_cuda_graph
        if backend_can_run is not None and not backend_can_run(forward_batch):
            return False
        # Multi-req replay is supported by body-capture backends via the
        # layer_model.forward monkey-patch in replay(): the captured graph runs
        # the transformer stack, then the outer model.forward runs
        # logits_processor eagerly on top with live request metadata.
        return True

    def capture_one_shape(self, size: int, *, prefix_num_chunks: int = 0) -> None:
        """Per-shape capture: build dummy ForwardBatch + run_once,
        delegate to backend. size is the prefill token count.
        """
        num_tokens = size
        forward_batch, attn_backend = self.capture_prepare(num_tokens)
        # Recapture can follow decode replay, which may change model-owned storage.
        prepare = getattr(self.model_runner.model, "prepare_for_forward", None)
        if prepare is not None:
            prepare(forward_batch)
        if self.enable_cp_bcg_capture:
            assert self.prefill_cp_bcg_input is not None
            self.prefill_cp_bcg_input.prepare(
                self,
                forward_batch,
                static_num_tokens=num_tokens,
                capture=True,
            )
        if forward_batch.lora_ids is not None:
            # Fill the static prefill LoRA batch info the captured kernels
            # will read (all-None ids: ranks stay 0, kernels no-op).
            lora_manager = self.model_runner.lora_manager
            assert lora_manager.can_use_prefill_cuda_graph(forward_batch), (
                f"Capture batch (req slots {self._capture_req_slots}, bucket "
                f"{num_tokens}) exceeds the LoRA backend's prefill CUDA graph "
                "limits; the graph would read stale LoRA metadata at replay."
            )
            lora_manager.prepare_lora_batch(forward_batch)
        shape_key = N.ShapeKey(
            size=num_tokens,
            variant_label=(
                N._chunked_prefix_variant(prefix_num_chunks)
                if prefix_num_chunks
                else None
            ),
        )
        if prefix_num_chunks:
            self._prepare_chunked_prefix_capture(
                forward_batch, shape_key, prefix_num_chunks
            )
        if self._is_full_backend:
            if not prefix_num_chunks:
                attn_backend.init_forward_metadata_out_graph(
                    forward_batch, in_capture=True
                )
            # The prefix variant intentionally reuses the capture-stable
            # metadata object initialized by the suffix-only variant above.
            # Reaching into a backend-specific metadata cache here would make
            # this path incompatible with the OSS FlashAttention backend.
        else:
            self._init_forward_metadata_for_capture(forward_batch, shape_key)

        def run_once():
            # Record LoRA kernels even when capture uses base-model requests.
            with (
                N.model_capture_mode()
                if self._is_full_backend and self._capture_lora
                else N.nullcontext()
            ):
                return self._run_forward(forward_batch, num_tokens)

        # Main's monolithic BCG runner never invokes
        # on_after_cuda_graph_warmup between warmup iterations — the BCG
        # contract is to keep warmup metadata untouched and let
        # init_forward_metadata_in_graph (recorded inside the captured
        # forward) do any raw->full upgrade. cg-refactor's runner_backend
        # abstraction exposes a post_warmup_hook for backends that need
        # workspace cleanup between iterations; suppress it for BCG so
        # DSV4's hook (which restores forward_metadata to a stale
        # _current_capture_raw left over from decode CG capture) doesn't
        # corrupt warmup iter 2's metadata read.
        if isinstance(self.backend, N.BreakableCudaGraphBackend):
            post_warmup_hook = None
            req_pool = self.model_runner.req_to_token_pool
            mamba_pool = getattr(req_pool, "mamba_pool", None)
            if mamba_pool is not None and not prefix_num_chunks:
                capture_state_indices = req_pool.translate_mamba_indices(
                    req_pool.get_mamba_indices(forward_batch.req_pool_indices)
                ).unique()

                def post_warmup_hook():
                    mamba_pool.clear_slots(capture_state_indices)

                post_warmup_hook()
        else:
            post_warmup_hook = getattr(attn_backend, "on_after_cuda_graph_warmup", None)
        self.backend.capture_one(
            shape_key,
            run_once,
            # DP padding can install capture-only tensors on this dummy batch;
            # BCG retains it so their recorded addresses remain valid.
            capture_inputs=(
                forward_batch
                if forward_batch.global_num_tokens_gpu is not None
                else None
            ),
            post_warmup_hook=post_warmup_hook,
        )

    def load_batch(self, forward_batch: N.ForwardBatch, **kwargs) -> N.ForwardBatch:
        """Pad, populate static buffers, and build the static_forward_batch
        the model code reads during replay.
        """
        num_tokens = len(forward_batch.input_ids)
        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        if getattr(self, "enable_cp_bcg_capture", False) and N.is_cp_active(
            forward_batch
        ):
            assert self.prefill_cp_bcg_input is not None
            static_num_tokens = (
                self.prefill_cp_bcg_input.select_replay_bucket_for_batch(
                    num_tokens=num_tokens,
                    extend_seq_lens=forward_batch.extend_seq_lens_cpu,
                    capture_num_tokens=self.capture_num_tokens,
                    max_padding_factor=N._MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR,
                )
            )
            if static_num_tokens is None:
                raise RuntimeError(
                    "Prefill CUDA graph replay was admitted without a fitting bucket"
                )
        self.raw_num_tokens = num_tokens

        if self.max_context_size is not None:
            batch_max_context_len = self._batch_max_context_len(forward_batch)
            if (
                batch_max_context_len is None
                or batch_max_context_len > self.max_context_size
            ):
                raise RuntimeError(
                    "Prefill CUDA graph replay was admitted without a fitting "
                    "maximum context size"
                )

        bs = forward_batch.batch_size
        self.raw_bs = bs

        self.buffer_registry.fill_from(
            forward_batch,
            raw_bs=bs,
            padded_bs=bs,
            raw_num_tokens=num_tokens,
            padded_num_tokens=static_num_tokens,
            pp_proxy_tensors=kwargs.get("pp_proxy_tensors"),
        )

        if self._use_draft_input_embeds:
            # Keep one address-stable embedding input for the captured draft
            # body. Multimodal requests use the target-composed embeds with
            # the tail-token row replaced, and text requests use an
            # embed_tokens lookup.
            draft_input_embeds = forward_batch.mm_input_embeds
            if draft_input_embeds is not None:
                # Replace each request's last row with the embedding of its
                # sampled tail token, mirroring the native EAGLE drafts.
                last_indices = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                draft_input_embeds[last_indices] = self.layer_model.embed_tokens(
                    forward_batch.input_ids[last_indices]
                )
            else:
                draft_input_embeds = self.layer_model.embed_tokens(
                    forward_batch.input_ids
                )
            self.buffer_registry.get_slot("input_embeds").buffer[:num_tokens].copy_(
                draft_input_embeds
            )

        registry = self.buffer_registry

        def _slot(name):
            return registry.get_slot(name).slice_for(bs, static_num_tokens)

        mamba_track_indices = (
            _slot("mamba_track_indices")
            if registry.has_slot("mamba_track_indices")
            else None
        )
        mamba_track_mask = (
            _slot("mamba_track_mask") if registry.has_slot("mamba_track_mask") else None
        )
        mamba_track_seqlens = (
            _slot("mamba_track_seqlens")
            if registry.has_slot("mamba_track_seqlens")
            else None
        )

        input_ids = _slot("input_ids")
        input_embeds = (
            _slot("input_embeds") if registry.has_slot("input_embeds") else None
        )
        positions = _slot("positions")
        out_cache_loc = _slot("out_cache_loc")
        mrope_positions = (
            _slot("mrope_positions")
            if registry.has_slot("mrope_positions")
            and forward_batch.mrope_positions is not None
            else None
        )
        num_token_non_padded = (
            _slot("num_token_non_padded")
            if registry.has_slot("num_token_non_padded")
            else forward_batch.num_token_non_padded
        )

        # MIXED replays the EXTEND-captured graphs.
        pcg_forward_mode = (
            N.ForwardMode.EXTEND
            if forward_batch.forward_mode == N.ForwardMode.MIXED
            else forward_batch.forward_mode
        )
        pcg_global_forward_mode = (
            N.ForwardMode.EXTEND
            if forward_batch.global_forward_mode == N.ForwardMode.MIXED
            else forward_batch.global_forward_mode
        )

        # The draft tail concatenates hidden states with padded input embeddings;
        # expose the bucket-sized static view and refresh its live prefix below.
        padded_spec_info = forward_batch.spec_info
        if (
            self.static_draft_hidden_states is not None
            and padded_spec_info is not None
            and getattr(padded_spec_info, "hidden_states", None) is not None
        ):
            padded_spec_info = N.dataclasses.replace(
                padded_spec_info,
                hidden_states=self.static_draft_hidden_states[:static_num_tokens],
            )

        static_forward_batch = N.ForwardBatch(
            forward_mode=pcg_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            next_token_logits_buffer=self._next_token_logits_buffer(
                self._prefill_logits_buffer_rows(forward_batch)
            ),
            orig_seq_lens=forward_batch.orig_seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=forward_batch.seq_lens_sum,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            encoder_lens=forward_batch.encoder_lens,
            return_logprob=(
                forward_batch.return_logprob
                if isinstance(self.backend, N.BreakableCudaGraphBackend)
                else False
            ),
            is_prefill_only=forward_batch.is_prefill_only,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_start_loc=forward_batch.extend_start_loc,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            multi_item_delimiter_indices=forward_batch.multi_item_delimiter_indices,
            extend_num_tokens=forward_batch.extend_num_tokens,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            positions=positions,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            dp_padding_mode=forward_batch.dp_padding_mode,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=padded_spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            num_token_non_padded=num_token_non_padded,
            global_num_token_non_padded_cpu=forward_batch.global_num_token_non_padded_cpu,
            attn_tp_sequence_sharded=self.model_runner.attn_tp_sequence_sharded(
                static_num_tokens
            ),
            global_forward_mode=pcg_global_forward_mode,
            lora_ids=forward_batch.lora_ids,
            sampling_info=forward_batch.sampling_info,
            mm_inputs=forward_batch.mm_inputs,
            # Multimodal preparation consumes mm_inputs but retains embeddings for
            # later MTP draft extend, so copy that side channel into the replay view.
            mm_input_embeds=forward_batch.mm_input_embeds,
            temperature=forward_batch.temperature,
            top_p=forward_batch.top_p,
            dimensions=forward_batch.dimensions,
            return_pooled_hidden_states=(
                self.capture_return_pooled_hidden_states
                or forward_batch.return_pooled_hidden_states
            ),
            max_seq_len_override=self.max_context_size,
        )
        _plan = getattr(forward_batch, "kv_loc_plan", None)
        if _plan is None:
            # 1be97a0 renamed KVLocPlan -> IterationIds.
            _plan = forward_batch.iteration_ids
        _translator = getattr(self.model_runner, "kv_index_translator", None)
        if _translator is None:
            _translator = self.model_runner.runner_ids
        _plan.bind_replay(
            static_forward_batch,
            _translator,
            slots=out_cache_loc,
        )
        # The n-gram hasher runs outside the graph and reads this at replay.
        static_forward_batch.ngram_embedding_info = forward_batch.ngram_embedding_info
        static_forward_batch.engram_history = forward_batch.engram_history
        # The encoder replay fold: a replay builds the same window floors and
        # compressor slots as an eager prefill.
        static_forward_batch.encoder_swa_replay = forward_batch.encoder_swa_replay
        static_forward_batch.encoder_swa_row_floor = forward_batch.encoder_swa_row_floor
        static_forward_batch.encoder_swa_compress_rows = (
            forward_batch.encoder_swa_compress_rows
        )
        static_forward_batch = self.model_runner.prepare_dummy_forward_batch(
            static_forward_batch
        )
        if self._is_full_backend:
            forward_batch.next_token_logits_buffer = (
                static_forward_batch.next_token_logits_buffer
            )

        if (
            isinstance(self.backend, N.BreakableCudaGraphBackend)
            and self.has_mha_companion_layers
            and not N.is_cuda()
        ):
            self._restore_mha_capture_state(static_forward_batch)

        # Under Breakable / Full, copy serving-time values into the static
        # buffers so the addresses captured segments hold stay live with
        # current data.
        if self._prefill_static_buffers is not None:
            bs = forward_batch.batch_size
            s = self._prefill_static_buffers
            s["seq_lens"][:bs].copy_(forward_batch.seq_lens)
            s["extend_seq_lens"][:bs].copy_(forward_batch.extend_seq_lens)
            s["extend_prefix_lens"][:bs].copy_(forward_batch.extend_prefix_lens)
            s["extend_start_loc"][:bs].copy_(forward_batch.extend_start_loc)
            s["req_pool_indices"][:bs].copy_(forward_batch.req_pool_indices)
            if forward_batch.orig_seq_lens is not None:
                s["orig_seq_lens"][:bs].copy_(forward_batch.orig_seq_lens)
            if self._is_full_backend and bs < self._capture_req_slots:
                # Sentinel tail for slots [bs:req_slots]: the captured graph
                # reads all req_slots entries (e.g. the logits-processor
                # cumsum), so stale values from the previous replay must be
                # cleared. Zero lengths make the sentinels no-ops;
                # extend_start_loc sentinels sit at the flat end of the real
                # tokens.
                r = self._capture_req_slots
                s["seq_lens"][bs:r].zero_()
                s["extend_seq_lens"][bs:r].zero_()
                s["extend_prefix_lens"][bs:r].zero_()
                s["extend_start_loc"][bs:r].fill_(self.raw_num_tokens)
                s["req_pool_indices"][bs:r].zero_()
                s["orig_seq_lens"][bs:r].zero_()
                # The captured track scatter reads these rows too, and a stale
                # mask row still carries a live destination slot: it would land
                # this replay's window in an earlier request's checkpoint.
                registry = self.buffer_registry
                for name in ("mamba_track_mask", "mamba_track_indices"):
                    if registry.has_slot(name):
                        registry.get_slot(name).buffer[bs:r].zero_()

        # Refresh the static buffer the captured graph reads from.
        if (
            self.static_draft_hidden_states is not None
            and forward_batch.spec_info is not None
        ):
            self.static_draft_hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )

        metadata_forward_batch = forward_batch
        if self.enable_cp_bcg_capture:
            assert self.prefill_cp_bcg_input is not None
            self.prefill_cp_bcg_input.prepare(
                self,
                static_forward_batch,
                static_num_tokens=static_num_tokens,
                capture=False,
            )
            metadata_forward_batch = static_forward_batch

        shape_key = self._shape_key(static_num_tokens, forward_batch)
        self._prepare_forward_metadata_for_replay(
            metadata_forward_batch, static_forward_batch, shape_key
        )

        return static_forward_batch

    def _execute_body_capture(
        self,
        forward_batch: N.ForwardBatch,
        static_forward_batch: N.ForwardBatch,
        static_num_tokens: int,
        raw_num_tokens: int,
        shape_key: N.ShapeKey,
        **kwargs,
    ):
        # BCG / Full: replay the captured body, run the LM head +
        # logits_processor eagerly.
        full_path = self._is_full_backend

        def replay_layer_forward(*args, **layer_kwargs):
            prepared_ids = layer_kwargs.get("input_ids", args[0] if args else None)
            # TODO: Add general multimodal replay tests before enabling for other models.
            if self.is_dsv41_model and prepared_ids is not None:
                static_forward_batch.input_ids[: prepared_ids.shape[0]].copy_(
                    prepared_ids
                )
            # The captured body graph reads activations from the static
            # input_embeds slot. The outer model.forward (run eagerly)
            # passes the live embeddings into layer_model.forward as the
            # 4th positional arg (or input_embeds kwarg): for multimodal
            # batches these are the composed text+vision embeds, for
            # text-only batches they are get_input_embeddings()(input_ids).
            # Copy them into the slot before replay so the graph sees the
            # current request's embeddings (mirrors main's BCG closure).
            # The Qwen MTP draft passes its HC stream as inputs_embeds;
            # its slot is registered at hidden_size * hc_count width.
            if (
                self.model_runner.pp_group.is_first_rank
                and self.buffer_registry.has_slot("input_embeds")
            ):
                self._fill_input_embeds_slot(args, layer_kwargs, static_num_tokens)
            hs = self.backend.replay(shape_key, static_forward_batch, **kwargs)
            hs = self._restore_qwen_bcg_hc_output(hs)
            return N._slice_output_rows(hs, raw_num_tokens) if full_path else hs

        # Same MegaMoE valid-token refresh the decode graph runner performs.
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if get_moe_a2a_backend().is_megamoe():
            from hcu_sglang.mega_moe_graph import (
                set_mega_moe_cuda_graph_num_tokens,
            )

            set_mega_moe_cuda_graph_num_tokens(raw_num_tokens)

        original_layer_forward = self.layer_model.forward
        self.layer_model.forward = replay_layer_forward
        if full_path:
            # Run the eager tail against the raw user-facing batch so it uses
            # real request metadata instead of padded slots. Its SP verdict came
            # from the unpadded count, though, while the captured body froze one
            # at the padded bucket count; the two can differ, so carry the body's
            # on a private view rather than mutating the caller's batch.
            tail_batch = N.copy.copy(forward_batch)
            tail_batch.attn_tp_sequence_sharded = (
                static_forward_batch.attn_tp_sequence_sharded
            )
        else:
            # BCG has no request-slot padding, so static_forward_batch is
            # already the serving batch and already holds the body's verdict.
            tail_batch = static_forward_batch
        if not full_path:
            # MTP consumes the target model's live multimodal embeddings in its
            # eager wrapper before the captured transformer body is replayed.
            tail_batch.mm_input_embeds = forward_batch.mm_input_embeds
        model_kwargs = kwargs
        if self._use_draft_input_embeds:
            # Draft forwards that accept `input_embeds` (all current EAGLE
            # draft classes) route it into the draft transformer, selecting
            # the same explicit-embedding branch as capture instead of
            # replaying the captured token lookup.
            model_kwargs = {
                **kwargs,
                "input_embeds": static_forward_batch.input_embeds,
            }
        try:
            with self._prefill_forward_context(
                static_forward_batch,
                num_tokens=static_num_tokens,
                raw_num_tokens=raw_num_tokens,
            ):
                return self.model_runner.model.forward(
                    tail_batch.input_ids,
                    tail_batch.positions,
                    tail_batch,
                    **model_kwargs,
                )
        finally:
            self.layer_model.forward = original_layer_forward

    def _trim_logits_output(
        self, output: N.LogitsProcessorOutput
    ) -> N.LogitsProcessorOutput:
        # Preserve mm_input_embeds for speculative decoding.
        mm_input_embeds = None
        if (
            self.model_runner.spec_algorithm.is_speculative()
            and output.mm_input_embeds is not None
        ):
            mm_input_embeds = output.mm_input_embeds[: self.raw_num_tokens]
        logits_rows = self.raw_bs if self._is_full_backend else self.raw_num_tokens
        return N.LogitsProcessorOutput(
            next_token_logits=(
                output.next_token_logits[:logits_rows]
                if output.next_token_logits is not None
                else None
            ),
            hidden_states=(
                output.hidden_states[: self.raw_num_tokens]
                if output.hidden_states is not None
                else None
            ),
            draft_topk_index=(
                output.draft_topk_index[:logits_rows]
                if output.draft_topk_index is not None
                else None
            ),
            input_token_logprobs=output.input_token_logprobs,
            input_top_logprobs_val=output.input_top_logprobs_val,
            input_top_logprobs_idx=output.input_top_logprobs_idx,
            input_token_ids_logprobs_val=output.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=output.input_token_ids_logprobs_idx,
            input_logprobs_copy_done=output.input_logprobs_copy_done,
            customized_info=output.customized_info,
            mm_input_embeds=mm_input_embeds,
            # Bounded SWA replay returns the late layers' tail rows plus the
            # extend indices they came from; the DSpark injector needs both.
            hidden_states_token_indices=output.hidden_states_token_indices,
        )

    def _wait_for_layer_loads(self) -> None:
        # Full replay skips the per-layer load waits in the pool accessors; breakable
        # replay still runs them at its eager breaks.
        if not self._is_full_backend:
            return
        counter = self.model_runner.token_to_kv_pool.layer_transfer_counter
        if counter is not None and counter.consumer_index >= 0:
            counter.wait_until(counter.num_layers - 1)

    def execute(
        self, forward_batch: N.ForwardBatch, **kwargs
    ) -> Optional[Union[N.LogitsProcessorOutput, N.PPProxyTensors, N.EmbeddingPoolerOutput]]:
        self._validate_capture_hidden_mode(forward_batch)
        with self.backend.replay_session():
            static_forward_batch = self.load_batch(forward_batch, **kwargs)
            static_num_tokens = len(static_forward_batch.input_ids)
            raw_num_tokens = self.raw_num_tokens
            shape_key = self._shape_key(static_num_tokens, forward_batch)
            # The only variants this runner records are chunked-prefix ones.
            if shape_key.variant_label is not None:
                self._prepare_chunked_prefix_replay(shape_key, forward_batch)
            # Replay prep, including the optional chunked-prefix gather above,
            # has finished every scheduler-shared read.
            N.maybe_publish_prefill_shared_read_done(
                self.model_runner, forward_batch, self.device_module
            )
            self._wait_for_layer_loads()

            if self.enable_cp_bcg_capture:
                assert self.max_context_size is None, (
                    "CP-v2 BCG replay does not support a fixed prefill context size"
                )
                output = N.execute_prefill_cp_bcg(
                    self,
                    forward_batch,
                    static_forward_batch,
                    static_num_tokens,
                    raw_num_tokens,
                    **kwargs,
                )
            elif self._uses_eager_prefill_tail():
                output = self._execute_body_capture(
                    forward_batch,
                    static_forward_batch,
                    static_num_tokens,
                    raw_num_tokens,
                    shape_key,
                    **kwargs,
                )
            else:
                output = self._execute_tc_piecewise(
                    static_forward_batch,
                    static_num_tokens,
                    raw_num_tokens,
                    **kwargs,
                )
            return self._finalize_execute_output(output)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.model_executor.runner.prefill_cuda_graph_runner.PrefillCudaGraphRunner", PrefillCudaGraphRunner, HookType.REPLACE)
