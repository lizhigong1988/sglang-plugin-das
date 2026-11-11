"""HCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

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

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C
from sglang.srt import utils as U
import sglang.srt.models.deepseek_v2 as N0
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla as N1
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha as N2
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_fused_rope_rocm as N3
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_fused_rope_cpu as N4


def __init__(
    self,
    config: N0.PretrainedConfig,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    rope_theta: float = 10000,
    rope_scaling: N0.Optional[N0.Dict[str, N0.Any]] = None,
    max_position_embeddings: int = 8192,
    quant_config: N0.Optional[N0.QuantizationConfig] = None,
    reduce_results: bool = True,
    layer_id: int = None,
    prefix: str = "",
    alt_stream: N0.Optional[N0.torch.cuda.Stream] = None,
    skip_rope: bool = False,
    is_nextn: bool = False,
) -> None:
    super(N0.DeepseekV2AttentionMLA, self).__init__()
    from sglang.srt.configs.model_config import is_glm_moe_dsa
    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
    from sglang.srt.layers.moe.utils import is_sbo_enabled

    use_native_glm = C._is_hcu and is_glm_moe_dsa(config)
    if use_native_glm:
        allow_alt_stream = (
            N0.envs.SGLANG_NPU_USE_MULTI_STREAM.get() or is_sbo_enabled()
            if is_nextn
            else (
                N0.envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                or N0.envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                or (
                    is_dsa_enable_prefill_cp()
                    and N0.get_parallel().attn_cp_size > 1
                )
            )
        )
        if not allow_alt_stream:
            alt_stream = None
    self.layer_id = layer_id
    self.hidden_size = hidden_size
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.quant_config = quant_config
    self.is_nextn = is_nextn
    attn_tp_size = N0.get_parallel().attn_tp_size
    self.use_dsa = N0.is_deepseek_dsa(config)
    self.dcp_size = (
        1 if (is_nextn and self.use_dsa) else N0.get_parallel().attn_dcp_size
    )
    self.num_heads = num_heads
    assert num_heads % attn_tp_size == 0
    self.num_local_heads = num_heads // attn_tp_size
    self.scaling = self.qk_head_dim**-0.5
    self.rope_theta = rope_theta
    self.max_position_embeddings = max_position_embeddings
    self.kv_cache_dtype = N0.get_model().kv_cache_dtype

    # NOTE modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
    if rope_scaling:
        rope_scaling["rope_type"] = "deepseek_yarn"

    # For tensor parallel attention
    if self.q_lora_rank is not None:
        self.fused_qkv_a_proj_with_mqa = N0.ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=N0.add_prefix("fused_qkv_a_proj_with_mqa", prefix),
        )
        self.q_a_layernorm = N0.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = N0.ColumnParallelLinear(
            q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=self._get_q_b_proj_quant_config(quant_config),
            prefix=N0.add_prefix("q_b_proj", prefix),
            parallel_group="attn_tp",
        )
    else:
        self.q_proj = N0.ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=N0.add_prefix("q_proj", prefix),
            parallel_group="attn_tp",
        )
        self.kv_a_proj_with_mqa = N0.ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=N0.add_prefix("kv_a_proj_with_mqa", prefix),
        )

    self.skip_topk = None
    self.next_skip_topk = None
    self.indexer = None
    if self.use_dsa:
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
        # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
        if is_nextn:
            self.skip_topk = True
            self.next_skip_topk = True
        else:
            self.skip_topk = N0.dsa_layer_skips_topk(config, layer_id)
            self.next_skip_topk = N0.dsa_layer_skips_topk(config, layer_id + 1)

        if not self.skip_topk or is_nextn:
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            # 1be97a0: deepseek_v2 no longer re-exports Indexer; the class lives
            # in layers.attention.dsa.dsa_indexer.
            if not hasattr(N0, "Indexer"):
                from sglang.srt.layers.attention.dsa.dsa_indexer import (
                    Indexer as _DsaIndexer,
                )

                N0.Indexer = _DsaIndexer
            indexer_cls = (
                N0.IndexerKPool if N0.get_dsa_index_kpool(config) > 1 else N0.Indexer
            )
            from hcu_sglang.models.deepseek.deepseek_common.model_types import is_glm5_next_hcu

            if is_glm5_next_hcu(config):
                from sglang.srt.layers.attention.glm5_next.indexer import (
                    Indexer as GlmIndexer,
                )
                from sglang.srt.layers.attention.glm5_next.kpool.indexer import (
                    IndexerKPool as GlmIndexerKPool,
                )

                indexer_cls = (
                    GlmIndexerKPool
                    if N0.get_dsa_index_kpool(config) > 1
                    else GlmIndexer
                )
            indexer_kwargs = dict(
                hidden_size=hidden_size,
                index_n_heads=N0.get_dsa_index_n_heads(config),
                index_head_dim=N0.get_dsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=N0.get_dsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=N0.add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
                config=config,
            )
            if N0.get_dsa_index_kpool(config) > 1:
                indexer_kwargs["skip_rope"] = skip_rope
            self.indexer = indexer_cls(**indexer_kwargs)

    self.kv_b_proj = N0.ColumnParallelLinear(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
        quant_config=quant_config,
        prefix=N0.add_prefix("kv_b_proj", prefix),
        parallel_group="attn_tp",
    )
    # O projection.
    self.o_proj = N0.RowParallelLinear(
        self.num_heads * self.v_head_dim,
        self.hidden_size,
        bias=False,
        quant_config=quant_config,
        reduce_results=reduce_results,
        prefix=N0.add_prefix("o_proj", prefix),
        parallel_group="attn_tp",
    )
    self.kv_a_layernorm = N0.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
    from hcu_sglang.layers.fused_rms_quant import (
        is_lightop_sglang_mla_qkv_a_rms_quant_available,
    )

    self.use_lightop_mla_qkv_a_rms_quant = (
        C._is_hcu
        and self.q_lora_rank is not None
        and U.get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")
        and is_lightop_sglang_mla_qkv_a_rms_quant_available()
        and getattr(
            self.q_b_proj.quant_method, "supports_prequantized_input", False
        )
    )

    if not skip_rope and qk_rope_head_dim > 0:
        is_neox_style = not getattr(config, "rope_interleave", True)
        self.rotary_emb = N0.get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device=N0.get_device().device,
        )

        if use_native_glm:
            from hcu_sglang.layers.rotary import configure_glm_rope_cache

            configure_glm_rope_cache(self.rotary_emb)

        if rope_scaling and rope_scaling.get("apply_yarn_scaling", True):
            self.scaling = N0.compute_mla_mscale_scaling(rope_scaling, self.scaling)
    else:
        self.rotary_emb = None
    self.use_deepseek_yarn_rope = rope_scaling is not None

    self.attn_mqa = N0.RadixAttention(
        self.num_local_heads,
        self.kv_lora_rank + self.qk_rope_head_dim,
        self.scaling,
        num_kv_heads=1,
        layer_id=layer_id,
        v_head_dim=self.kv_lora_rank,
        quant_config=quant_config,
        prefix=N0.add_prefix("attn_mqa", prefix),
    )
    # Reused DSA TopK indices were already sorted by the producing layer.
    self.attn_mqa.reuse_topk_indices = bool(self.skip_topk and not self.is_nextn)
    # use num_local_heads * dcp_world_size because q_nope, q_rope is all gathered from dcp ranks
    if self.dcp_size > 1:
        self.attn_mqa_for_dcp_decode = N0.RadixAttention(
            self.num_local_heads * self.dcp_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=N0.add_prefix("attn_mqa", prefix),
        )

    self.attn_mha = N0.RadixAttention(
        self.num_local_heads,
        self.qk_nope_head_dim + self.qk_rope_head_dim,
        self.scaling,
        num_kv_heads=self.num_local_heads,
        layer_id=layer_id,
        v_head_dim=self.v_head_dim,
        quant_config=quant_config,
        prefix=N0.add_prefix("attn_mha", prefix),
    )

    self.alt_stream = alt_stream
    self.attn_mha.kv_b_proj = None

    self.w_kc = None
    self.w_vc = None
    self.w_scale = 1.0

    # Full-head Q/absorb weights for --dcp-replicate-q-proj, gathered once
    # pre-CUDA-graph-capture by the model runner; None unless replicate is on.
    self.w_kc_qrep = None
    self.q_b_proj_qrep_weight = None

    self.w_scale_k = None
    self.w_scale_v = None
    self.use_deep_gemm_bmm = False

    self.current_attention_backend = (
        None  # Attention backend used by current forward batch
    )

    self.has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
    self.is_packed_weight = (
        self.has_fused_proj
        and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
        and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
        in {"awq", "awq_marlin", "moe_wna16"}
    )
    self._use_min_latency_fused_a_gemm: bool | None = None
    self.fused_a_gemm_backend = "auto"

    self.has_q_b_proj = hasattr(self, "q_b_proj")
    q_b_proj_verified_shapes = {(2048, 2048), (4096, 2048)}
    q_b_weight = (
        getattr(self.q_b_proj, "weight", None) if self.has_q_b_proj else None
    )
    self._q_b_proj_verified_shape = q_b_weight is not None and (
        tuple(q_b_weight.shape) in q_b_proj_verified_shapes
    )
    self._use_min_latency_q_b_gemm: bool | None = None

    self.init_mha_forward()
    self.init_mla_forward()
    self.init_mla_fused_rope_rocm_forward()
    self.init_mla_fused_rope_cpu_forward()


def prepare_qkv_latent(
    self,
    hidden_states: N0.torch.Tensor,
    forward_batch: N0.ForwardBatch,
    input_quant_args: N0.Optional[N0.Tuple[N0.torch.Tensor, N0.torch.Tensor]] = None,
):
    assert self.q_lora_rank is not None
    if input_quant_args is not None:
        return self.fused_qkv_a_proj_with_mqa(
            hidden_states, input_quant_args=input_quant_args
        )[0]
    if hasattr(self, "q_a_proj"):
        return N0.torch.cat(
            (
                self.q_a_proj(hidden_states)[0],
                self.kv_a_proj_with_mqa(hidden_states)[0],
            ),
            dim=-1,
        )
    if self._use_min_latency_fused_a_gemm is None:
        self._use_min_latency_fused_a_gemm = (
            self.has_fused_proj
            and not N0.get_exec().deterministic.enable_deterministic_inference
            and not self.is_packed_weight
            and N0.fused_a_gemm_weight_eligible(self.fused_qkv_a_proj_with_mqa)
        )
    if self._use_min_latency_fused_a_gemm:
        return N0.linear_with_fused_a_gemm(
            self.fused_qkv_a_proj_with_mqa,
            hidden_states,
            backend=self.fused_a_gemm_backend,
        )
    return self.fused_qkv_a_proj_with_mqa(hidden_states)[0]


def forward_absorb_prepare(
    self: N0.DeepseekV2AttentionMLA,
    positions: N1.torch.Tensor,
    hidden_states: N1.torch.Tensor,
    forward_batch: N1.ForwardBatch,
    zero_allocator: N1.BumpAllocator,
    llama_4_scaling: N1.Optional[N1.torch.Tensor] = None,
    prev_topk_indices: N1.Optional[N1.torch.Tensor] = None,
):
    from sglang.srt.model_executor.runner import get_is_capture_mode

    # Q8KV8 q-prep/indexer overlap handshake (see the fork site below):
    # True between the alt-stream fork and its consumption in the born
    # block; also suppresses the duplicate split/rope on that path.
    self._q8kv8_qprep_overlap_pending = False
    attention_output_gate = (
        self.prepare_attention_output_gate(hidden_states)
        if hasattr(self, "prepare_attention_output_gate")
        else None
    )

    fuse_bmm_attention = (
        self.q_lora_rank is not None
        and self._can_fuse_bmm_into_attention(forward_batch)
    )
    # --dcp-replicate-q-proj: project full-head Q locally from pre-gathered
    # weights and skip the per-layer Q all-gather (bf16 decode absorb only).
    q_replicate_active = (
        self.is_dcp_q_replicate_phase(forward_batch)
        and not self.use_deep_gemm_bmm
        and self.w_kc_qrep is not None
        and self.q_b_proj_qrep_weight is not None
    )
    if q_replicate_active:
        # force standard absorb so the full-head w_kc bmm runs
        fuse_bmm_attention = False
    q_lora = None
    topk_indices = None
    q_nope = None
    q_pe = None
    k_pe = None
    fusion_plan: N1.Optional[N1.MlaBmmFusionPlan] = None
    if self.q_lora_rank is not None:
        q, latent_cache = (
            N1.get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        )
        k_nope = latent_cache[..., : self.kv_lora_rank]

        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = N1.torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q = self.q_a_layernorm(q)
            with N1.torch.cuda.stream(self.alt_stream):
                k_nope = self.kv_a_layernorm(k_nope)
            current_stream.wait_stream(self.alt_stream)
        else:
            q = self.q_a_layernorm(q)
            k_nope = self.kv_a_layernorm(k_nope)

        # q_lora needed by indexer
        if self.use_dsa:
            if q_lora is None:
                q_lora = q

        # overlap q_b_proj and indexer during decode
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and forward_batch.forward_mode.is_decode_or_idle()
            and q_lora is not None
            and not q_replicate_active
        ):
            current_stream = N1.torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with N1.torch.cuda.stream(self.alt_stream):
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj_forward(q)
            if self.should_run_indexer(prev_topk_indices):
                topk_indices = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                )
            else:
                # skip_topk reuses prev layer's indices; mirror into this
                # layer's slot so the captured buffer matches what's used.
                topk_indices = N1.maybe_capture_indexer_topk(
                    self.layer_id, prev_topk_indices
                )
            current_stream.wait_stream(self.alt_stream)
        else:
            k_nope = k_nope.unsqueeze(1)
            if q_replicate_active:
                # full-head Q from the gathered weight (skips Q all-gather)
                q = N1.torch.nn.functional.linear(q, self.q_b_proj_qrep_weight).view(
                    -1,
                    self.num_local_heads * self.dcp_size,
                    self.qk_head_dim,
                )
            else:
                q = self.q_b_proj_forward(q)

            # Hoist these above the DSA indexer split op so the indexer
            # and the composite bmm+attention split op are adjacent in FX.
            if fuse_bmm_attention:
                q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)
                fusion_plan = self._make_mla_bmm_fusion_plan(q, q_nope)

            # Q8KV8 q-prep/indexer overlap (opt-in): the born-fp8 q-prep
            # chain (split -> rope -> fused absorbed-bmm+cast, ~173us)
            # and the indexer chain both fork from the q_a_layernorm
            # output and never touch each other's tensors, so the q-prep
            # can run on alt_stream underneath the indexer.  The fork
            # must be enqueued BEFORE the indexer (a later wait_stream
            # would serialize behind it).  The born predicate itself
            # guarantees eager-only and the plain-rope branch (all fused
            # /skip-rope variants make it return None), so applying rope
            # here is exactly what the skipped block below would do.
            if (
                N1._ENABLE_DSA_Q8KV8_QPREP_OVERLAP
                and N1._ENABLE_DSA_Q8KV8_BORN_FP8_Q
                and fusion_plan is None
                and self.alt_stream is not None
                and q_lora is not None
                and self.rotary_emb is not None
            ):
                _born_backend_early = self._q8kv8_born_fp8_q_backend(
                    forward_batch, llama_4_scaling
                )
                if _born_backend_early is not None:
                    q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)
                    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
                    _q_fp8 = _born_backend_early.q8kv8_acquire_born_q_buffer(
                        q_nope.shape[0],
                        self.num_local_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                        q_nope.device,
                    )
                    self.alt_stream.wait_stream(N1.torch.cuda.current_stream())
                    with N1.torch.cuda.stream(self.alt_stream):
                        N1.absorbed_bmm_concat_cast_q_fp8(
                            _q_fp8,
                            q_nope,
                            self.w_kc,
                            q_pe,
                            self.num_local_heads,
                        )
                    self._q8kv8_qprep_overlap_pending = True

            if q_lora is not None:
                if self.should_run_indexer(prev_topk_indices):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = N1.maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
    else:
        if q_replicate_active:
            q = N1.torch.nn.functional.linear(
                hidden_states, self.q_b_proj_qrep_weight
            ).view(
                -1,
                self.num_local_heads * self.dcp_size,
                self.qk_head_dim,
            )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        k_nope = latent_cache[..., : self.kv_lora_rank]
        k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

    if q_nope is None:
        q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)

    _kvb_q = None
    born_q_backend = None
    if (
        N1._ENABLE_DSA_Q8KV8_BORN_FP8_Q
        and fusion_plan is None
        and q_nope.dtype == N1.torch.bfloat16
    ):
        born_q_backend = self._q8kv8_born_fp8_q_backend(
            forward_batch, llama_4_scaling
        )
    if q_replicate_active:
        # full-head absorb with the pre-gathered w_kc (q_nope already full-head)
        q_nope_out = (
            N1.torch.bmm(q_nope.transpose(0, 1), self.w_kc_qrep)
            .transpose(0, 1)
            .contiguous()
        )
    elif getattr(self, "_kimi_split_gguf_kv_b", False):
        from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

        k_type = int(self.k_b_qweight_type.weight_type)
        q_nope_out = N1.torch.stack(
            [
                fused_mul_mat_gguf(q_nope[:, head], self.k_b_qweight[head], k_type)
                for head in range(self.num_local_heads)
            ],
            dim=1,
        )
    elif fusion_plan is not None:
        # The composite split op fills q_nope_out_buf and attention reads
        # this transposed alias directly.
        q_nope_out = fusion_plan.q_nope_out_view
    elif born_q_backend is not None:
        # Born-fp8 q: skip the bf16 absorbed bmm entirely; the fused
        # bmm+concat+cast kernel (launched after rope below) writes the
        # fp8 q directly into the q8kv8 backend buffer.
        q_nope_out = None
    else:
        if N1._SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b q-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_q_prepare,
            )

            _kvb_q = kv_b_lora_q_prepare(self, q_nope)

        if self.use_deep_gemm_bmm:
            (
                q_nope_val,
                q_nope_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = N1.per_token_group_quant_mla_deep_gemm_masked_fp8(
                q_nope.transpose(0, 1)
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            N1.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif self.w_kc.dtype == N1.torch.float8_e4m3fn:
            if N1._is_cpu:
                q_nope_out = N1.torch.bmm(
                    q_nope.to(N1.torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(N1.torch.bfloat16) * self.w_scale,
                )
            else:
                # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                q_nope_val, q_nope_scale = N1.per_tensor_quant_mla_fp8(
                    q_nope.transpose(0, 1),
                    (
                        N1.torch.zeros((1,), dtype=N1.torch.float32, device=q_nope.device)
                        if N1._is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                q_nope_out = N1.bmm_fp8(
                    q_nope_val,
                    self.w_kc,
                    q_nope_scale,
                    self.w_scale,
                    N1.torch.bfloat16,
                )
        else:
            q_nope_out = N1.torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)
        if N1._SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_q_apply,
            )

            q_nope_out = kv_b_lora_q_apply(self, q_nope, q_nope_out, _kvb_q)
        elif N1.is_kv_b_lora_active(self):
            q_nope_out = N1.apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

    fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)
    if (
        self.rotary_emb is not None
        and not fuse_rope_for_trtllm_mla
        # Already applied at the q-prep/indexer overlap fork.
        and not self._q8kv8_qprep_overlap_pending
    ):
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

    if born_q_backend is not None:
        # Born-fp8 q (SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q): one fused
        # kernel replaces bmm -> bf16 q_nope_out ->
        # concat_and_cast_q_fp8_pad.  q_nope is the pre-absorb bf16 view
        # (rope only touched the disjoint q_pe columns) and q_pe carries
        # the post-rope values.  The stash is consumed by
        # _forward_flashmla_sparse_q8kv8; q_nope_out becomes a
        # NaN-poisoned shape-only sentinel.
        num_tokens = q_nope.shape[0]
        if self._q8kv8_qprep_overlap_pending:
            # q_fp8 was produced on alt_stream at the fork above; join so
            # everything downstream (incl. the next layer's fork, which
            # reuses the single born-q slot) orders after it.
            N1.torch.cuda.current_stream().wait_stream(self.alt_stream)
            self._q8kv8_qprep_overlap_pending = False
        else:
            q_fp8 = born_q_backend.q8kv8_acquire_born_q_buffer(
                num_tokens,
                self.num_local_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
                q_nope.device,
            )
            N1.absorbed_bmm_concat_cast_q_fp8(
                q_fp8, q_nope, self.w_kc, q_pe, self.num_local_heads
            )
        born_q_backend.q8kv8_stash_born_q(
            N1.padded_extend_real_tokens(q_nope, forward_batch) or num_tokens,
            self.attn_mqa.layer_id,
        )
        q_nope_out = born_q_backend.q8kv8_born_q_sentinel(
            num_tokens, self.num_local_heads, self.kv_lora_rank, q_nope.device
        )

    k_nope, k_pe = N1.prepare_kv_for_attention(
        self,
        forward_batch,
        k_nope,
        k_pe,
        defer_materialization=fuse_rope_for_trtllm_mla,
    )

    # all_gather q_pe, q_nope_out,take tp8 as an example， q_pe [B, H, ROPE_DIM], q_nope_out [B, H, NOPE_DIM] gathered to [B, H * dcp_world_size, ROPE_DIM] [B, H * dcp_world_size, NOPE_DIM] for decode batch, and all gather k_pe, k_nope for extend batch.
    if self.dcp_size > 1:
        if self.is_dcp_mla_decode_phase(forward_batch):
            if not q_replicate_active:
                q_nope_out, q_pe = N1.all_gather_q_for_mla_decode(
                    q_nope_out=q_nope_out,
                    q_pe=q_pe,
                )
        elif forward_batch.forward_mode.is_extend():
            # for extend, gather kv
            N1.all_gather_kv_cache_for_mla_extend(
                N1.get_token_to_kv_pool(),
                self.attn_mqa,
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.attn_dcp_metadata.dcp_local_prefix_kv_indices,
                forward_batch.attn_dcp_metadata.dcp_extend_prefix_lens_sum,
                forward_batch.attn_dcp_metadata.dcp_kv_buffer,
                self.kv_lora_rank,
                k_nope,
                k_pe,
            )
        else:
            N1.logger.warning(
                f"not supported forward_mode {forward_batch.forward_mode}"
            )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        fusion_plan,
        # Bailing's DsV3MLA appends its own gate to inner_state, so this
        # slot is emitted only for models owning the gate hook.
        *(
            (attention_output_gate,)
            if hasattr(self, "prepare_attention_output_gate")
            else ()
        ),
    )


def forward_absorb_core(
    self: N0.DeepseekV2AttentionMLA,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    forward_batch,
    zero_allocator,
    positions,
    topk_indices,
    llama_4_scaling,
    fusion_plan: N1.Optional[N1.MlaBmmFusionPlan] = None,
    attention_output_gate: N1.Optional[N1.torch.Tensor] = None,
):
    save_kv_cache = True

    if self.current_attention_backend in N1.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
        extra_args = {}
        if getattr(self, "learnable_sink_param", None) is not None:
            extra_args["attn_sink"] = self.learnable_sink_param
        if self._fuse_rope_for_trtllm_mla(forward_batch):
            extra_args.update(
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                is_neox=self.rotary_emb.is_neox_style,
                llama_4_scaling=llama_4_scaling,
            )
        if fusion_plan is not None:
            self._eager_bmm_attention(
                fusion_plan.q_nope_t,
                self.w_kc,
                fusion_plan.q_nope_out_buf,
                q_nope_out,
                k_nope,
                fusion_plan.attn_output_buf,
                save_kv_cache,
                q_pe,
                k_pe,
                cos_sin_cache=extra_args.get("cos_sin_cache"),
                is_neox=extra_args.get("is_neox"),
                llama_4_scaling=extra_args.get("llama_4_scaling"),
                topk_indices=topk_indices,
            )
            attn_output = fusion_plan.attn_output_buf
        elif self.is_dcp_mla_decode_phase(forward_batch):
            # set return_lse=True to correct attn_output
            attn_output, lse = self.attn_mqa_for_dcp_decode(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
                **(
                    dict(topk_indices=topk_indices)
                    if topk_indices is not None
                    else {}
                ),
            )
        else:
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
                **(
                    dict(topk_indices=topk_indices)
                    if topk_indices is not None
                    else {}
                ),
            )
    else:
        q = N1.torch.cat([q_nope_out, q_pe], dim=-1)
        k = N1.torch.cat([k_nope, k_pe], dim=-1)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_output = self.attn_mqa(
            q,
            k,
            k_nope,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
        )

    # correct attn_output with respect to lse from other ranks
    if self.is_dcp_mla_decode_phase(forward_batch):
        attn_output = attn_output.view(
            -1,
            self.num_local_heads * self.dcp_size,
            self.kv_lora_rank,
        )
        if N1.get_in_autotune_dummy_run():
            # The synthetic FlashInfer MoE autotune pass discards model
            # outputs. Avoid an unnecessary cross-node MNNVL exchange of
            # zero attention partials.
            attn_output = N1._select_local_dcp_heads_for_autotune(
                attn_output, self.num_local_heads
            )
        else:
            dcp_comm_backend = N1.get_parallel().dcp_comm_backend
            is_lse_base_on_e = N1.is_mla_dcp_lse_base_on_e(
                self.current_attention_backend
            )
            if dcp_comm_backend in ("a2a", "fi_a2a"):
                # A2A exchange of head partials + LSE, then local Triton combine.
                attn_output = N1.dcp_a2a_lse_reduce(
                    attn_output.contiguous(),
                    lse.contiguous(),
                    N1.get_parallel().dcp_group,
                    is_lse_base_on_e=is_lse_base_on_e,
                    comm_backend=dcp_comm_backend,
                )
            else:
                attn_output = N1.cp_lse_ag_out_rs_mla(
                    attn_output,
                    lse,
                    N1.get_parallel().dcp_group,
                    is_lse_base_on_e=is_lse_base_on_e,
                )
                attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

    _kvb_v = None
    if N1._SGLANG_EXPERIMENTAL_LORA_OPTI:
        # Fork the kv_b v-correction A-step onto the LoRA side stream to overlap the bmm.
        from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
            kv_b_lora_v_prepare,
        )

        _kvb_v = kv_b_lora_v_prepare(self, attn_output)

    if getattr(self, "_kimi_split_gguf_kv_b", False):
        from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

        v_type = int(self.v_b_qweight_type.weight_type)
        attn_bmm_output = N1.torch.stack(
            [
                fused_mul_mat_gguf(
                    attn_output[:, head], self.v_b_qweight[head], v_type
                )
                for head in range(self.num_local_heads)
            ],
            dim=1,
        ).flatten(1, 2)
    elif self.use_deep_gemm_bmm:
        (
            attn_output_val,
            attn_output_scale,
            masked_m,
            expected_m,
            aligned_m,
        ) = N1.per_token_group_quant_mla_deep_gemm_masked_fp8(
            attn_output.transpose(0, 1)
        )
        attn_bmm_output = attn_output.new_empty(
            (self.num_local_heads, aligned_m, self.v_head_dim)
        )
        N1.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (attn_output_val, attn_output_scale),
            (self.w_vc, self.w_scale_v),
            attn_bmm_output,
            masked_m,
            expected_m,
        )
        attn_bmm_output = (
            attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
        )
    elif self.w_vc.dtype == N1.torch.float8_e4m3fn:
        if N1._is_cpu:
            attn_bmm_output = N1.torch.bmm(
                attn_output.to(N1.torch.bfloat16).transpose(0, 1),
                self.w_vc.to(N1.torch.bfloat16) * self.w_scale,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            attn_output_val, attn_output_scale = N1.per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                (
                    N1.torch.zeros(
                        (1,), dtype=N1.torch.float32, device=attn_output.device
                    )
                    if N1._is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            attn_bmm_output = N1.bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                N1.torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
    elif N1._is_musa:
        attn_bmm_output = N1.torch.bmm(
            attn_output.to(N1.torch.bfloat16).transpose(0, 1), self.w_vc
        )
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
    else:
        attn_bmm_output = N1.torch.empty(
            (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        N1.torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=attn_bmm_output.view(
                -1, self.num_local_heads, self.v_head_dim
            ).transpose(0, 1),
        )
    if N1._SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
            kv_b_lora_v_apply,
        )

        attn_bmm_output = kv_b_lora_v_apply(
            self, attn_output, attn_bmm_output, _kvb_v
        )
    elif N1.is_kv_b_lora_active(self):
        attn_bmm_output = N1.apply_kv_b_lora_v_correction(
            self, attn_output, attn_bmm_output
        )
    if attention_output_gate is not None:
        attn_bmm_output = N1._apply_attention_output_gate(
            self, attn_bmm_output, attention_output_gate
        )
    output, _ = self.o_proj(attn_bmm_output)

    if self.next_skip_topk is None:
        return output

    # Return topk_indices for the next layer when enabling index cache
    if not self.next_skip_topk:
        return output, None
    else:
        return output, topk_indices


def _set_mla_kv_buffer(
    self: N0.DeepseekV2AttentionMLA,
    latent_cache: N2.torch.Tensor,
    kv_a: N2.torch.Tensor,
    k_pe: N2.torch.Tensor,
    forward_batch: N2.ForwardBatch,
):
    if N2._is_cuda:
        # Save latent cache
        N2.get_token_to_kv_pool().set_mla_kv_buffer(
            self.attn_mha,
            N2.KVWriteLoc.for_batch(forward_batch),
            kv_a.unsqueeze(1),
            k_pe,
        )
    elif N2._is_npu:
        # To reduce a time-costing split operation
        N2.get_token_to_kv_pool().set_kv_buffer(
            self.attn_mha,
            N2.KVWriteLoc.for_batch(forward_batch),
            kv_a.unsqueeze(1),
            k_pe,
        )
    else:
        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe.clone()

        # Save latent cache
        N2.get_token_to_kv_pool().set_kv_buffer(
            self.attn_mha, N2.KVWriteLoc.for_batch(forward_batch), latent_cache, None
        )


def _get_mla_kv_buffer(
    self: N0.DeepseekV2AttentionMLA,
    kv_indices: N2.torch.Tensor,
    dst_dtype: N2.torch.dtype,
    forward_batch: N2.ForwardBatch,
):
    if N2._is_cuda or N2._use_aiter_gfx95:
        kv_indices = N2.filter_dcp_local_kv_indices(kv_indices=kv_indices)
        kv_a, k_pe = N2.get_token_to_kv_pool().get_mla_kv_buffer(
            self.attn_mha, kv_indices, dst_dtype
        )
        kv_a = kv_a.squeeze(1)
    else:
        latent_cache_buf = N2.get_token_to_kv_pool().get_key_buffer(
            self.attn_mha.layer_id
        )
        latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)
        # HCU's dense FlashMLA cache may pad no-rope rows from 512 to 576.
        latent_cache = latent_cache[
            ..., : self.kv_lora_rank + self.qk_rope_head_dim
        ]

        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_a = kv_a.squeeze(1).contiguous()
    return kv_a, k_pe


def _get_mla_kv_buffer_from_fp8_for_dsa(
    self: N0.DeepseekV2AttentionMLA,
    forward_batch: N2.ForwardBatch,
):
    """
    Dequantize FP8 KV cache to BF16 for MLA attention (DSA-specific format).

    Returns: (kv_a, k_pe) both in BF16
    """
    backend = N2.resolve_attn_backend(forward_batch)
    kv_indices = backend.forward_metadata.page_table_1_flattened
    assert kv_indices is not None, (
        "page_table_1_flattened should have been generated for FP8 MHA path"
    )

    if N2._use_aiter_gfx95:
        # ROCm (gfx950) stores the FP8 MLA KV in the raw
        # (kv_lora_rank + qk_rope_head_dim) layout, not the scaled 656-byte
        # layout that dequantize_k_cache_paged expects (it asserts dim==656).
        # Dequantize the raw layout via the pool's HIP-aware path instead —
        # the same routine _get_mla_kv_buffer uses for the BF16 MHA path.
        # Without this, a chunked-prefill split (extend_prefix_lens != 0) that
        # reads cached prefix KV crashes with "576 != 656".
        kv_indices = N2.filter_dcp_local_kv_indices(kv_indices=kv_indices)
        # Read door: the pool never translates, so the production site does.
        # Only the FlashAttention/FlashInfer backends bind a translator; the
        # base class documents its None default as "no translate", and the DSA
        # backend the EAGLE draft model runs on keeps that default. With no
        # translator the ids never went VIRTUAL, so there is nothing to
        # collapse.
        # 1be97a0 dropped the backend-level kv_index_translator default
        # (runner_ids on the base class now); absent means "no translate".
        translator = getattr(N2.get_attn_backend(), "kv_index_translator", None)
        if translator is not None:
            kv_indices = translator.translate_dcp_read_ids(kv_indices)
        kv_a, k_pe = N2.get_token_to_kv_pool().get_mla_kv_buffer(
            self.attn_mha, kv_indices, N2.torch.bfloat16
        )
        kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

    kv_cache_fp8 = N2.get_token_to_kv_pool().get_key_buffer(self.attn_mha.layer_id)

    kv_latent_bf16 = N2.dequantize_k_cache_paged(kv_cache_fp8, kv_indices)

    kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
    k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]

    return kv_a, k_pe


def _concat_and_cast_mha_k(
    self: N0.DeepseekV2AttentionMLA,
    k_nope: N2.torch.Tensor,
    k_pe: N2.torch.Tensor | None,
    forward_batch: N2.ForwardBatch,
):
    if self.qk_rope_head_dim == 0:
        assert k_pe is None or k_pe.shape[-1] == 0
        k = k_nope.contiguous()
        if (
            N2._is_cuda
            and self.current_attention_backend == "fa3"
            and self.kv_cache_dtype != "auto"
        ):
            # fa3 requires k in the pool dtype when KV cache is fp8; the
            # concat branch below does the same cast for roped models.
            k = k.to(N2.get_token_to_kv_pool().dtype)
        return k

    # Temporary for DeepSeek V3/R1 only, but can generalize if needed
    k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
    if (
        (N2._is_cuda or N2._is_musa)
        and (self.num_local_heads == 128)
        and (self.qk_nope_head_dim == 128)
        and (self.qk_rope_head_dim == 64)
    ):
        k = k_nope.new_empty(*k_shape)
        N2.concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
    elif (
        N2._is_cuda
        and N2.next_power_of_2(self.num_local_heads) == self.num_local_heads
        and N2.next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim
        and N2.next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim
    ):
        # fa3 mha support fp8 inputs
        if (
            self.current_attention_backend == "fa3"
            and self.kv_cache_dtype != "auto"
        ):
            attn_dtype = N2.get_token_to_kv_pool().dtype
        else:
            attn_dtype = k_nope.dtype
        k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
        N2.concat_and_cast_mha_k_triton(k, k_nope, k_pe)
    else:
        k = k_nope.new_empty(*k_shape)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
    return k


def _chunked_prefix_attn_mha(
    self: N0.DeepseekV2AttentionMLA,
    q: N2.torch.Tensor,
    accum_output: N2.torch.Tensor,
    accum_lse: N2.torch.Tensor,
    forward_batch: N2.ForwardBatch,
) -> N2.torch.Tensor:
    # kv_b_proj needs BF16 input, but legacy q.dtype was BF16 by accident.
    from sglang.srt.layers.attention.merge_state import merge_state

    backend = N2.resolve_attn_backend(forward_batch)
    pack_fn = getattr(backend, "pack_prefix_chunk_kv", None)
    kv_a_dtype = N2.torch.bfloat16 if pack_fn is not None else q.dtype
    get_mla_kv_buffer = (
        self._get_mla_kv_buffer_rocm if N2._is_hip else self._get_mla_kv_buffer
    )

    # If the complete prefix fits the capacity, gather/project/pack once.
    # Larger prefixes retain bounded per-chunk materialization.
    packed_prefix_k = None
    packed_prefix_v = None
    pack_all_prefix_chunks = N2.use_packed_prefix_chunks(backend, forward_batch)
    if pack_all_prefix_chunks:
        kv_a_normed, k_pe = get_mla_kv_buffer(
            forward_batch.prefix_all_kv_indices,
            kv_a_dtype,
            forward_batch,
        )
        kv = self.kv_b_proj(kv_a_normed)[0]
        kv = kv.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v_dense = kv[..., self.qk_nope_head_dim :]
        packed_prefix_k, packed_prefix_v = pack_fn(k_nope, k_pe, v_dense)
        del kv_a_normed, k_pe, kv, k_nope, v_dense

    assert forward_batch.num_prefix_chunks is not None
    packed_offset = 0
    for i in range(forward_batch.num_prefix_chunks):
        forward_batch.set_prefix_chunk_idx(i)

        if pack_all_prefix_chunks:
            chunk_num_tokens = forward_batch.prefix_chunk_num_tokens[i]
            packed_end = packed_offset + chunk_num_tokens
            k = packed_prefix_k[packed_offset:packed_end]
            v = packed_prefix_v[packed_offset:packed_end]
            packed_offset = packed_end
        else:
            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            kv_a_normed, k_pe = get_mla_kv_buffer(
                kv_indices, kv_a_dtype, forward_batch
            )
            kv_a_normed, k_pe = N2.all_gather_kv_cache_for_mha_chunk_extend(
                kv_a_normed,
                k_pe,
                forward_batch.prefix_chunk_seq_lens_cpu[i],
                forward_batch.prefix_chunk_starts_cpu[i],
            )
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1,
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            v_dense = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            if pack_fn is not None:
                k, v = pack_fn(k_nope, k_pe, v_dense)
            else:
                v = v_dense
                k = N2.torch.empty(
                    (
                        k_nope.shape[0],
                        self.num_local_heads,
                        self.qk_nope_head_dim + self.qk_rope_head_dim,
                    ),
                    dtype=v.dtype,
                    device=v.device,
                )
                k[..., : self.qk_nope_head_dim] = k_nope
                k[..., self.qk_nope_head_dim :] = k_pe
            del kv_a_normed, k_pe, kv, k_nope, v_dense

        output, lse = self.attn_mha(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=False,
            # Prefix K/V is independent of the suffix query length. Under
            # FullCG this is the fixed captured chunk extent; per-request
            # active lengths remain encoded in the backend metadata.
            key_value_num_tokens=k.shape[0],
        )
        tmp_output = N2.torch.empty_like(accum_output)
        tmp_lse = N2.torch.empty_like(accum_lse)
        merge_state(
            output,
            lse,
            accum_output,
            accum_lse,
            tmp_output,
            tmp_lse,
        )
        accum_output, accum_lse = tmp_output, tmp_lse
        del k, v, output, lse, tmp_output, tmp_lse

    if pack_all_prefix_chunks:
        assert packed_offset == packed_prefix_k.shape[0]
        del packed_prefix_k, packed_prefix_v

    return accum_output


def forward_normal_chunked_kv_prepare(
    self: N0.DeepseekV2AttentionMLA,
    positions: N2.torch.Tensor,
    hidden_states: N2.torch.Tensor,
    forward_batch: N2.ForwardBatch,
    zero_allocator: N2.BumpAllocator,
):
    # In normal mha, the k and v tensors will become overly large when the prefix length is long.
    # To avoid this, we split the kv cache into chunks and process them one after another.
    # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
    # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
    # will be helpful for understanding the purpose of this function.

    # Preserve the ROCm fused RMS/quantized projection path for the current
    # chunk; the shared preparation path is primarily the CUDA version.
    if N2._is_hip:
        return self.forward_normal_rocm_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    # First do normal mha forward to get output for extended part.
    return self.forward_normal_prepare(
        positions, hidden_states, forward_batch, zero_allocator
    )


def forward_normal_chunked_kv_core(
    self: N0.DeepseekV2AttentionMLA,
    q: N2.torch.Tensor,
    k: N2.torch.Tensor,
    v: N2.torch.Tensor,
    forward_batch: N2.ForwardBatch,
    gate: N2.Optional[N2.torch.Tensor] = None,
) -> N2.torch.Tensor:
    has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
        forward_batch.extend_prefix_lens_cpu
    )
    backend = N2.resolve_attn_backend(forward_batch)
    prepare_qkv_fn = getattr(backend, "prepare_chunked_prefill_qkv", None)
    if has_extend_prefix and prepare_qkv_fn is not None:
        q, k, v = prepare_qkv_fn(q, k, v, forward_batch)

    fused_prefix = has_extend_prefix and N2.use_fused_prefix_extend(
        backend, forward_batch
    )

    # Only initialize the info once
    if has_extend_prefix and forward_batch.num_prefix_chunks is None:
        forward_batch.prepare_chunked_prefix_cache_info(
            q.device,
            pack_all_prefix_chunks=N2.use_packed_prefix_chunks(backend, forward_batch),
            single_chunk=fused_prefix,
            dense_metadata=N2.use_dense_prefix_kv(backend),
        )
        if hasattr(N2.get_attn_backend(), "init_mha_chunk_metadata"):
            N2.get_attn_backend().init_mha_chunk_metadata(forward_batch)

    if fused_prefix:
        attn_output = self._fused_prefix_extend_attn_mha(q, k, v, forward_batch)
        attn_output = attn_output.reshape(
            -1, self.num_local_heads * self.v_head_dim
        )
        output, _ = self.o_proj(attn_output)
        return output

    forward_batch.mha_return_lse = has_extend_prefix
    # Do mha for extended part without prefix
    forward_batch.set_attn_attend_prefix_cache(False)
    attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

    # Do mha attention with chunked prefix cache if there are any sequence with prefix
    if has_extend_prefix:
        attn_output, lse = attn_output
        forward_batch.set_attn_attend_prefix_cache(True)
        attn_output = self._chunked_prefix_attn_mha(
            q=q,
            accum_output=attn_output,
            accum_lse=lse,
            forward_batch=forward_batch,
        )

    attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
    if gate is not None:
        attn_output = self._apply_gated(attn_output, gate)
    output, _ = self.o_proj(attn_output)
    return output


def forward_normal_one_shot_prepare(
    self: N0.DeepseekV2AttentionMLA,
    positions: N2.torch.Tensor,
    hidden_states: N2.torch.Tensor,
    forward_batch: N2.ForwardBatch,
    zero_allocator: N2.BumpAllocator,
):
    forward_batch.mha_one_shot = True
    return self.forward_normal_prepare(
        positions, hidden_states, forward_batch, zero_allocator
    )


def forward_normal_one_shot_core(
    self: N0.DeepseekV2AttentionMLA,
    q: N2.torch.Tensor,
    k: N2.torch.Tensor,
    v: N2.torch.Tensor,
    forward_batch: N2.ForwardBatch,
    gate: N2.Optional[N2.torch.Tensor] = None,
) -> N2.torch.Tensor:
    has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
    # Only initialize the info once
    if has_extend_prefix and forward_batch.num_prefix_chunks is None:
        forward_batch.num_prefix_chunks = 0
        if hasattr(N2.get_attn_backend(), "init_mha_chunk_metadata"):
            N2.get_attn_backend().init_mha_chunk_metadata(forward_batch)
    forward_batch.mha_return_lse = False
    # Do mha for extended part without prefix
    forward_batch.set_attn_attend_prefix_cache(False)
    # K and V cover every token of the sequences, the cached prefix too.
    return self.forward_normal_core(
        q, k, v, forward_batch, gate, key_value_num_tokens=k.shape[0]
    )


def forward_absorb_fused_mla_rope_prepare(
    self: N0.DeepseekV2AttentionMLA,
    positions: N3.torch.Tensor,
    hidden_states: N3.torch.Tensor,
    forward_batch: N3.ForwardBatch,
    zero_allocator: N3.BumpAllocator,
):
    enable_rope_fusion = (
        N3.os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
    )
    # NOTE: hidden_states can be a tuple for some quantization paths.
    # For shape/device/dtype, use the first tensor; still pass the original
    # hidden_states through linear ops which may accept tuple inputs.
    hidden_states_tensor = (
        hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    )

    q_len = hidden_states_tensor.shape[0]
    q_input = hidden_states_tensor.new_empty(
        q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
    )
    if self.q_lora_rank is not None:
        q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
    else:
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    if N3._is_hip:
        # TODO(haishaw): add bmm_fp8 to ROCm
        q_nope_out = N3.torch.bmm(
            q_nope.to(N3.torch.bfloat16).transpose(0, 1),
            self.w_kc.to(N3.torch.bfloat16) * self.w_scale,
        )
    elif self.w_kc.dtype == N3.torch.float8_e4m3fn:
        q_nope_val, q_nope_scale = N3.per_tensor_quant_mla_fp8(
            q_nope.transpose(0, 1),
            zero_allocator.allocate(1),
            dtype=N3.torch.float8_e4m3fn,
        )
        q_nope_out = N3.bmm_fp8(
            q_nope_val, self.w_kc, q_nope_scale, self.w_scale, N3.torch.bfloat16
        )
    else:
        q_nope_out = N3.torch.bmm(q_nope.transpose(0, 1), self.w_kc)
    q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
    v_input = latent_cache[..., : self.kv_lora_rank]
    v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., : self.kv_lora_rank] = v_input

    if not enable_rope_fusion:
        k_pe = k_input[..., self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_input[..., self.kv_lora_rank :] = q_pe
        k_input[..., self.kv_lora_rank :] = k_pe
        k_pe_output = None
    else:
        k_pe_output = N3.torch.empty_like(k_input[..., self.kv_lora_rank :])

    q_input[..., self.kv_lora_rank :] = q_pe

    # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
    # Use Fused ROPE with use_rope=OFF.
    attn_output = N3.torch.empty(
        (q_len, self.num_local_heads, self.kv_lora_rank),
        dtype=q.dtype,
        device=q.device,
    )
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
        N3.get_attn_backend().forward_metadata
    )
    cos_sin_cache = self.rotary_emb.cos_sin_cache
    num_kv_split = N3.get_attn_backend().num_kv_splits
    sm_scale = self.attn_mqa.scaling
    if attn_logits is None:
        attn_logits = N3.torch.empty(
            (
                forward_batch.batch_size,
                self.num_local_heads,
                num_kv_split,
                self.kv_lora_rank + 1,
            ),
            dtype=N3.torch.float32,
            device=q.device,
        )

    # save current latent cache.
    N3.get_token_to_kv_pool().set_kv_buffer(
        self.attn_mqa, N3.KVWriteLoc.for_batch(forward_batch), k_input, None
    )
    key_cache_buf = N3.get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id)
    val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

    return (
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    )


def forward_absorb_fused_mla_rope_core(
    self: N0.DeepseekV2AttentionMLA,
    q_input,
    key_cache_buf,
    val_cache_buf,
    attn_output,
    kv_indptr,
    kv_indices,
    k_pe_output,
    cos_sin_cache,
    positions,
    attn_logits,
    num_kv_split,
    sm_scale,
    enable_rope_fusion,
    k_input,
    forward_batch,
    zero_allocator,
    gate=None,
):
    N3.decode_attention_fwd_grouped_rope(
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        self.kv_lora_rank,
        self.rotary_emb.rotary_dim,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        logit_cap=self.attn_mqa.logit_cap,
        use_rope=enable_rope_fusion,
        is_neox_style=self.rotary_emb.is_neox_style,
    )

    if enable_rope_fusion:
        k_input[..., self.kv_lora_rank :] = k_pe_output
        N3.get_token_to_kv_pool().set_kv_buffer(
            self.attn_mqa, N3.KVWriteLoc.for_batch(forward_batch), k_input, None
        )

    attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

    if N3._is_hip:
        # TODO(haishaw): add bmm_fp8 to ROCm
        attn_bmm_output = N3.torch.bmm(
            attn_output.to(N3.torch.bfloat16).transpose(0, 1),
            self.w_vc.to(N3.torch.bfloat16) * self.w_scale,
        )
    elif self.w_vc.dtype == N3.torch.float8_e4m3fn:
        attn_output_val, attn_output_scale = N3.per_tensor_quant_mla_fp8(
            attn_output.transpose(0, 1),
            zero_allocator.allocate(1),
            dtype=N3.torch.float8_e4m3fn,
        )
        attn_bmm_output = N3.bmm_fp8(
            attn_output_val,
            self.w_vc,
            attn_output_scale,
            self.w_scale,
            N3.torch.bfloat16,
        )
    else:
        attn_bmm_output = N3.torch.bmm(attn_output.transpose(0, 1), self.w_vc)
    attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
    if gate is not None:
        attn_output = self._apply_gated(attn_output, gate)
    output, _ = self.o_proj(attn_output)

    return output


def forward_absorb_fused_mla_rope_cpu_prepare(
    self: N0.DeepseekV2AttentionMLA,
    positions: N4.torch.Tensor,
    hidden_states: N4.torch.Tensor,
    forward_batch: N4.ForwardBatch,
    zero_allocator: N4.BumpAllocator,
):
    assert self.q_lora_rank is not None and N4.use_intel_amx_backend(self), (
        "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"
    )

    q_input, k_input, v_input = (
        N4.torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
            hidden_states,
            self.fused_qkv_a_proj_with_mqa.weight,
            self.q_b_proj.weight,
            self.w_kc,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.kv_a_layernorm.variance_epsilon,
            self.qkv_proj_with_rope_is_int8,
            self.qkv_proj_with_rope_is_fp8,
            (
                self.fused_qkv_a_proj_with_mqa.weight_scale
                if self.qkv_proj_with_rope_is_int8
                else (
                    self.fused_qkv_a_proj_with_mqa.weight_scale_inv
                    if self.qkv_proj_with_rope_is_fp8
                    else None
                )
            ),
            (
                self.q_b_proj.weight_scale
                if self.qkv_proj_with_rope_is_int8
                else (
                    self.q_b_proj.weight_scale_inv
                    if self.qkv_proj_with_rope_is_fp8
                    else None
                )
            ),
            self.w_scale if self.qkv_proj_with_rope_is_fp8 else None,
            True,  # is_vnni
            self.weight_block_size,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )
    )
    return (q_input, k_input, v_input, forward_batch, zero_allocator)


def forward_absorb_fused_mla_rope_cpu_core(
    self: N0.DeepseekV2AttentionMLA,
    q_input,
    k_input,
    v_input,
    forward_batch,
    zero_allocator,
    gate=None,
):
    assert self.q_lora_rank is not None and N4.use_intel_amx_backend(self), (
        "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"
    )

    attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
    attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

    # [Note] Align shapes of bmm inputs.
    # Shapes of inputs:
    #   q_nope: [M, B, K]
    #   original self.w_kc: [B, K, N]
    #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

    # Shapes of inputs to sgl_kernel.cpu.bmm:
    #   out: [B, M, N]
    #   mat1: [B, M, K]
    #   mat2: [B, N, K]
    B = self.w_vc.size(0)
    N = self.w_vc.size(1)
    M = attn_output.size(0)
    output = N4.torch.empty([M, int(B * N)], dtype=attn_output.dtype)
    attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
    N4.torch.ops.sgl_kernel.bmm_cpu(
        attn_bmm_output,
        attn_output.transpose(0, 1),
        self.w_vc,
        True,  # is_vnni
        self.w_scale if self.qkv_proj_with_rope_is_fp8 else None,  # scale
    )
    attn_output = output
    if gate is not None:
        attn_output = self._apply_gated(attn_output, gate)
    output, _ = self.o_proj(attn_output)

    return output
