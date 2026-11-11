# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
"""Owned HCU MQA classes with their original class closures and current ABI."""
from __future__ import annotations
import functools
import sglang.srt.models.deepseek_v4 as upstream
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_helpers import (
    is_cp_cache_layer_split_deepseek_v4_pool, maybe_prefetch_cp_kv_extra,
    maybe_prefetch_cp_kv_swa, maybe_wait_cp_kv_swa_prefetch,
)
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from hcu_sglang.models.deepseek.current_v4.replay_scope import bcg_tail_kv_store, in_replay_tail_graph

# Reuse the pinned official operators/value types rather than importing a fork.
# Framework class replacements propagate imported aliases after apply_hooks.
globals().update({name:value for name,value in vars(upstream).items()
                  if not name.startswith('__')})
_is_hcu = is_hcu()
_use_aiter = _use_aiter and not _is_hcu
if _is_hcu:
    _is_cuda = False
_wo_a_fp8_mxscale_prequant = None
if _wo_a_fp8_mxscale is not None:
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_wo_a_fp8 as wo_a_ops
    _wo_a_fp8_mxscale_prequant = getattr(wo_a_ops, 'apply_wo_a_fp8_mxscale_prequant', None)
    if _wo_a_fp8_mxscale_prequant is None:
        raise ImportError('Selected FP8 wo_a backend lacks the current prequant entry point')

def _fp8_wo_a_einsum(
    o_fp8: torch.Tensor,
    o_s: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    recipe: Tuple[int, int, int],
) -> torch.Tensor:
    import deep_gemm

    groups, rank = weight.shape[:2]
    output = torch.empty(
        o_fp8.shape[0], groups, rank, device=o_fp8.device, dtype=torch.bfloat16
    )
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (o_fp8, o_s), (weight, weight_scale), output, recipe=recipe
    )
    return output


@register_custom_op(mutates_args=["output"])
@register_split_op()
def hcu_deepseek_v4_attention_with_output(
    query: torch.Tensor,
    key_value: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
    compress_ratio: int,
    attn_sink: torch.Tensor,
    save_kv_cache: bool,
) -> None:
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_num_tokens = forward_batch.global_num_token_non_padded_cpu

    if real_num_tokens == 0:
        output.zero_()
        return

    query = query[:real_num_tokens]
    key_value = key_value[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    attn_backend = get_attn_backend()
    try:
        ret = attn_backend.forward(
            q=query,
            k=key_value,
            v=key_value,
            layer=attention_layer,
            forward_batch=forward_batch,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
            save_kv_cache=save_kv_cache,
        )
    finally:
        forward_batch.out_cache_loc = original_out_cache_loc

    assert output[:real_num_tokens].numel() == ret.numel(), (
        f"Output tensor element mismatch: {output[:real_num_tokens].numel()} != {ret.numel()}"
    )

    output[:real_num_tokens].view(ret.shape).copy_(ret)
    output[real_num_tokens:].zero_()
    return


bcg_deepseek_v4_attention_with_output = eager_on_graph(True)(hcu_deepseek_v4_attention_with_output)

def deepseek_v4_mega_attention_with_output(layer, query, output, attn_sink) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    real_num_tokens = forward_batch.global_num_token_non_padded_cpu
    if real_num_tokens:
        output[:real_num_tokens].copy_(
            layer._mega_core(
                query[:real_num_tokens], attn_sink, forward_batch, get_attn_backend()
            )
        )
    output[real_num_tokens:].zero_()


bcg_deepseek_v4_mega_attention_with_output = eager_on_graph(True)(deepseek_v4_mega_attention_with_output)

def deepseek_v4_low_ratio_sources(layer, x, q_lora, positions) -> None:
    # The compressor and prefill indexer sync with the host, like the attention.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    real_num_tokens = forward_batch.global_num_token_non_padded_cpu
    if real_num_tokens == 0:
        return
    get_attn_backend().forward_low_ratio_sources(
        layer=layer,
        x=x[:real_num_tokens],
        q_lora=(
            _hip.live_rows(q_lora, real_num_tokens)
            if _is_hip
            else q_lora[:real_num_tokens]
        ),
        positions=positions[:real_num_tokens],
        forward_batch=forward_batch,
    )


bcg_deepseek_v4_low_ratio_sources = eager_on_graph(True)(deepseek_v4_low_ratio_sources)

class MqaAttentionBase(nn.Module):
    # Class-level default for subclasses that read it without running __init__.
    wo_a_fp8: bool = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        *,
        compress_ratio: Optional[int] = None,
        fuse_wqa_wkv: Optional[bool] = None,
        wo_a_fp8: Optional[bool] = None,
        wo_a_keeps_quant_config: Optional[bool] = None,
        wo_b_reduce_results: Optional[bool] = None,
        rope_original_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.is_dsv41 = getattr(config, "model_type", None) == "deepseek_v41"
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size
        if _is_hcu:
            attn_tp_rank = get_parallel().attn_tp_rank
            attn_tp_size = get_parallel().attn_tp_size
            if self.dsa_enable_prefill_cp:
                self.cp_size = get_parallel().attn_cp_size
                attn_tp_rank, attn_tp_size = 0, 1
        self.attn_tp_rank: int = attn_tp_rank
        self.attn_tp_size: int = attn_tp_size

        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.hidden_size = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // self.attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // self.attn_tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.q_head_norm = config.q_head_norm

        self.compress_ratio: int = (
            compress_ratio
            if compress_ratio is not None
            else config.compress_ratios[layer_id]
        )
        assert self.compress_ratio in (
            0,
            1,
            2,
            4,
            128,
        ), (
            f"compress_ratio: expected one of (0, 1, 2, 4, 128), got {self.compress_ratio}"
        )

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        fuse: bool = (
            envs.SGLANG_OPT_FUSE_WQA_WKV.get() if fuse_wqa_wkv is None else fuse_wqa_wkv
        )
        fp8: bool = (
            wo_a_fp8_gemm_enabled(quant_config) if wo_a_fp8 is None else wo_a_fp8
        )
        reduce_results: bool = (
            (self.attn_tp_size == get_parallel().tp_size and self.attn_tp_size > 1)
            if wo_b_reduce_results is None
            else wo_b_reduce_results
        )
        # NPU arch35 runs wo_a as a batched MXFP8 GEMM instead of deep_gemm's FP8 one,
        # but it needs the same quantized weights.
        self.use_npu_arch35_mxfp8_wo_a = use_npu_arch35_mxfp8_wo_a(quant_config)
        quantize_wo_a = fp8 or self.use_npu_arch35_mxfp8_wo_a
        if wo_a_keeps_quant_config is None:
            keep_source_quant = (
                quant_config is not None and quant_config.get_name() == "expert_pack"
            )
            wo_a_quant_config: Optional[QuantizationConfig] = (
                quant_config if quantize_wo_a or keep_source_quant else None
            )
        elif wo_a_keeps_quant_config:
            wo_a_quant_config = quant_config
        else:
            wo_a_quant_config = None

        self.fuse_wqa_wkv = fuse
        self.wo_a_fp8 = fp8

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self._attn_sink_local: Optional[torch.Tensor] = None
        if fuse:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            parallel_group="attn_tp",
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=wo_a_quant_config,
            prefix=add_prefix("wo_a", prefix),
            parallel_group="attn_tp",
            **({} if quantize_wo_a else {"params_dtype": torch.bfloat16}),
        )
        if quantize_wo_a:
            assert hasattr(self.wo_a, "weight_scale_inv"), (
                "FP8 quant_config must create weight_scale_inv"
            )
        if self.use_npu_arch35_mxfp8_wo_a:
            # Read by the NPU arch35 MXFP8 weight processor to batch the
            # weight/scale per attention group for npu_transpose_quant_batchmatmul.
            self.wo_a._dsv4_npu_arch35_mxfp8_wo_a = True
            self.wo_a._dsv4_num_groups = self.n_local_groups
            self.wo_a._dsv4_o_lora_rank = self.o_lora_rank
        elif fp8:
            self.wo_a.weight_scale_inv.format_ue8m0 = _FP8_WO_A_UE8M0
            # wo_a is quantized but never *applied* through its quant method:
            # the absorb GEMM in forward() reads .weight / .weight_scale_inv and
            # runs its own batched kernel (DeepGEMM fp8_einsum on CUDA, aiter
            # mxscale BMM on gfx950), both of which want the plain row-major
            # [G, R, D] weight. Opt out of any backend-private weight layout the
            # linear method would otherwise install for its own GEMM -- on ROCm
            # that is aiter's B-preshuffle, which silently permutes the weight
            # in place (same shape, dtype and strides) and makes this GEMM
            # return noise.
            self.wo_a.keep_plain_weight_layout = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("wo_b", prefix),
            parallel_group="attn_tp",
        )

        from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis

        rope_theta, rope_scaling = get_rope_config(config)
        self.rope_scaling = dict(rope_scaling) if rope_scaling else None
        scaling = self.rope_scaling or {}

        # RoPE is selected at layer granularity in the reference model. Pure
        # SWA layers use the main unscaled RoPE, while C4/C128 layers use the
        # compressed YaRN RoPE for Q, their SWA branch, and compressed KV.
        self.rope_base = (
            config.compress_rope_theta if self.compress_ratio else rope_theta
        )
        original_seq_len: int = (
            rope_original_seq_len
            if rope_original_seq_len is not None
            else (
                scaling["original_max_position_embeddings"]
                if self.compress_ratio
                else 0
            )
        )
        self.rope_kwargs = dict(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=self.rope_base,
            factor=scaling.get("factor", 1.0),
            beta_fast=scaling.get("beta_fast", 32),
            beta_slow=scaling.get("beta_slow", 1),
        )
        freqs_cis = precompute_freqs_cis(**self.rope_kwargs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

    @functools.cached_property
    def use_flashinfer_mxfp8_wo_b(self) -> bool:
        """Whether wo_b consumes FlashInfer-swizzled MXFP8, so wo_a can fuse the
        quantization into its epilogue. Not known until wo_b's weights load."""
        quant_method = getattr(self.wo_b, "quant_method", None)
        return getattr(
            quant_method, "mxfp8_dense_backend", None
        ) == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL and (
            getattr(quant_method, "use_mxfp8", False)
            or getattr(self.wo_b, "block_fp8_mxfp8_ready", False)
        )

    def _kernel_num_heads(self, num_tokens: int) -> int:
        if self.attn_tp_size == 1:
            return self.n_local_heads

        if get_platform().is_sm120:
            # Prefill already accepts the native per-rank query width.
            if num_tokens > SM120_DECODE_MAX_TOKENS:
                return self.n_local_heads

            if envs.SGLANG_SM120_FLASHMLA_BACKEND.get() == "flashinfer":
                from sglang.kernels.ops.attention.flash_mla_sm120 import (
                    flashinfer_dsv4_decode_supports_num_heads,
                )

                if flashinfer_dsv4_decode_supports_num_heads(
                    self.n_local_heads, num_tokens
                ):
                    return self.n_local_heads

        # Other FlashMLA implementations retain their existing padded shape.
        return 64 if self.n_local_heads <= 64 else self.n_heads

    def _wo_a_quant_freqs(
        self, forward_batch: ForwardBatch, q_rope: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Rope table when the fp8 unified_kv decode (aiter FlyDSL) should also
        do the inverse RoPE and the wo_a mxfp8 quant; else None."""
        mode = forward_batch.forward_mode
        if q_rope is None or not (mode.is_decode() or mode.is_target_verify()):
            return None
        if not hasattr(self, "_wo_a_freqs"):
            from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

            fused = (
                _wo_a_fp8_mxscale_prequant is not None
                and self.wo_a_fp8
                and runtime.flydsl_decode_supported(q_rope.shape[1])
            )
            self._wo_a_freqs = (
                torch.view_as_real(self.freqs_cis).flatten(1) if fused else None
            )
        return self._wo_a_freqs

    def _local_attn_sink(self, kernel_num_heads: Optional[int] = None) -> torch.Tensor:
        if self.attn_tp_size == 1:
            return self.attn_sink

        rank = self.attn_tp_rank
        num_heads = self.n_local_heads
        padded_num_heads = 64 if num_heads <= 64 else self.n_heads
        if kernel_num_heads is None:
            # Preserve the legacy contract for subclasses such as DSpark that
            # always pad their attention query independently of this helper.
            kernel_num_heads = padded_num_heads
        assert kernel_num_heads >= num_heads

        # Keep one fallback-width allocation and return a view matching Q.
        # Prefill and decode can alternate, and CUDA graphs can retain the
        # view, so replacing this tensor when the path changes would
        # both reallocate every transition and risk invalidating a captured
        # pointer.
        sink_num_heads = max(kernel_num_heads, padded_num_heads)
        if self._attn_sink_local is None:
            sink = self.attn_sink.new_zeros(sink_num_heads)
            sink[:num_heads] = self.attn_sink[rank * num_heads : (rank + 1) * num_heads]
            self._attn_sink_local = sink
        return self._attn_sink_local[:kernel_num_heads]

    def _q_lora_quant_e8m0(self, with_indexer: bool) -> bool:
        """Whether the fused q_norm quant feeds wq_b (and the indexer's wq_b, which
        reuses it) with ue8m0 scales: only when each consumer runs the e8m0 mxscale
        GEMM. Resolved on first use, after the weights are processed."""
        key = "_q_lora_e8m0_idx" if with_indexer else "_q_lora_e8m0"
        cached = self.__dict__.get(key)
        if cached is None:
            from sglang.srt.layers.quantization.fp8_utils import dense_mxscale_ready

            indexer = getattr(self, "indexer", None) if with_indexer else None
            cached = dense_mxscale_ready(
                self.wq_b, getattr(indexer, "wq_b", None) if indexer else None
            )
            self.__dict__[key] = cached
        return cached

    @contextmanager
    def maybe_use_decode_attn_tp(self, forward_batch: ForwardBatch):
        ctx = get_cp_decode_attn_tp_ctx()
        attn = self.attn_mqa if isinstance(self, MQALayer) else self.attn
        with ctx.maybe_use_decode_attn_tp(
            forward_batch,
            [self.wq_b, self.wo_a, self.wo_b],
            radix_attn=attn,
        ):
            if ctx.use_decode_attn_tp:
                orig = (
                    self.n_local_heads,
                    self.n_local_groups,
                    self.attn_tp_rank,
                    self.attn_tp_size,
                )
                decode_tp_size = ctx.decode_tp_size
                self.n_local_heads = self.n_heads // decode_tp_size
                self.n_local_groups = self.n_groups // decode_tp_size
                self.attn_tp_rank = ctx.decode_tp_rank
                self.attn_tp_size = decode_tp_size
                try:
                    yield
                finally:
                    (
                        self.n_local_heads,
                        self.n_local_groups,
                        self.attn_tp_rank,
                        self.attn_tp_size,
                    ) = orig
            else:
                yield


class MQALayer(MqaAttentionBase):
    is_dsv41: bool = False
    use_mega_attention: bool = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            compress_ratio=compress_ratio_override,
        )

        active_rope_scaling = None
        if self.compress_ratio:
            active_rope_scaling = dict(self.rope_scaling or {})
            active_rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=self.rope_base,
            rope_scaling=active_rope_scaling,
            is_neox_style=False,
            device=get_device().device,
        )

        if _is_npu:
            rope = Dsv4NpuRoPE.for_freqs(
                self.freqs_cis, getattr(self, "rotary_emb", None)
            )
            # fp32 tables feed the compressor gather; bf16 tables make the
            # activation-dtype gathers cast-free. Bit-identical values:
            # rounding the table once equals rounding each gathered element.
            rope.ensure_tables(torch.float32)
            rope.ensure_tables(torch.bfloat16)
            # npu_rms_norm has no weight-free overload; the per-head q norm
            # reads this cached ones vector instead of paying a per-call
            # alloc + fill.
            self.register_buffer(
                "q_rms_norm_ones",
                torch.ones(self.head_dim, dtype=torch.bfloat16),
                persistent=False,
            )

        if _is_hip:
            cos_cache = (
                self.freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            )
            sin_cache = (
                self.freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            )
            self.register_buffer("cos_cache", cos_cache, persistent=False)
            self.register_buffer("sin_cache", sin_cache, persistent=False)

        if alt_streams is not None and (
            (
                # the gfx950 overlap is validated on V4.1 only; DSv4 on ROCm keeps one stream
                (_is_cuda or (_is_gfx95_supported and self.is_dsv41))
                and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            )
            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
        ):
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        self._multi_stream_bs_limit = 128 if get_platform().is_blackwell else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio in (4, 128):
            expert_pack_quant_config = (
                quant_config
                if quant_config is not None and quant_config.get_name() == "expert_pack"
                else None
            )
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=self.freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
                quant_config=expert_pack_quant_config,
                rotary_emb=self.rotary_emb,
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=self.freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                    rotary_emb=self.rotary_emb,
                    fp4_cos=(self.cos_cache[:, 0, 0, :] if _is_hip else None),
                    fp4_sin=(self.sin_cache[:, 0, 0, :] if _is_hip else None),
                )
        elif self.compress_ratio in (1, 2):
            # The layers in between read both through the attention backend.
            if self.layer_id in config.kv_source_layer_ids:
                self.compressor = DeepseekV41Compressor(
                    hidden_size=config.hidden_size,
                    head_dim=self.head_dim,
                    compress_ratio=self.compress_ratio,
                    eps=config.rms_norm_eps,
                )
            if self.layer_id in config.index_source_layer_ids:
                self.indexer = DeepseekV41Indexer(
                    config,
                    layer_id=self.layer_id,
                    head_dim=self.head_dim,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.use_mega_attention = (
            self.is_dsv41
            and _is_cuda
            and get_exec().kernel.dsv4_attn_backend == "flashmla_mega"
        )
        self.mega_cos_sin_cache = None
        if self.use_mega_attention:
            from sglang.kernels.ops.attention.deepseek_v4_rope import (
                precompute_cos_sin_table,
            )

            self.mega_cos_sin_cache = precompute_cos_sin_table(**self.rope_kwargs)
        self.mega_wo_a = None
        self.mega_wq_b_permuted = False

        self.use_fused_qk_norm_rope = (
            _is_hip and not _is_hcu and _hip.use_fused_qk_norm_rope(self)
        )
        # Static eligibility; token count and wo_b's output format are checked
        # in forward, after weights have loaded.
        self.use_fused_wo_a = (
            self.is_dsv41
            and envs.SGLANG_DSV41_FUSED_WO_A.get()
            and _fused_wo_a_arch_supported()
            and not self.wo_a_fp8
            and not self.use_npu_arch35_mxfp8_wo_a
            and self.wo_a.weight.dtype == torch.bfloat16
            and self.wo_a.weight.shape == (self.n_local_groups * self.o_lora_rank, 4096)
            and (self.n_local_groups, self.o_lora_rank) == (2, 1024)
        )
        if _is_hip:
            _hip.init_mqa_layer(self, quant_config)

        # KV cache write is always fused into the K kernel
        # (`_compute_kv_to_cache`), so the legacy "overlap store cache" flag
        # has no effect here -- the fused path is on by default.

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.indexer is not None and hasattr(self.indexer.compressor, "fp4_cos"):
            self.indexer.compressor.fp4_cos = self.cos_cache[:, 0, 0, :]
            self.indexer.compressor.fp4_sin = self.sin_cache[:, 0, 0, :]
        return result

    def setup_mega_attention(self) -> None:
        if not self.use_mega_attention:
            return
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
            interleave_output_columns,
            interleave_query_rows,
            quantize_wo_a,
        )

        self.mega_wq_b_permuted = (
            self.n_local_heads in (64, 128) and self._wq_b_uses_flashinfer_mxfp8()
        )
        if self.mega_wq_b_permuted:
            method = self.wq_b.quant_method
            if method.use_mxfp8:
                for param in (self.wq_b.weight, self.wq_b.weight_scale_inv):
                    param.data.copy_(
                        interleave_query_rows(param.data, self.n_local_heads)
                    )
                method._process_mxfp8_linear_weight_scale(self.wq_b)
            else:
                block_n, block_k = method.weight_block_size
                sf = self.wq_b.weight_scale_inv.data.repeat_interleave(
                    block_n // 16, dim=0
                )
                sf = (
                    sf.view(self.n_local_heads, 32, -1)
                    .transpose(0, 1)
                    .reshape(sf.shape)
                )
                self.wq_b.weight.data.copy_(
                    interleave_query_rows(self.wq_b.weight.data, self.n_local_heads)
                )
                self.wq_b.weight_scale_inv.data = sf.contiguous()
                method.weight_block_size = [16, block_k]
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        self.mega_wo_a = quantize_wo_a(interleave_output_columns(wo_a))

    def _wq_b_uses_flashinfer_mxfp8(self) -> bool:
        method = self.wq_b.quant_method
        backend = getattr(method, "mxfp8_dense_backend", None)
        return bool(
            (
                getattr(method, "use_mxfp8", False)
                or getattr(method, "block_fp8_as_mxfp8", False)
            )
            and backend is not None
            and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
        )

    def _forward_mega(
        self,
        q: torch.Tensor,
        kernel_num_heads: int,
        attn_sink: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import pack_query

        if forward_batch.forward_mode.is_idle():
            return q.new_zeros(q.shape[0], self.n_local_groups, self.o_lora_rank)
        if not self.mega_wq_b_permuted:
            q = pack_query(q, kernel_num_heads)
        if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
            o = q.new_empty((q.shape[0], self.n_local_groups, self.o_lora_rank))
            bcg_deepseek_v4_mega_attention_with_output(self, q, o, attn_sink)
            return o
        return self._mega_core(q, attn_sink, forward_batch, attn_backend)

    def _mega_core(
        self,
        q: torch.Tensor,
        attn_sink: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
    ) -> torch.Tensor:
        data, scale = attn_backend.forward(
            q=q,
            k=q,
            v=q,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=attn_sink,
            save_kv_cache=False,
            cos_sin_cache=self.mega_cos_sin_cache,
        )
        groups = self.n_local_groups
        return _fp8_wo_a_einsum(
            data[:, :groups], scale[:, :groups], *self.mega_wo_a, recipe=(1, 1, 32)
        )

    def _get_npu_rope_position_cache(
        self,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        dtype: torch.dtype,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``rotary_emb`` is shared by layers with the same RoPE configuration and
        # can also be shared by the target and NextN models.  Only the immutable
        # full table is cached on it; position-gathered tensors are memoized per
        # forward (prime_rope_cos_sin / rope_cos_sin), never across forwards --
        # reusing them based on shape alone gives MTP decode the previous step's
        # RoPE values when positions change but batch size does not.
        return rope_cos_sin(
            self.freqs_cis,
            getattr(self, "rotary_emb", None),
            forward_batch,
            positions,
            dtype,
            inverse=inverse,
        )

    def accepts_mxfp8_swizzled_input(self) -> bool:
        """Whether the first projection consumes a 128x4 MXFP8 activation tuple."""
        cached = getattr(self, "_accepts_mxfp8_swizzled_input", None)
        if cached is not None:
            return cached
        if self.fuse_wqa_wkv:
            linears = [getattr(self, "wqkv_a", None)]
        else:
            # Both projections read the same activation on this path.
            linears = [getattr(self, "wq_a", None), getattr(self, "wkv", None)]

        def _takes_swizzled(linear) -> bool:
            method = getattr(linear, "quant_method", None)
            return bool(
                linear is not None
                and getattr(method, "mxfp8_dense_backend", None)
                in (
                    Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
                    Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS,
                )
                and (
                    getattr(method, "use_mxfp8", False)
                    or getattr(linear, "block_fp8_mxfp8_ready", False)
                )
            )

        ok = all(_takes_swizzled(linear) for linear in linears)
        self._accepts_mxfp8_swizzled_input = ok
        return ok

    def _normalize_q_lora(
        self, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | Mxfp8SwizzledInput]:
        # The indexer needs the BF16 normalized row; wq_b needs the quantized one.
        if _is_hip:
            return _hip.q_norm_for_wq_b(self, q)
        method = self.wq_b.quant_method
        if (
            _is_cuda
            and self.is_dsv41
            and get_platform().is_blackwell
            and q.dtype == self.q_norm.weight.dtype == torch.bfloat16
            and q.ndim == 2
            and 0 < q.shape[0] <= 8
            and q.shape[1] == 1280
            and q.stride(1) == 1
            and getattr(method, "mxfp8_dense_backend", None)
            == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
            and (
                getattr(method, "use_mxfp8", False)
                or getattr(self.wq_b, "block_fp8_mxfp8_ready", False)
            )
        ):
            from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
            from sglang.srt.runtime_context import get_exec

            if not (
                is_batch_invariant_mode_enabled()
                or get_exec().deterministic.enable_deterministic_inference
            ):
                from sglang.kernels.ops.layernorm.mxfp8_epilogue import rmsnorm_mxfp8

                y, quant, scale = rmsnorm_mxfp8(
                    q, self.q_norm.weight, self.q_norm.variance_epsilon
                )
                return y, Mxfp8SwizzledInput(quant, scale)
        q = self.q_norm(q)
        return q, q

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | Mxfp8SwizzledInput]:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        return self._normalize_q_lora(q)

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_mega_attention:
            return q
        if not self.q_head_norm:
            if q_out is not None and q_out.dtype == torch.float8_e4m3fn:
                fused_q_norm_rope(q, q_out, None, self.freqs_cis, positions)
                return q_out
            # TODO: enable zero-copy path to skip the extra no-rope copy overhead
            if (_is_cuda or _is_gfx95_supported) and q_out is not None:
                fused_q_norm_rope(
                    q,
                    q_out,
                    None,  # eps = None means no norm
                    self.freqs_cis,
                    positions,
                )
                return q_out
            fused_rope_inplace(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
            if q_out is None:
                return q
            q_out.copy_(q)
            return q_out
        if q_out is None:
            q_out = torch.empty_like(q)
        # Fused warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

    def _compute_kv_to_cache(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        qkv_a: Optional[torch.Tensor] = None,
        q_rope: Optional[torch.Tensor] = None,
    ) -> None:
        """Fused: rmsnorm + RoPE + write directly to FlashMLA paged cache.

        Replaces the bf16-kv-intermediate path. Used everywhere except the DSA
        prefill-CP case (which needs bf16 kv for the cross-rank all-gather).
        q_rope ([T, H, head_dim]) has its query heads roped by the same launch.
        """
        if envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():
            assert q_rope is None
            # Quantize the nope payload from bf16-rounded values (the fused
            # kernel quantizes from fp32 registers; the bf16 rounding moves
            # values across fp8 bins relative to bf16-sourced consumers).
            kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
            attn_backend.store_cache(
                layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch
            )
            return
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        token_to_kv_pool = get_token_to_kv_pool()
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
            # the query RoPE fused into this store is HIP only
            **({} if q_rope is None else {"q": q_rope}),
        )

    def _compute_kv_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bf16-kv path used by the DSA prefill-CP case (needs all-gather)."""
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = kv.contiguous()
        fused_norm_rope_inplace(
            kv,
            self.kv_norm.weight.data,
            self.eps,
            self.freqs_cis,
            positions,
        )
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        x_linear = x_quant if x_quant is not None else x
        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            qkv_a_ready = current_stream.record_event()

        q_lora, q_for_wqb = self._compute_q_a(x_linear, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            # Fused norm + rope + cache write -- no bf16 KV intermediate.
            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_for_wqb, positions, q_out)
        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)
        del qkv_a

        return q

    def _forward_prepare_low_ratio_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        # Both side streams are joined before returning, and nothing they read is
        # released before the join.
        assert self.alt_streams is not None
        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_sources = self.alt_streams[-1]
        x_linear = x_quant if x_quant is not None else x

        # NOTE: wait for x ready
        if self.compressor is not None:
            stream_sources.wait_stream(current_stream)
        qkv_a: Optional[torch.Tensor] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)

        if self.compressor is not None:
            with torch.cuda.stream(stream_sources):
                attn_backend.forward_low_ratio_sources(
                    layer=self,
                    x=x,
                    q_lora=None,
                    positions=positions,
                    forward_batch=forward_batch,
                    run_indexer=False,
                )

        stream_kv.wait_stream(current_stream)
        q_lora, q_for_wqb = self._compute_q_a(x_linear, qkv_a=qkv_a)
        # NOTE: wait for the q_lora ready
        if self.indexer is not None:
            stream_sources.wait_stream(current_stream)

        q = self._compute_q_b(q_for_wqb, positions, q_out)
        if self.indexer is not None:
            # Forked above, right after q_lora; recorded here, after the Q chain.
            with torch.cuda.stream(stream_sources):
                attn_backend.forward_low_ratio_sources(
                    layer=self,
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    run_compressor=False,
                )

        with torch.cuda.stream(stream_kv):
            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        current_stream.wait_stream(stream_kv)
        if self.compressor is not None or self.indexer is not None:
            current_stream.wait_stream(stream_sources)
        return q

    def _forward_prepare_multi_stream_npu(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        # NPU multi-stream: KV on stream_kv, Q on stream_q, overlapped with
        # indexer/compressor on current. rope is split; the kv-only call passes
        # kv.unsqueeze(1) as q_rope so the op sees [T,1,1,head_dim] like the
        # fused path.
        assert self.alt_streams is not None
        current_stream = torch.npu.current_stream()
        stream_kv = self.alt_streams[0]
        stream_q = self.alt_streams[1]
        stream_kv.wait_stream(current_stream)
        stream_q.wait_stream(current_stream)

        x_linear = x_quant if x_quant is not None else x
        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            qkv_a_ready = current_stream.record_event()
        if qkv_a is not None:
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
        q_lora = self.q_norm(q_lora)
        q_lora_ready = current_stream.record_event()

        # KV block on stream_kv.
        with torch.npu.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            if qkv_a is not None:
                kv = qkv_a[..., self.q_lora_rank :]
            else:
                kv, _ = self.wkv(x)
            kv = self.kv_norm(kv)
            cos4_k, sin4_k = self._get_npu_rope_position_cache(
                forward_batch, positions, kv.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                kv.unsqueeze(1),
                None,
                cos4_k,
                sin4_k,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )

        # Q block on stream_q (needs only q_lora).
        with torch.npu.stream(stream_q):
            stream_q.wait_event(q_lora_ready)
            q, _ = self.wq_b(q_lora)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            q = torch_npu.npu_rms_norm(q, self.q_rms_norm_ones, self.eps)[0]
            cos4_q, sin4_q = self._get_npu_rope_position_cache(
                forward_batch, positions, q.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                q,
                None,
                cos4_q,
                sin4_q,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            if q_out is not None:
                q_out.copy_(q)
            q.record_stream(stream_q)

        # Indexer + compressor: serial on current.
        if self.indexer is not None:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )

        # Join stream_kv + stream_q before downstream attention.
        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_q)
        del qkv_a
        return q

    def _forward_prepare_multi_stream_hip(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> torch.Tensor:
        """ATOM-style ROCm path: overlap compressors, keep Q/KV on main stream."""
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 1

        current_stream = torch.cuda.current_stream()
        stream_compressor = self.alt_streams[0]
        stream_indexer_compressor = (
            self.alt_streams[1] if len(self.alt_streams) > 1 else None
        )

        if self.compressor is not None:
            stream_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        if self.indexer is not None and stream_indexer_compressor is not None:
            stream_indexer_compressor.wait_stream(current_stream)
            with torch.cuda.stream(stream_indexer_compressor):
                attn_backend.forward_indexer_compressor(
                    x=x,
                    forward_batch=forward_batch,
                    layer_id=self.indexer.layer_id,
                    compressor=self.indexer.compressor,
                )

        x_linear = x_quant if x_quant is not None else x
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        if self.use_fused_qk_norm_rope:
            if _is_gfx95_supported or _is_gfx1250_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                    e8m0=self._q_lora_quant_e8m0(with_indexer=False),
                )
                q, _ = self.wq_b(q_for_wqb)
            else:
                q_lora = self.q_norm(q_lora)
                q, _ = self.wq_b(q_lora)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
            swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
            swa_page_size = token_to_kv_pool.swa_page_size

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
            )
        else:
            q_lora, q_for_wqb = self._normalize_q_lora(q_lora)
            q = self._compute_q_b(q_for_wqb, positions, q_out)
            self._compute_kv_to_cache(
                x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
            )

        del qkv_a

        if self.indexer is not None:
            current_stream.wait_stream(stream_compressor)
            if stream_indexer_compressor is not None:
                current_stream.wait_stream(stream_indexer_compressor)
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                skip_compressor=True,
            )
        elif self.compressor is not None:
            current_stream.wait_stream(stream_compressor)

        return q

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
        q_rope_out: Optional[torch.Tensor] = None,
        k_nope_out: Optional[torch.Tensor] = None,
        k_rope_out: Optional[torch.Tensor] = None,
        overlap_layer_split_prefetch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_linear = x_quant if x_quant is not None else x
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        # HCU shards inputs through legacy DSA CP, not the generic CP strategy.
        use_cp = (
            dsa_use_prefill_cp(forward_batch) if _is_hcu else is_cp_active(forward_batch)
        )
        layer_split_cp = (
            is_cp_cache_layer_split_deepseek_v4_pool(get_token_to_kv_pool())
            and forward_batch.forward_mode.is_context_parallel_extend()
        )
        kv: Optional[torch.Tensor]

        from hcu_sglang.utils.env_gate_compat import (
            is_unified_kv_fp8,
            is_unified_kv_triton,
        )

        unified = is_unified_kv_triton()
        fp8_2buff = is_unified_kv_fp8()
        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        # The kernel is token-indexed (q, kv and positions are all length M), so
        # a verify batch carrying several draft tokens per request is a shape it
        # already handles. Only the cache store differs between decode and
        # verify, and under fp8 that store takes the packed pair instead of bf16.
        fuse_verify = (
            envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.get()
            and forward_batch.forward_mode.is_target_verify()
        )
        # fp8 verify stores like decode: the kernel writes each token's ring row
        # (swa_loc) before attention, so no packed pair, kv copy or backend
        # scatter. Keyed off the caller's buffers the same way fuse_prefill is.
        fuse_verify_fp8 = (
            fuse_verify
            and unified
            and fp8_2buff
            and k_nope_out is not None
            and k_rope_out is not None
        )
        # Prefill under fp8 goes through the same fused store: the 2-source
        # kernel reads this chunk as its extend region in the pool's packed form,
        # and the ring write after attention reuses those same rows, so they are
        # materialised once here rather than quantized on both sides. Keyed off
        # the caller's buffers the way q_rope_out keys the packed Q, so the two
        # cannot disagree about the layout; both halves are required because the
        # nope one leaves on the kv slot and a missing one would read as "the
        # fused store did not run". Verify packs the same way but is its own arm
        # above: it stores before attention, not after.
        fuse_prefill = (
            unified
            and fp8_2buff
            and k_nope_out is not None
            and k_rope_out is not None
            and not is_decode
            and not forward_batch.forward_mode.is_target_verify()
        )
        do_fused_qk_norm_rope = not layer_split_cp and (
            (unified and (is_decode or fuse_verify or fuse_prefill))
            or (not unified and self.use_fused_qk_norm_rope)
        )

        # The indexer's wq_b reuses the fp8 quant of the normed q_lora made here.
        q_lora_quant = None
        if do_fused_qk_norm_rope:
            if _is_gfx95_supported or _is_gfx1250_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
                    e8m0=self._q_lora_quant_e8m0(with_indexer=True),
                )
                q, _ = self.wq_b(q_for_wqb)
                q_lora_quant = q_for_wqb
            else:
                q_lora, q_for_wqb = self._normalize_q_lora(q_lora)
                q, _ = self.wq_b(q_for_wqb)

            kv = (
                qkv_a[..., self.q_lora_rank :]
                if qkv_a is not None
                else self.wkv(x_linear)[0]
            )

            token_to_kv_pool = get_token_to_kv_pool()
            swa_rope_cache = None
            if unified and fuse_verify and not fuse_verify_fp8:
                # Target-verify runs through the unified_kv decode path. The
                # backend writes the current chunk's KV into the ring *before*
                # attention (save_kv_cache=True -> store_swa_into_unified ahead
                # of runtime.decode), and per-token causal index streams -- built
                # once per step in the backend metadata -- keep each draft query
                # attending only to positions up to itself. Causal masking among
                # the draft tokens comes from those index streams, not from store
                # timing. So this path skips only the fused kernel's *own* store
                # and returns kv, letting that existing causally-indexed backend
                # store run unchanged; we fuse just the norm+RoPE. swa_loc is not
                # computed -- it only addresses the kernel store this path drops.
                #
                # kv is a strided slice of qkv_a and the ring store requires a
                # contiguous buffer, so materialise it before the kernel norms
                # it in place. The unfused path pays the same copy inside
                # _compute_kv_bf16.
                #
                # Under fp8, verify takes the decode arm below instead: the ring
                # rows are per token (swa_loc is built for verify too) and are
                # written before attention either way, so the kernel stores them
                # itself, off the strided kv.
                kv = kv.contiguous()
                swa_cache, swa_loc = None, None
                swa_page_size, bf16_store = 1, True
            elif unified and fuse_prefill:
                # No pools, so the kernel norms + RoPEs + packs and writes no
                # ring row. It must not: those rows are this fwd's extend region
                # and the prefix pool has to stay as attention expects to find
                # it. The backend stores them after attention from the pair.
                swa_cache, swa_loc = None, None
                swa_page_size, bf16_store = 1, False
                # kv stays the strided slice of qkv_a. Under fp8 the kernel only
                # reads it -- the packed pair goes to k_nope_out/k_rope_out, it
                # does not norm in place -- and it takes the row stride as an
                # argument, so materialising it was a copy on every fp8 layer.
            elif unified:
                swa_cache = token_to_kv_pool.get_unified_kv(self.layer_id)
                # swa_loc is layer-independent; computed once per forward by the
                # backend and cached on the metadata (read here by every layer).
                swa_loc = attn_backend.get_unified_swa_loc(forward_batch)
                swa_page_size, bf16_store = 1, not fp8_2buff
                if fp8_2buff:
                    swa_rope_cache = token_to_kv_pool.get_unified_kv_rope(self.layer_id)
                    # kv stays the strided slice of qkv_a -- the group-quant
                    # kernel takes the row stride as an argument.
            else:
                swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
                swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
                swa_page_size, bf16_store = (
                    token_to_kv_pool.swa_page_size,
                    False,
                )
                if use_cp:
                    # Local fused KV must be gathered before writing global slots.
                    swa_cache, swa_loc = None, None

            from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
                fused_qk_norm_rope_swa_store,
            )

            q = fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=self.kv_norm.weight,
                q_rms_eps=self.eps,
                kv_rms_eps=self.eps,
                rope_head_dim=self.qk_rope_head_dim,
                cos_cache=self.cos_cache,
                sin_cache=self.sin_cache,
                positions=positions,
                swa_cache=swa_cache,
                swa_loc=swa_loc,
                swa_page_size=swa_page_size,
                q_out=q_out,
                dtype=x.dtype,
                bf16_store=bf16_store,
                fp8_2buff=fp8_2buff,
                swa_rope_cache=swa_rope_cache,
                k_nope_out=k_nope_out if fuse_prefill else None,
                k_rope_out=k_rope_out if fuse_prefill else None,
                q_rope_out=q_rope_out,
            )
            # On the verify path the kernel normed + RoPE'd kv in place and wrote
            # nothing, so hand it back: the caller feeds it to attention as the
            # current chunk (attn_k = kv) and save_kv_cache = kv is not None lets
            # the backend do its normal causally-indexed store into the ring
            # before the decode kernel runs -- exactly as the unfused path did.
            if unified and fuse_prefill:
                # The packed nope half rides out on the kv slot -- attention
                # takes it as attn_k and save_kv_cache stays on so the backend
                # does the ring write. Its rope half went to the caller's buffer,
                # which has no second return slot here. Prefill's write lands
                # after attention, verify's before it; both read this pair.
                kv = k_nope_out
            elif not unified and use_cp:
                kv = cp_materialize_global_token_order(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )
            elif not (unified and fuse_verify and not fuse_verify_fp8):
                kv = None
        elif _is_npu:
            q_lora = self.q_norm(q_lora)
            q, _ = self.wq_b(q_lora)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            q = torch_npu.npu_rms_norm(q, self.q_rms_norm_ones, self.eps)[0]

            if qkv_a is not None:
                kv = qkv_a[..., self.q_lora_rank :]
            else:
                kv, _ = self.wkv(x)
            kv = self.kv_norm(kv)

            cos4, sin4 = self._get_npu_rope_position_cache(
                forward_batch, positions, q.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                q,
                kv.unsqueeze(1),
                cos4,
                sin4,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            kv_for_cache = kv
            if use_cp:
                kv_for_cache = cp_gather_full_sequence_states(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv_for_cache,
                forward_batch=forward_batch,
            )
            kv = None
            if q_out is not None:
                q_out.copy_(q)
        else:
            q_lora, q_for_wqb = self._normalize_q_lora(q_lora)
            if _is_hip and not _is_hcu:
                q, q_lora, fuse_q_rope = _hip.compute_q_b(
                    self, q_lora, q_for_wqb, positions, q_out, unified, use_cp
                )
            elif not overlap_layer_split_prefetch:
                q = self._compute_q_b(q_for_wqb, positions, q_out)
                fuse_q_rope = False
            if unified:
                # unified_kv prefill: keep bf16 kv; the backend writes
                # the ring AFTER attention (2-source path).
                kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
                if use_cp:
                    # 2-source attention reads the whole logical chunk.
                    kv = cp_materialize_global_token_order(
                        kv.contiguous(),
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
            elif use_cp and not self.is_dsv41:
                kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
                kv = cp_materialize_global_token_order(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )
            elif use_cp:
                # every rank writes the whole chunk's window KV with the fused fp32 store
                if qkv_a is not None:
                    kv = qkv_a[..., self.q_lora_rank :]
                else:
                    kv, _ = self.wkv(x_linear)
                kv = cp_materialize_global_token_order(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                tail = attn_backend.forward_metadata.late_layer_tail
                global_positions = (
                    tail.pos_global
                    if tail is not None
                    else forward_batch.positions[: kv.shape[0]]
                )
                get_token_to_kv_pool().set_swa_key_buffer_radix_fused_norm_rope(
                    layer_id=self.layer_id,
                    swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
                    kv=kv,
                    kv_weight=self.kv_norm.weight.data,
                    eps=self.eps,
                    freqs_cis=self.freqs_cis,
                    positions=global_positions,
                )
                kv = None
            elif in_replay_tail_graph():
                assert not fuse_q_rope
                bcg_tail_kv_store(self, x_linear, positions, qkv_a)
                kv = None
            else:
                self._compute_kv_to_cache(
                    x_linear,
                    positions,
                    forward_batch,
                    attn_backend,
                    qkv_a=qkv_a,
                    q_rope=q if fuse_q_rope else None,
                )
                kv = None

        del qkv_a

        if self.compress_ratio in (1, 2) and (
            self.compressor is not None or self.indexer is not None
        ):
            if (
                forward_batch.forward_mode.is_extend()
                and is_in_breakable_cuda_graph()
                and (
                    not getattr(attn_backend, "low_ratio_prefill_graph", False)
                    or in_replay_tail_graph()
                )
            ):
                bcg_deepseek_v4_low_ratio_sources(self, x, q_lora, positions)
            else:
                attn_backend.forward_low_ratio_sources(
                    layer=self,
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                )
        else:
            use_npu_cp_full_metadata = use_cp and _is_npu
            if self.indexer is not None:
                if use_npu_cp_full_metadata:
                    with attn_backend.use_dsv4_cp_full_metadata(forward_batch):
                        attn_backend.forward_indexer_compressor(
                            x,
                            forward_batch,
                            self.indexer.layer_id,
                            self.indexer.compressor,
                        )
                    self.indexer(
                        x=x,
                        q_lora=q_lora,
                        forward_batch=forward_batch,
                        attn_backend=attn_backend,
                        skip_compressor=True,
                    )
                else:
                    self.indexer(
                        x=x,
                        q_lora=q_lora,
                        forward_batch=forward_batch,
                        attn_backend=attn_backend,
                        q_lora_quant=q_lora_quant,
                    )
            if self.compressor is not None:
                if use_npu_cp_full_metadata:
                    with attn_backend.use_dsv4_cp_full_metadata(forward_batch):
                        attn_backend.forward_core_compressor(
                            x,
                            forward_batch,
                            self.layer_id,
                            self.compressor,
                        )
                else:
                    attn_backend.forward_core_compressor(
                        x,
                        forward_batch,
                        self.layer_id,
                        self.compressor,
                    )

        if overlap_layer_split_prefetch:
            # Keep collective order unchanged. KV and indexer sources are ready;
            # Q projection only reads q_lora and can overlap the side-stream fetch.
            pool = get_token_to_kv_pool()
            maybe_prefetch_cp_kv_swa(pool, self.layer_id, forward_batch)
            maybe_prefetch_cp_kv_extra(pool, self.layer_id, forward_batch)
            q = self._compute_q_b(q_for_wqb, positions, q_out)

        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant: Optional[torch.Tensor] = None,
        defer_all_reduce: bool = False,
    ) -> Union[torch.Tensor, mhc.AttnOutput]:
        layer_split_prefill = (
            is_cp_cache_layer_split_deepseek_v4_pool(get_token_to_kv_pool())
            and forward_batch.forward_mode.is_context_parallel_extend()
        )
        if (
            not get_attn_tp_context().input_scattered
            and x.shape[0] == 0
            and not layer_split_prefill
        ):
            return x

        attn_backend = get_attn_backend()
        if TYPE_CHECKING:
            assert isinstance(
                attn_backend,
                (DeepseekV4AttnBackend, DeepseekV4HipRadixBackend),
            )

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and (
                is_in_breakable_cuda_graph()
                or x.shape[0] <= self._multi_stream_bs_limit
            )
            and not is_cp_active(forward_batch)
            and not (_is_hip and self.compressor is None)
            and self.compress_ratio not in (1, 2)
        ) or (
            _is_npu
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and self.alt_streams is not None
            and x.shape[0] <= self._multi_stream_bs_limit
            and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        )

        low_ratio_multi_stream = (
            ((_is_cuda and get_platform().is_blackwell) or _is_gfx95_supported)
            and self.compress_ratio in (1, 2)
            and self.alt_streams is not None
            and (
                forward_batch.forward_mode.is_decode()
                or (
                    forward_batch.forward_mode.is_target_verify()
                    # Other MXFP8 backends may share mutable GEMM workspace.
                    and (
                        _is_gfx95_supported
                        or getattr(self.wq_b.quant_method, "mxfp8_dense_backend", None)
                        == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
                    )
                )
            )
        )
        from hcu_sglang.utils.env_gate_compat import (
            is_unified_kv_fp8,
            is_unified_kv_triton,
        )

        unified = is_unified_kv_triton()
        unified_fp8_verify = (
            unified
            and is_unified_kv_fp8()
            and forward_batch.forward_mode.is_target_verify()
        )
        # The v4 nm asm reader takes Q in the pool's own packed form, so fp8
        # decode wants a contiguous fp8 buffer of exactly the local heads --
        # q_padded below is a FlashMLA layout and buys nothing here. Verify runs
        # that same reader over the ring, so it takes the same Q.
        unified_fp8_decode = (
            unified
            and is_unified_kv_fp8()
            and (forward_batch.forward_mode.is_decode_or_idle() or unified_fp8_verify)
        )
        # The 2-source prefill kernel wants the same packed Q plus this chunk's
        # K in the pool's layout. Verify is not prefill here even though it takes
        # the same branch below -- it reads rows the ring already holds, so it
        # goes with decode above. Multi-stream picks a different prepare that has
        # no unified arm at all, so it keeps the bf16 buffers it always had.
        unified_fp8_prefill = (
            unified
            and is_unified_kv_fp8()
            and not enable_multi_stream
            and not forward_batch.forward_mode.is_decode_or_idle()
            and not forward_batch.forward_mode.is_target_verify()
        )
        if unified_fp8_verify and not envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.get():
            # The packed pair is produced by the fused norm+RoPE store; with that
            # off the unfused arm hands the backend bf16 kv and the ring scatter
            # dies on a dtype assert that says nothing about MTP.
            raise NotImplementedError(
                "fp8 two-pool unified_kv needs the fused verify store for "
                "speculative decoding: set "
                "SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY=1, or run with "
                "SGLANG_DSV4_UNIFIED_KV_FP8=0."
            )

        tp_slice, q_padded, q_out, q_rope = slice(None), None, None, None
        k_nope, k_rope = None, None
        if self.use_mega_attention:
            kernel_num_heads = 64 if self.n_local_heads <= 64 else 128
        elif (
            self.is_dsv41
            and not self.q_head_norm
            and get_token_to_kv_pool().uniform_fp8
            and self.n_local_heads in (8, 16, 32, 64, 128)
        ):
            kernel_num_heads = self.n_local_heads
            q_out = torch.empty(
                (x.shape[0], self.n_local_heads, self.head_dim),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            )
        elif unified_fp8_decode or unified_fp8_prefill:
            # width and dtype come off the pools themselves; the kernel reads Q
            # with the kv row stride, so the two must not drift
            kv_pool = get_token_to_kv_pool()
            nope_pool = kv_pool.get_unified_kv(self.layer_id)
            rope_pool = kv_pool.get_unified_kv_rope(self.layer_id)
            q_out = nope_pool.new_empty(
                (x.shape[0], self.n_local_heads, nope_pool.shape[-1])
            )
            q_rope = rope_pool.new_empty(
                (x.shape[0], self.n_local_heads, rope_pool.shape[-1])
            )
            if unified_fp8_prefill or unified_fp8_verify:
                k_nope = nope_pool.new_empty((x.shape[0], nope_pool.shape[-1]))
                k_rope = rope_pool.new_empty((x.shape[0], rope_pool.shape[-1]))
            kernel_num_heads = self.n_local_heads
        else:
            kernel_num_heads = (
                self.n_local_heads
                if _is_hip and not unified and _hip.skip_head_pad(self)
                else self._kernel_num_heads(x.shape[0])
            )
            if kernel_num_heads != self.n_local_heads:
                # Backends without an exact-head specialization retain the existing
                # padded shape. attn_sink is sliced to this rank and padded to match.
                if self.is_dsv41:
                    # V4.1 kernels read all padded heads, so the padding must be
                    # zero. The buffer is reused per layer; no consumer may keep it.
                    want = (x.shape[0], kernel_num_heads, self.head_dim)
                    meta = getattr(attn_backend, "forward_metadata", None)
                    q_padded = getattr(meta, "q_pad_buffer", None)
                    if (
                        q_padded is None
                        or tuple(q_padded.shape) != want
                        or q_padded.dtype != x.dtype
                    ):
                        q_padded = x.new_zeros(*want)
                        if meta is not None:
                            try:
                                meta.q_pad_buffer = q_padded
                            except (AttributeError, TypeError):
                                pass
                elif _is_gfx942_supported or _is_hcu:
                    # Sparse attention may read padded heads; keep them finite
                    # on gfx942 and HCU even after allocator reuse.
                    q_padded = x.new_zeros(x.shape[0], kernel_num_heads, self.head_dim)
                else:
                    q_padded = x.new_empty(x.shape[0], kernel_num_heads, self.head_dim)
                tp_slice = slice(0, self.n_local_heads)
                q_out = q_padded[:, tp_slice, :]
        attn_sink = self._local_attn_sink(kernel_num_heads)

        overlap_layer_split_prefetch = (
            _is_hcu
            and self.is_dsv41
            and layer_split_prefill
            and self.compress_ratio in (1, 2)
            and self.dsa_enable_prefill_cp
            and dsa_use_prefill_cp(forward_batch)
            and not unified
            and not is_in_breakable_cuda_graph()
        )

        if enable_multi_stream:
            # Multi-stream path always fuses cache write into the K kernel,
            # so the bf16 KV intermediate is gone.
            if _is_hip:
                q = self._forward_prepare_multi_stream_hip(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            elif _is_npu:
                q = self._forward_prepare_multi_stream_npu(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            else:
                q = self._forward_prepare_multi_stream(
                    x,
                    positions,
                    forward_batch,
                    attn_backend,
                    q_out,
                    x_quant=x_quant,
                )
            kv = None
        elif low_ratio_multi_stream:
            q = self._forward_prepare_low_ratio_multi_stream(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
            )
            kv = None
        else:
            q, kv = self._forward_prepare(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
                q_rope_out=q_rope,
                k_nope_out=k_nope,
                k_rope_out=k_rope,
                overlap_layer_split_prefetch=overlap_layer_split_prefetch,
            )

        if unified_fp8_prefill and is_cp_active(forward_batch):
            # kv is the packed noPE half the fused prefill store left on the kv slot.
            kv, k_rope = _materialize_cp_unified_fp8_kv(
                kv, k_rope, forward_batch, torch.cuda.current_stream()
            )
        if layer_split_prefill:
            pool = get_token_to_kv_pool()
            if not overlap_layer_split_prefetch:
                maybe_prefetch_cp_kv_swa(pool, self.layer_id, forward_batch)
                if self.compress_ratio in (1, 2):
                    maybe_prefetch_cp_kv_extra(pool, self.layer_id, forward_batch)
            maybe_wait_cp_kv_swa_prefetch(pool, self.layer_id, forward_batch)
            if x.shape[0] == 0 and not get_attn_tp_context().input_scattered:
                return x

        if self.use_mega_attention:
            o = self._forward_mega(
                q, kernel_num_heads, attn_sink, forward_batch, attn_backend
            )
            return self._project_wo_b(o, defer_all_reduce)

        # save_kv_cache = kv is not None selects who writes the ring. When kv is
        # None the store was already fused into _forward_prepare* (decode) or
        # done inline, so the backend skips its own store_cache; pass `q` as a
        # sentinel for the `k is v` assert (attention won't read it once
        # save_kv_cache=False). When kv is not None (target-verify, or DSA-CP),
        # _forward_prepare* deliberately left the store off and the backend does
        # its normal causally-indexed store from attn_k = kv.
        attn_k = kv if kv is not None else q

        # HIP: the attention kernel may apply the inverse RoPE itself; the else-branch below keeps it
        inv_rope = (
            _hip.attention_inv_rope(
                self,
                positions,
                forward_batch,
                unified,
                wo_a_applies_inv_rope=self.wo_a_fp8
                and _wo_a_fp8_mxscale_fused_invrope is not None,
            )
            if _is_hip
            else None
        )
        wo_a_quant_freqs = None
        if unified:
            # only the HIP radix backend takes these; passing them always would
            # leave non-ROCm depending on the **_ in its forward() to drop them, and
            # no test on that side would notice if the **_ went away
            rope_kwargs = {}
            if q_rope is not None:
                rope_kwargs["q_rope"] = q_rope
            if k_rope is not None:
                rope_kwargs["k_rope"] = k_rope
            if inv_rope is not None:
                rope_kwargs["inv_rope"] = inv_rope
            wo_a_quant_freqs = self._wo_a_quant_freqs(forward_batch, q_rope)
            if wo_a_quant_freqs is not None:
                rope_kwargs["wo_a_quant_freqs"] = wo_a_quant_freqs
            o = attn_backend.forward(
                q=q_out if q_out is not None else q,
                k=attn_k,
                v=attn_k,
                layer=self.attn_mqa,
                forward_batch=forward_batch,
                compress_ratio=self.compress_ratio,
                attn_sink=attn_sink[: self.n_local_heads],
                save_kv_cache=kv is not None,
                **rope_kwargs,
            )
        else:
            attn_q = q_padded if q_padded is not None else q
            save_kv_cache = False
            if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
                o = attn_q.new_empty(
                    (*attn_q.shape[:-1], self.attn_mqa.v_head_dim),
                    # TRT-LLM returns BF16 even with FP8 queries.
                    dtype=x.dtype,
                )
                bcg_deepseek_v4_attention_with_output(
                    attn_q,
                    attn_k,
                    o,
                    self.attn_mqa.layer_id,
                    self.compress_ratio,
                    attn_sink,
                    save_kv_cache,
                )
            else:
                o = attn_backend.forward(
                    q=attn_q,
                    k=attn_k,
                    v=attn_k,
                    layer=self.attn_mqa,
                    forward_batch=forward_batch,
                    compress_ratio=self.compress_ratio,
                    attn_sink=attn_sink,
                    save_kv_cache=save_kv_cache,
                    # HIP only, for the same reason as rope_kwargs above
                    **({} if inv_rope is None else {"inv_rope": inv_rope}),
                )
            o = o[:, tp_slice, :]
        if wo_a_quant_freqs is not None:
            # the decode already returned the wo_a mxfp8 (codes, scales)
            G = self.n_local_groups
            o = _wo_a_fp8_mxscale_prequant(
                *o,
                self.wo_a.weight.view(G, self.o_lora_rank, -1),
                self.wo_a.weight_scale_inv.data,
            )
        elif (
            self.wo_a_fp8
            and _wo_a_fp8_mxscale_fused_invrope is not None
            and not _is_npu
        ):
            # ROCm gfx950 fused path: inverse-RoPE + per-token-group mxfp8 quant
            # in one aiter kernel on the pre-view [T,H,Dh] output, then the a8w8
            # mxscale absorb GEMM. Replaces the standalone inverse RoPE, the
            # [T,G,D] view, and the quant inside the two-kernel fp8 path below.
            G = self.n_local_groups
            cos_c, sin_c = _freqs_cis_to_cos_sin(self.freqs_cis, o.dtype, o.device)
            o = _wo_a_fp8_mxscale_fused_invrope(
                o,
                positions,
                cos_c,
                sin_c,
                G,
                self.wo_a.weight.view(G, self.o_lora_rank, -1),
                self.wo_a.weight_scale_inv.data,
            )
        else:
            fuse_mxfp8_quant = (
                self.use_flashinfer_mxfp8_wo_b and not get_forward().sp_active
            )
            fuse_rope_wo_a = (
                self.use_fused_wo_a and 0 < o.shape[0] <= _FUSED_WO_A_MAX_TOKENS
            )

            if _is_npu:
                cos4, sin4 = self._get_npu_rope_position_cache(
                    forward_batch, positions, o.dtype, inverse=True
                )
                Dsv4NpuRoPE.apply_rotary_mul_inplace(
                    o,
                    None,
                    cos4,
                    sin4,
                    qk_nope_dim=self.qk_nope_head_dim,
                )
            elif inv_rope is None and not fuse_rope_wo_a:
                # The fused path folds this in; it must not run twice.
                fused_rope_inplace(
                    o[..., -self.qk_rope_head_dim :],
                    None,
                    self.freqs_cis,
                    positions=positions,
                    inverse=True,
                )

            o = o.view(o.shape[0], self.n_local_groups, -1)

            if fuse_rope_wo_a:
                wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
                out = fused_rope_wo_a_bf16(
                    o,
                    wo_a,
                    torch.view_as_real(self.freqs_cis).flatten(1),
                    positions,
                    out_mxfp8=fuse_mxfp8_quant,
                )
                o = Mxfp8SwizzledInput(*out) if fuse_mxfp8_quant else out[0]
            elif self.use_npu_arch35_mxfp8_wo_a:
                o, o_scale = torch_npu.npu_dynamic_mx_quant(
                    o, dst_type=torch.float8_e4m3fn
                )
                o = torch_npu.npu_transpose_quant_batchmatmul(
                    o,
                    self.wo_a.weight,
                    dtype=torch.bfloat16,
                    bias=None,
                    group_sizes=(0, 0, 32),
                    x1_scale=o_scale.view(torch.float8_e8m0fnu),
                    x2_scale=self.wo_a.weight_scale_inv.view(torch.float8_e8m0fnu),
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                )
            elif self.wo_a_fp8 and _wo_a_fp8_mxscale is not None:
                # ROCm gfx950: same fp8 absorb GEMM as the DeepGEMM path below,
                # but through aiter's e8m0 block-scale batched GEMM. The
                # activation is quantized per token-group inside the helper.
                T, G, D = o.shape
                o = _wo_a_fp8_mxscale(
                    o,
                    self.wo_a.weight.view(G, self.o_lora_rank, D),
                    self.wo_a.weight_scale_inv.data,
                )
            elif self.wo_a_fp8:
                T, G, D = o.shape
                R = self.o_lora_rank
                if _FP8_WO_A_UE8M0:
                    # Blackwell (including SM120): UE8M0 scales via the dedicated
                    # JIT kernel.
                    o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_wo_a(o)
                    recipe = (1, 1, 128)
                else:
                    # sm90 (Hopper): fp32 scales.
                    o_fp8, o_s = sglang_per_token_group_quant_fp8(
                        o.reshape(T * G, D).contiguous(),
                        group_size=128,
                        scale_ue8m0=False,
                    )
                    o_fp8 = o_fp8.view(T, G, D)
                    o_s = o_s.view(T, G, -1)
                    recipe = (1, 128, 128)
                o = _fp8_wo_a_einsum(
                    o_fp8,
                    o_s,
                    self.wo_a.weight.view(G, R, D),
                    self.wo_a.weight_scale_inv.data,
                    recipe=recipe,
                )
            else:
                wo_a_weight = getattr(self.wo_a, "weight", None)
                if wo_a_weight is not None:
                    if (
                        _NPU_BF16_WO_A_GEMM
                        and forward_batch.forward_mode.is_decode()
                        and self.n_local_groups == 1
                        and o.dtype == wo_a_weight.dtype == torch.bfloat16
                        and wo_a_weight.is_contiguous()
                    ):
                        # One local group needs no grouped contraction; linear
                        # avoids materializing a transpose of the BF16 weight.
                        o = F.linear(o, wo_a_weight)
                    else:
                        wo_a = wo_a_weight.view(
                            self.n_local_groups, self.o_lora_rank, -1
                        )
                        o = _apply_wo_a_bf16_matmul(
                            o,
                            wo_a,
                            is_decode=forward_batch.forward_mode.is_decode(),
                            is_target_verify=forward_batch.forward_mode.is_target_verify(),
                            is_prefill=forward_batch.forward_mode.is_extend_without_speculative(),
                            fast_path=self.is_dsv41,
                            fuse_mxfp8_quant=fuse_mxfp8_quant,
                            fp8_grid=_is_hip and _hip.wo_a_emits_fp8_grid(self),
                            emit_fp8=_is_hip
                            and _hip.wo_b_emits_mxfp8(self, o.shape[0]),
                        )
                else:
                    o = _apply_gguf_grouped_wo_a(
                        o,
                        self.wo_a.qweight,
                        self.wo_a.qweight_type.weight_type,
                        self.o_lora_rank,
                    )

        return self._project_wo_b(o, defer_all_reduce)

    def _project_wo_b(self, o, defer_all_reduce: bool = False):
        from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce

        defer_all_reduce = defer_all_reduce and (
            self.attn_tp_size > 1
            and (
                _is_cuda
                or (
                    self.attn_tp_size == get_parallel().tp_size
                    and self.wo_b.reduce_results
                    and not self.wo_b.use_decode_attn_tp
                )
            )
            and not should_skip_mlp_all_reduce()
        )
        o, _ = self.wo_b(
            o.flatten(1) if isinstance(o, torch.Tensor) else o,
            skip_all_reduce=defer_all_reduce,
        )
        if defer_all_reduce:
            return mhc.AttnOutput(o)
        if self.attn_tp_size > 1 and self.attn_tp_size < get_parallel().tp_size:
            o = attn_tp_all_reduce(o)
        return o

    # ---- TBO op decomposition (prefill two-batch-overlap) ----
    def op_attn(self, state):
        """Run the attention forward as a single TBO op.

        Consumes the post-input-norm hidden states produced by
        ``DeepseekV4DecoderLayer.op_mhc_prepare_attn`` and stores the attention
        output for ``op_mhc_post_attn_pre_mlp``.
        """
        state.hidden_states_after_attn = self.forward(
            x=state.pop("hidden_states_after_input_norm"),
            positions=state.positions,
            forward_batch=state.forward_batch,
            x_quant=state.pop("attn_x_quant"),
            defer_all_reduce=_is_cuda,
        )


