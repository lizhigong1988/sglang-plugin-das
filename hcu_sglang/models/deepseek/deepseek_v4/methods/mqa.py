"""HCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v4 import constants as C
from hcu_sglang.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super(MQALayer, self).__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            compress_ratio=compress_ratio_override,
        )

        active_rope_scaling = None
        if self.compress_ratio in (4, 128):
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
            Dsv4NpuRoPE.for_freqs(
                self.freqs_cis, getattr(self, "rotary_emb", None)
            ).ensure_tables(torch.float32)

        if _is_hip:
            cos_cache = (
                self.freqs_cis.real.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            )
            sin_cache = (
                self.freqs_cis.imag.to(torch.bfloat16).unsqueeze(-2).unsqueeze(-2)
            )
            self.register_buffer("cos_cache", cos_cache, persistent=False)
            self.register_buffer("sin_cache", sin_cache, persistent=False)

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio in (4, 128):
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=self.freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
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

        self.use_fused_qk_norm_rope = (
            _is_hip
            and not C._is_hcu
            and envs.SGLANG_OPT_USE_FUSED_QK_NORM_ROPE.get()
        )

def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if q_out is None:
            q_out = torch.empty_like(q)
        # Fused warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        q_out: Optional[torch.Tensor] = None,
        x_quant=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_linear = x_quant if x_quant is not None else x
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x_linear)
            q_lora = qkv_a[..., : self.q_lora_rank]
        else:
            q_lora, _ = self.wq_a(x_linear)
            qkv_a = None

        use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        kv: Optional[torch.Tensor]

        from hcu_sglang.utils.env_gate_compat import (
            is_unified_kv_triton,
        )

        unified = is_unified_kv_triton()
        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        do_fused_store = (unified and is_decode) or (
            not unified and self.use_fused_qk_norm_rope
        )

        if do_fused_store:
            if _is_gfx95_supported:
                q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(
                    q_lora,
                    self.q_norm.weight,
                    self.q_norm.variance_epsilon,
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

            token_to_kv_pool = get_token_to_kv_pool()
            if unified:
                swa_cache = token_to_kv_pool.get_unified_kv(self.layer_id)
                # swa_loc is layer-independent; computed once per forward by the
                # backend and cached on the metadata (read here by every layer).
                swa_loc = attn_backend.get_unified_swa_loc(forward_batch)
                swa_page_size, bf16_store = 1, True
            else:
                swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
                swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
                swa_page_size, bf16_store = (
                    token_to_kv_pool.swa_kv_pool.page_size,
                    False,
                )

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
            )
            kv = None

            if not unified and use_cp:
                # DSA CP: keep bf16 kv around for the cross-rank all-gather, then
                # write to the FlashMLA cache after gather.
                kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
                kv = cp_materialize_global_token_order(
                    kv.contiguous(),
                    forward_batch,
                    torch.cuda.current_stream(),
                )
        elif _is_npu:
            q_lora = self.q_norm(q_lora)
            q, _ = self.wq_b(q_lora)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            _dummy = q.new_ones(q.shape[-1])
            q = torch_npu.npu_rms_norm(q, _dummy, self.eps)[0]

            if qkv_a is not None:
                kv = qkv_a[..., self.q_lora_rank :]
            else:
                kv, _ = self.wkv(x)
            kv = self.kv_norm(kv)

            cos4, sin4 = self._get_npu_rope_position_cache(
                positions, q.dtype, inverse=False
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                q,
                kv.unsqueeze(1),
                cos4,
                sin4,
                qk_nope_dim=self.qk_nope_head_dim,
            )
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )
            kv = None
            if q_out is not None:
                q_out.copy_(q)
        else:
            q_lora = self.q_norm(q_lora)
            q = self._compute_q_b(q_lora, positions, q_out)
            if unified:
                # unified_kv prefill: keep bf16 kv; the backend writes
                # the ring AFTER attention (2-source path).
                kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
                # HIP/ROCm-only: the unified_kv 2-source prefill path is exclusive
                # to DeepseekV4HipRadixBackend. Guard with _is_hip so this CP
                # all-gather never enters the NVIDIA (DeepseekV4AttnBackend) path.
                if use_cp and _is_hip:
                    # unified_kv + DSA CP: the 2-source prefill path needs the
                    # FULL current-chunk KV (extend source + ring write), so
                    # all-gather the per-rank bf16 KV across the CP group.
                    kv = cp_materialize_global_token_order(
                        kv.contiguous(),
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
            elif use_cp:
                # NSA CP: keep bf16 kv around for the cross-rank all-gather, then
                # write to the FlashMLA cache after gather.
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
            else:
                self._compute_kv_to_cache(
                    x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a
                )
                kv = None

        del qkv_a

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

        return q, kv

def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        x_quant=None,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
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
            and not (self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch))
            and not (_is_hip and self.compressor is None)
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.attn_tp_size > 1:
            # FlashMLA's fp8 sparse decode kernel only specializes h_q for {64, 128}.
            # Pad the per-rank heads to 64 (not the full n_heads) when they fit, to
            # dispatch the cheaper decode::head64 variant; attn_sink is sliced to
            # this rank and padded to match.
            padded_num_heads = 64 if self.n_local_heads <= 64 else self.n_heads
            # Only [0:n_local_heads] is written below. Uninitialized padded TP
            # heads inject NaN into attention on gfx942 (fnuz), so zero-init
            # there; other archs tolerate new_empty and skip the per-forward
            # memset.
            if _is_gfx942_supported:
                q_padded = x.new_zeros(x.shape[0], padded_num_heads, self.head_dim)
            else:
                q_padded = x.new_empty(x.shape[0], padded_num_heads, self.head_dim)
            tp_slice = slice(0, self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]
        attn_sink = self._local_attn_sink()

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
        else:
            q, kv = self._forward_prepare(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
            )

        # The cache write is always fused / already done by _forward_prepare* --
        # tell the backend to skip its own store_cache. When `kv is None`
        # (no DSA-CP), pass `q` as a sentinel for the `k is v` assert; the
        # attention path doesn't read it once `save_kv_cache=False`.
        attn_k = kv if kv is not None else q
        from hcu_sglang.utils.env_gate_compat import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            o = attn_backend.forward(
                q=q_out if q_out is not None else q,
                k=attn_k,
                v=attn_k,
                layer=self.attn_mqa,
                forward_batch=forward_batch,
                compress_ratio=self.compress_ratio,
                attn_sink=self.attn_sink,
                save_kv_cache=kv is not None,
            )
        else:
            attn_q = q_padded if q_padded is not None else q
            save_kv_cache = False
            if forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph():
                o = attn_q.new_empty(
                    (*attn_q.shape[:-1], self.attn_mqa.v_head_dim),
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
                )
            o = o[:, tp_slice, :]
        if _is_npu:
            cos4, sin4 = self._get_npu_rope_position_cache(
                positions, o.dtype, inverse=True
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                o,
                None,
                cos4,
                sin4,
                qk_nope_dim=self.qk_nope_head_dim,
            )
        else:
            fused_rope_inplace(
                o[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
                inverse=True,
            )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            from sglang.srt.layers import deep_gemm_wrapper

            T, G, D = o.shape
            R = self.o_lora_rank
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
                # sm100 (Blackwell): ue8m0 scales via the dedicated JIT kernel.
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
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8, o_s),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=recipe,
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))
        if self.attn_tp_size > 1 and self.attn_tp_size < get_parallel().tp_size:
            o = attn_tp_all_reduce(o)

        return o
