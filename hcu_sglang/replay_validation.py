from __future__ import annotations
from sglang.srt.arg_groups import deepseek_v4_hook as N
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()


def validate_deepseek_v41_encoder_replay_budget(server_args: ServerArgs) -> None:
    cfg = N.resolving_view(server_args)
    if not cfg.enable_encoder_swa_bounded_replay:
        return
    # A hit's first chunk also runs up to 128 replay rows and floors its extend to
    # whole pages, so a smaller chunk leaves the hit no page and it never admits.
    min_chunk = 128 + (cfg.page_size or 1)
    if (
        cfg.max_running_requests is None
        or cfg.max_running_requests <= 0
        or not cfg.chunked_prefill_size
        or cfg.chunked_prefill_size < min_chunk
    ):
        raise ValueError(
            "encoder SWA replay requires --max-running-requests > 0 and at least "
            f"{min_chunk} chunked prefill tokens per attention DP rank (128 replay "
            "tokens plus one page)"
        )


def validate_deepseek_v41_features(server_args: ServerArgs) -> None:
    from hcu_sglang.utils.env_gate_compat import (
        is_unified_kv_triton,
    )

    cfg = N.resolving_view(server_args)
    if N.model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        if cfg.enable_encoder_swa_bounded_replay:
            raise ValueError(
                "--enable-encoder-swa-bounded-replay requires DeepSeek-V4.1"
            )
        return
    if (
        cfg.dsv4_attn_backend == "trtllm"
        and cfg.cuda_graph_config.prefill.backend
        not in (N.Backend.DISABLED, N.Backend.BREAKABLE)
    ):
        raise ValueError(
            "DeepSeek-V4.1 TRT-LLM requires --cuda-graph-backend-prefill "
            "disabled or breakable"
        )
    if cfg.enable_encoder_swa_bounded_replay:
        incompatible = (
            (
                "hardware other than CUDA or gfx950",
                not (
                    N.get_platform().is_cuda
                    or (N.get_platform().is_hip and N.is_gfx95_supported())
                ),
            ),
            (
                "prefill CUDA graphs other than breakable",
                cfg.cuda_graph_config.prefill.backend
                not in (N.Backend.DISABLED, N.Backend.BREAKABLE),
            ),
            # ROCm replays hits in a forward of their own, outside the graph.
            (
                "the prefill CUDA graph on ROCm",
                cfg.cuda_graph_config.prefill.backend != N.Backend.DISABLED
                and N.get_platform().is_hip,
            ),
            ("context parallelism", cfg.attn_cp_size > 1),
            # ROCm replays hits in a forward of their own, which only the hitting
            # attention-DP ranks would enter; CUDA folds them into the extend.
            ("DP attention on ROCm", N.get_platform().is_hip and N.attn_dp_enabled_of(cfg)),
            ("external cache linker", cfg.enable_unified_cache_external_linker),
            ("unified memory", cfg.enable_unified_memory),
            # Decode receives the last SWA window from prefill and never replays.
            ("PD decode", cfg.disaggregation_mode == "decode"),
            # On ROCm the draft keeps paged SWA, so it cannot share the window's pages.
            (
                "PD prefill on ROCm",
                cfg.disaggregation_mode == "prefill" and N.get_platform().is_hip,
            ),
            ("mixed prefill/decode", cfg.enable_mixed_chunk),
            ("LoRA", cfg.enable_lora),
            ("radix sessions", cfg.enable_session_radix_cache),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    f"--enable-encoder-swa-bounded-replay does not support {feature} yet"
                )

    unsupported = (
        (
            "speculative decoding other than DSpark",
            cfg.speculative_algorithm is not None
            and str(cfg.speculative_algorithm).upper() != "DSPARK",
        ),
        ("HiSparse", cfg.enable_hisparse),
        ("the unified KV layout", is_unified_kv_triton()),
        (
            "TRT-LLM with encoder SWA bounded replay",
            cfg.dsv4_attn_backend == "trtllm" and cfg.enable_encoder_swa_bounded_replay,
        ),
        ("two-batch overlap", cfg.enable_two_batch_overlap),
        ("pipeline parallelism", cfg.pp_size > 1),
    )
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(
                f"DeepSeek-V4.1 does not support {feature} yet; disable it to "
                "serve this model."
            )

    if cfg.disaggregation_mode != "null" and cfg.speculative_algorithm is not None:
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        # CP size may still be implicit here; validate_deepseek_v4_cp resolves it.
        supported_prefill_cp = (
            cfg.disaggregation_mode == "prefill"
            and cfg.enable_prefill_cp
            and cfg.cp_strategy == "interleave"
        )
        if (
            read_ragged_verify_mode() is not RaggedVerifyMode.STATIC
            or cfg.disaggregation_transfer_backend not in ("mooncake", "nixl")
            or (
                (cfg.enable_prefill_cp or cfg.attn_cp_size != 1)
                and not supported_prefill_cp
            )
            or cfg.dcp_size != 1
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD requires static verify, Mooncake or NIXL, "
                "DCP=1, and CP=1 on decode. Prefill CP requires "
                "--enable-prefill-cp with the interleave strategy. "
                "DP attention is supported when "
                "both servers use the same block size and target/draft KV layout."
            )

        if _is_hcu:
            # HCU PD additionally pins the replica/attention DP split per role.
            if cfg.disaggregation_mode == "prefill":
                topology_invalid = not (
                    cfg.dp_size == 1
                    and cfg.attn_dp_size == 1
                    and cfg.enable_prefill_cp
                    and cfg.attn_cp_size == cfg.tp_size
                )
                topology = "P requires replica DP=1, attention DP=1 and interleave prefill CP=TP"
            else:
                topology_invalid = not (
                    cfg.disaggregation_mode == "decode"
                    # The pipeline normalizes the legacy DP-attention flags
                    # before this hook; dp_size now counts model replicas.
                    and cfg.dp_size == 1
                    and cfg.attn_dp_size == cfg.tp_size
                    and N.attn_dp_enabled_of(cfg)
                    and not cfg.enable_prefill_cp
                    and cfg.attn_cp_size == 1
                )
                topology = "D requires replica DP=1, attention DP=TP and CP=1"
            if topology_invalid:
                raise ValueError(
                    f"DeepSeek-V4.1 DSpark PD on HCU requires {topology}. "
                    "Both servers must use the same DSpark block size and TP size."
                )

    incompatible = (
        (
            "the prefill CUDA graph",
            cfg.cuda_graph_config.prefill.backend
            not in (N.Backend.DISABLED, N.Backend.BREAKABLE),
        ),
        (
            "DP attention other than attention TP 1 with supported MoE backends",
            N.attn_dp_enabled_of(cfg)
            and not (
                cfg.attn_dp_size == cfg.tp_size
                and cfg.attn_cp_size == 1
                and (
                    (N.get_platform().is_hip and cfg.moe_a2a_backend == "none")
                    or (
                        N.get_platform().is_cuda
                        and cfg.moe_a2a_backend in ("none", "megamoe")
                    )
                )
            ),
        ),
    )
    if cfg.enable_decoder_swa_bounded_replay:
        for feature, unsupported in incompatible:
            if unsupported:
                raise ValueError(
                    "--enable-decoder-swa-bounded-replay cannot be combined with "
                    f"{feature} yet; disable one of them."
                )
        return

    if cfg.enable_decoder_swa_bounded_replay is False:
        return

    enabled = (
        (N.get_platform().is_cuda or N.get_platform().is_hip)
        # Both PD roles run it on CUDA; PD on ROCm has not been run with it.
        and (cfg.disaggregation_mode == "null" or N.get_platform().is_cuda)
        and not (cfg.enable_prefill_cp or cfg.attn_cp_size != 1 or cfg.dcp_size != 1)
        # Returning every hidden state runs every prompt row: nothing to trim.
        and not cfg.enable_return_hidden_states
        and not any(enabled for _, enabled in incompatible)
    )
    N.declare_resolution(
        server_args,
        "validate_deepseek_v41_features",
        enable_decoder_swa_bounded_replay=enabled,
    )
    N.logger.info("DeepSeek-V4.1 decoder SWA bounded replay resolved to %s", enabled)


def around_features(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return validate_deepseek_v41_features(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features", around_features, HookType.AROUND)
