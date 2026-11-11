# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for the channel-wise W4A8 marlin scheme
(``hcu_sglang/layers/quantization/slimquant_w4a8_marlin.py``), pinned to the
Kimi-K3.1 channel-wise W4A8 checkpoint contract:

- Kimi-K3.1-Pro-CHANNEL-INT4-w4a8 ships ``quant_method: slimquant_w4a8`` with a
  ``modules_to_not_convert`` keep-BF16 list (dense attention/mlp projections,
  indexer, norms); only routed experts (and the latent routed-expert
  projections) stay INT4. The plugin marlin override maps this checkpoint to
  ``slimquant_w4a8_marlin`` when the user requests a marlin alias.
- ``SlimQuantW4A8Int8MarlinConfig`` declares
  ``supports_kimi_k3_quantized_latent_projections = True`` so sglang's
  kimi_k3 model passes the quant config into the latent routed-expert
  projections (upstream ``sglang/srt/models/kimi_k3.py`` getattr gate).
- Kimi-K3.1 dims: hidden_size=7168, moe_intermediate_size=3072,
  num_experts=896 — both TP4 and TP8 shards satisfy the marlin repack tiles.

The module under test imports sglang/triton/aiter-heavy names at module
scope, so it is loaded against minimal in-process stubs installed for the
duration of the class (snapshot/restore keeps other tests unaffected).
``should_ignore_layer`` is a faithful port of upstream
``sglang/srt/layers/quantization/compressed_tensors/utils.py`` (suffix
matching, ``re:`` regexes, fused all-shards-or-error) so the ignore
semantics exercised here are the real upstream ones.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "hcu_sglang" / "layers" / "quantization" / "slimquant_w4a8_marlin.py"

# --- Kimi-K3.1-Pro-CHANNEL-INT4-w4a8 checkpoint contract (real values) ------
KIMI_K31_ARCH = "KimiK31ForConditionalGeneration"
KIMI_K31_LMSLIM_QUANT = {
    "activation_scheme": "dynamic",
    "quant_method": "slimquant_w4a8",
    "modules_to_not_convert": [
        "lm_head", "embed_tokens", "vision_tower", "mm_projector",
        "block_sparse_moe.gate", "routed_expert_norm",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "self_attention_res_proj", "mlp_res_proj", "output_attn_res_proj",
        "self_attn.o_proj", "self_attn.kv_b_proj",
        "self_attn.f_a_proj", "self_attn.f_b_proj", "self_attn.g_proj",
        "self_attn.b_proj", "self_attn.q_conv1d",
        "self_attn.k_conv1d", "self_attn.v_conv1d",
        "self_attn.indexer", "self_attn.o_norm",
        "self_attn.q_a_layernorm", "self_attn.kv_a_layernorm",
    ],
}
# HCU-converted Kimi-K3 checkpoints carry the keep-BF16 list as a native
# compressed-tensors ``ignore`` list (plain suffix entries + ``re:`` regexes),
# per the slimquant_w4a8_marlin get_quant_method comment.
KIMI_K31_HCU_IGNORE = [
    "lm_head",
    "re:.*self_attn\\.(q_conv1d|k_conv1d|v_conv1d|indexer|o_norm|q_a_layernorm|kv_a_layernorm)",
    "self_attn.o_proj",
    "self_attn.kv_b_proj",
    "self_attn.f_a_proj",
    "self_attn.f_b_proj",
    "self_attn.g_proj",
    "self_attn.b_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "routed_expert_norm",
    "self_attention_res_proj",
    "mlp_res_proj",
    "output_attn_res_proj",
]

# Kimi-K3.1 text_config: hidden 7168, moe_intermediate 3072, 896 experts.
KIMI_K31_HIDDEN = 7168
KIMI_K31_MOE_INTER = 3072


# --- faithful port of upstream compressed_tensors ignore matching -----------
def _is_equal_or_regex_match(value, target, check_contains=False):
    if target.startswith("re:"):
        import re

        if re.match(target[3:], value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def check_equal_or_regex_match(layer_name, targets):
    for target in targets:
        if target.startswith("re:"):
            if _is_equal_or_regex_match(layer_name, target):
                return True
        elif target == layer_name or layer_name.endswith("." + target):
            return True
    return False


def should_ignore_layer(layer_name, ignore=(), fused_mapping={}):
    if layer_name is None:
        return False
    proj_name = layer_name.split(".")[-1]
    if proj_name in fused_mapping and layer_name not in ignore:
        shard_proj_names = fused_mapping[proj_name]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]
        should_ignore_layer_value = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(shard_name, ignore)
            if should_ignore_layer_value is None:
                should_ignore_layer_value = should_ignore_shard
            elif should_ignore_shard != should_ignore_layer_value:
                raise ValueError(
                    "Found different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. SGLang "
                    "requires all to use the same scheme."
                )
    else:
        should_ignore_layer_value = check_equal_or_regex_match(layer_name, ignore)
    return should_ignore_layer_value


class _EnvVar:
    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value


def _normalize_w4a8_tpmoe_backend(requested_backend, *, env_name):
    backends = {"auto", "lightop", "aiter", "triton"}
    backend = requested_backend.strip().lower()
    if backend not in backends:
        raise ValueError(f"Unsupported {env_name}={requested_backend!r}.")
    return backend


def _install_stubs():
    """Install minimal sglang/triton/hcu stubs so the module imports offline."""

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        for key, value in attrs.items():
            setattr(m, key, value)
        _record_previous(name)
        sys.modules[name] = m
        installed.append(name)
        return m

    installed = []
    # Captured BEFORE each stub overwrites sys.modules so teardown restores
    # the true pre-install state (real module or absent), never the stub
    # itself — a post-install snapshot would leak every stub into later tests.
    previous = {}

    def _record_previous(name):
        if name not in previous:
            previous[name] = sys.modules.get(name)

    class _PermissiveModule(types.ModuleType):
        # torch._dynamo probes triton.language.<attr> at import time; return a
        # fresh hashable placeholder for anything it asks for.
        def __getattr__(self, name):
            value = object()
            object.__setattr__(self, name, value)
            return value

    triton = mod("triton")
    triton.jit = lambda fn=None, **kwargs: fn if callable(fn) else (lambda f: f)
    language = _PermissiveModule("triton.language")
    language.__path__ = []
    _record_previous("triton.language")
    sys.modules["triton.language"] = language
    triton.language = language

    sglang = mod("sglang")
    mod("sglang.srt")
    mod("sglang.srt.runtime_context", get_parallel=lambda *a, **k: None)
    mod("sglang.srt.environ", envs=SimpleNamespace(
        SGLANG_W4A8_TPMOE_BACKEND=_EnvVar("auto"),
        SGLANG_DSPARK_FORCE_W4A8_TPMOE_BACKEND=_EnvVar(None),
    ))
    layers = mod("sglang.srt.layers")
    linear_mod = mod("sglang.srt.layers.linear")
    linear_mod.LinearBase = type("LinearBase", (), {})
    moe_mod = mod(
        "sglang.srt.layers.moe",
        MoeRunner=type("MoeRunner", (), {}),
        MoeRunnerBackend=type("MoeRunnerBackend", (), {}),
        MoeRunnerConfig=type("MoeRunnerConfig", (), {}),
        get_deepep_mode=lambda *a, **k: None,
        get_moe_a2a_backend=lambda: SimpleNamespace(is_megamoe=lambda: False),
    )
    mod("sglang.srt.layers.moe.moe_runner")
    mod("sglang.srt.layers.moe.moe_runner.triton",
        TritonMoeQuantInfo=type("TritonMoeQuantInfo", (), {}))
    fused_moe = mod("sglang.srt.layers.moe.fused_moe_triton")
    fused_moe.FusedMoE = type("FusedMoE", (), {})
    quant = mod("sglang.srt.layers.quantization",
                QuantizationConfig=type("QuantizationConfig", (), {"__init__": lambda self: None}))
    mod("sglang.srt.layers.quantization.base_config",
        FusedMoEMethodBase=type("FusedMoEMethodBase", (), {}),
        QuantizeMethodBase=type("QuantizeMethodBase", (), {}))
    ct = mod("sglang.srt.layers.quantization.compressed_tensors")
    mod("sglang.srt.layers.quantization.compressed_tensors.utils",
        should_ignore_layer=should_ignore_layer,
        check_equal_or_regex_match=check_equal_or_regex_match,
        _is_equal_or_regex_match=_is_equal_or_regex_match)
    unquant = mod("sglang.srt.layers.quantization.unquant")
    unquant.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {"__init__": lambda self, *a, **k: None})
    unquant.UnquantizedFusedMoEMethod = type("UnquantizedFusedMoEMethod", (), {"__init__": lambda self, *a, **k: None})
    mod("sglang.srt.utils",
        get_bool_env_var=lambda name, default=False, **k: default,
        set_weight_attrs=lambda *a, **k: None)

    hcu = mod("hcu_sglang")
    mod("hcu_sglang.layers")
    mod("hcu_sglang.layers.moe")
    mod(
        "hcu_sglang.layers.moe.w4a8_backend",
        W4A8_TPMOE_BACKEND_AUTO="auto",
        W4A8_TPMOE_BACKEND_LIGHTOP="lightop",
        W4A8_TPMOE_BACKEND_AITER="aiter",
        W4A8_TPMOE_BACKEND_TRITON="triton",
        W4A8_TPMOE_BACKEND_ENV="SGLANG_W4A8_TPMOE_BACKEND",
        normalize_w4a8_tpmoe_backend=_normalize_w4a8_tpmoe_backend,
        get_dspark_w4a8_tpmoe_backend_override=lambda: None,
    )
    mod("hcu_sglang.layers.quantization")
    slim_linear = mod("hcu_sglang.layers.quantization.slimquant_w4a8")
    slim_linear.SlimQuantW4A8Int8LinearMethod = type(
        "SlimQuantW4A8Int8LinearMethod", (), {"__init__": lambda self, config: None})
    mod("hcu_sglang.layers.quantization.compressed_w4a8",
        HCUW4A8Int8DeepEPMoEMethod=type(
            "HCUW4A8Int8DeepEPMoEMethod", (), {"__init__": lambda self, *a, **k: None}))
    mod("hcu_sglang.layers.quantization.w4a8_utils",
        w4a8_weight_repack_impl=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("repack stub must not run offline")))
    mod("hcu_sglang.utils")
    mod("hcu_sglang.utils.detection", is_hcu=lambda: False)
    return {
        "linear": linear_mod,
        "moe": moe_mod,
        "quant": quant,
        "unquant": unquant,
        "fused_moe": fused_moe,
        "installed": installed,
        "previous": previous,
    }


class SlimQuantW4A8MarlinOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import torch (including the lazily-reached dynamo/inductor modules)
        # while no triton stub is visible: they probe for triton at import
        # time and would trip over the offline stub later.
        import torch  # noqa: F401
        import torch._dynamo.utils  # noqa: F401
        import torch._inductor.runtime.hints  # noqa: F401
        cls._stubs = _install_stubs()
        # Only the stub entries get torn down (restored to whatever they were
        # BEFORE the install — the installer captured that); real modules
        # imported along the way (torch & co) must stay cached — evicting them
        # breaks later tests in the same process.
        cls._stub_modules = cls._stubs["previous"]
        spec = importlib.util.spec_from_file_location(
            "slimquant_w4a8_marlin_under_test", MODULE_PATH
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    @classmethod
    def tearDownClass(cls):
        for name, previous in cls._stub_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    # -- scheme identity -----------------------------------------------------

    def test_config_declares_quantized_latent_projection_support(self):
        # kimi_k3.py gates latent routed-expert projections on this flag; the
        # channel-wise W4A8 checkpoint quantizes them, so it must be True
        # (mirrors upstream test/registered/expert_pack/test_kimi_k3_gguf.py).
        self.assertTrue(
            self.mod.SlimQuantW4A8Int8MarlinConfig.supports_kimi_k3_quantized_latent_projections
        )

    def test_override_routes_kimi_k31_lmslim_checkpoint_to_marlin(self):
        override = self.mod.SlimQuantW4A8Int8MarlinConfig.override_quantization_method
        get_name = self.mod.SlimQuantW4A8Int8MarlinConfig.get_name
        hf_cfg = dict(KIMI_K31_LMSLIM_QUANT)
        self.assertEqual(override(hf_cfg, "slimquant_w4a8_marlin"), "slimquant_w4a8_marlin")
        self.assertEqual(override(hf_cfg, "slimquant_marlin"), get_name())
        # Without an explicit marlin user_quant the plugin must not hijack
        # the checkpoint (the adapter's slimquant_w4a8 fallback owns it).
        self.assertIsNone(override(hf_cfg, None))
        self.assertIsNone(override(hf_cfg, "fp8"))

    def test_override_accepts_channelwise_pack_quantized_constellation(self):
        override = self.mod.SlimQuantW4A8Int8MarlinConfig.override_quantization_method
        channel_w4a8_group = {
            "weights": {
                "num_bits": 4, "type": "int", "strategy": "channel",
                "symmetric": True, "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8, "type": "int", "strategy": "token",
                "symmetric": True, "dynamic": True,
            },
        }

        def hf_cfg(weights=None, activations=None):
            group = {
                "weights": {**channel_w4a8_group["weights"], **(weights or {})},
                "input_activations": {
                    **channel_w4a8_group["input_activations"], **(activations or {})
                },
            }
            return {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {"group_0": group},
            }

        name = self.mod.SlimQuantW4A8Int8MarlinConfig.get_name()
        self.assertEqual(override(hf_cfg(), name), name)
        mutations = [
            ("weight group strategy", {"strategy": "group"}, None),
            ("weight num_bits", {"num_bits": 8}, None),
            ("weight asymmetric", {"symmetric": False}, None),
            ("weight dynamic", {"dynamic": True}, None),
            ("weight float type", {"type": "float"}, None),
            ("activation channel strategy", None, {"strategy": "channel"}),
            ("activation static", None, {"dynamic": False}),
            ("activation asymmetric", None, {"symmetric": False}),
            ("activation 4-bit", None, {"num_bits": 4}),
        ]
        for label, weights, activations in mutations:
            with self.subTest(label):
                self.assertIsNone(override(hf_cfg(weights, activations), name))
        # Empty groups / other formats never claim the checkpoint.
        self.assertIsNone(override(
            {"quant_method": "compressed-tensors", "format": "pack-quantized",
             "config_groups": {}}, name))
        self.assertIsNone(override(
            {"quant_method": "compressed-tensors", "format": "int-quantized",
             "config_groups": {"g": channel_w4a8_group}}, name))

    def test_from_config_arch_and_ignore_parsing(self):
        from_config = self.mod.SlimQuantW4A8Int8MarlinConfig.from_config
        kimi = from_config({
            "hf_config": {"architectures": [KIMI_K31_ARCH]},
            "ignore": KIMI_K31_HCU_IGNORE,
        })
        self.assertFalse(kimi.experts_only_linear)  # dense Linear quantized unless ignored
        self.assertEqual(kimi.ignore, KIMI_K31_HCU_IGNORE)
        self.assertFalse(kimi.compressed_tensors_packed)

        qwen = from_config({
            "hf_config": {"architectures": ["Qwen4ExpForConditionalGeneration"]},
            "mixed_precision": {"w8a8_include": {"modules": ["mlp.gate_up_proj"]}},
            "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
            "format": "pack-quantized",
        })
        self.assertTrue(qwen.experts_only_linear)
        self.assertEqual(qwen.w8a8_include, ["mlp.gate_up_proj"])
        self.assertEqual(qwen.packed_modules_mapping, {"gate_up_proj": ["gate_proj", "up_proj"]})
        self.assertTrue(qwen.compressed_tensors_packed)

    # -- Kimi-K3.1 layer dispatch -------------------------------------------

    def test_kimi_k31_ignore_dispatch_keeps_dense_bf16_and_quantizes_experts(self):
        cfg = self.mod.SlimQuantW4A8Int8MarlinConfig(ignore=list(KIMI_K31_HCU_IGNORE))
        LinearBase = self._stubs["linear"].LinearBase
        Slim = self.mod.SlimQuantW4A8Int8LinearMethod
        Unquant = self._stubs["unquant"].UnquantizedLinearMethod
        cases = [
            # (prefix, expected) — keep-BF16 dense side of the checkpoint
            ("lm_head", Unquant),
            ("model.layers.0.self_attn.o_proj", Unquant),
            ("model.layers.0.self_attn.kv_b_proj", Unquant),
            ("model.layers.0.self_attn.indexer", Unquant),
            ("model.layers.0.self_attn.q_conv1d", Unquant),
            ("model.layers.0.mlp.gate_proj", Unquant),
            ("model.layers.0.self_attention_res_proj", Unquant),
            # INT4 side: routed experts + latent routed-expert projections
            ("model.layers.0.mlp.experts.0.gate_proj", Slim),
            ("model.layers.92.mlp.experts.895.down_proj", Slim),
            ("model.layers.0.mlp.routed_expert_down_proj", Slim),
            ("model.layers.0.mlp.routed_expert_up_proj", Slim),
        ]
        for prefix, expected in cases:
            with self.subTest(prefix):
                method = cfg.get_quant_method(LinearBase(), prefix)
                self.assertIsInstance(method, expected)

        # Parent-module ignore entries must not swallow quantized children
        # (upstream suffix semantics: no prefix matching).
        cfg_child = self.mod.SlimQuantW4A8Int8MarlinConfig(
            ignore=["model.layers.0.mlp.experts.0"]
        )
        self.assertIsInstance(
            cfg_child.get_quant_method(
                LinearBase(), "model.layers.0.mlp.experts.0.gate_proj"),
            Slim,
        )

    def test_experts_only_linear_uses_w8a8_include_gating(self):
        # w8a8_include carries the *unfused* checkpoint shard names; a fused
        # layer goes W8A8 only when every shard is listed.
        cfg = self.mod.SlimQuantW4A8Int8MarlinConfig(
            experts_only_linear=True,
            w8a8_include=[
                "model.layers.*.mlp.gate_proj", "model.layers.*.mlp.up_proj",
            ],
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
        )
        LinearBase = self._stubs["linear"].LinearBase
        Slim = self.mod.SlimQuantW4A8Int8LinearMethod
        Unquant = self._stubs["unquant"].UnquantizedLinearMethod
        # Not listed -> dense stays BF16
        self.assertIsInstance(
            cfg.get_quant_method(LinearBase(), "model.layers.0.mlp.down_proj"), Unquant
        )
        # Fused layer is W8A8 only when every shard is listed
        self.assertIsInstance(
            cfg.get_quant_method(LinearBase(), "model.layers.3.mlp.gate_up_proj"), Slim
        )
        partial = self.mod.SlimQuantW4A8Int8MarlinConfig(
            experts_only_linear=True,
            w8a8_include=["model.layers.*.mlp.gate_proj"],
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
        )
        self.assertIsInstance(
            partial.get_quant_method(LinearBase(), "model.layers.3.mlp.gate_up_proj"),
            Unquant,
        )
        # experts_only False (Kimi/DeepSeek side): dense Linear quantized
        dense = self.mod.SlimQuantW4A8Int8MarlinConfig()
        self.assertIsInstance(
            dense.get_quant_method(LinearBase(), "model.layers.0.mlp.down_proj"), Slim
        )

    def assertIsMethod(self, method, expected_cls):
        # The real MoE method classes rebuild themselves onto
        # FusedMoEMethodBase on first instantiation (custom __new__), so the
        # returned object is never isinstance of the original class. Identity
        # is pinned via class name + the very __init__ function object.
        self.assertEqual(type(method).__name__, expected_cls.__name__)
        self.assertIs(
            type(method).__dict__["__init__"], expected_cls.__dict__["__init__"]
        )

    def test_moe_ignore_and_backend_selection(self):
        cfg = self.mod.SlimQuantW4A8Int8MarlinConfig(ignore=list(KIMI_K31_HCU_IGNORE))
        FusedMoE = self._stubs["fused_moe"].FusedMoE
        UnquantMoe = self._stubs["unquant"].UnquantizedFusedMoEMethod
        Marlin = self.mod.SlimQuantW4A8Int8MarlinMoEMethod
        Triton = self.mod.SlimQuantW4A8Int8TritonMoEMethod
        Aiter = self.mod.SlimQuantW4A8Int8AiterMoEMethod

        kimi_moe = FusedMoE()
        kimi_moe.hidden_size = KIMI_K31_HIDDEN
        kimi_moe.intermediate_size_per_partition = KIMI_K31_MOE_INTER // 4  # TP4

        # The real method classes wire lightop/aiter kernels in __init__; the
        # selection logic under test only needs the class identity, so defuse
        # their constructors for the offline run.
        with ExitStack() as stack:
            for method_cls in (Marlin, Triton, Aiter):
                stack.enter_context(patch.object(method_cls, "__init__", lambda self, cfg: None))

            # auto -> lightop marlin when the backend is available and
            # Kimi-K3.1 TP shapes satisfy the repack tiles
            # (2*768 % 64 == 0, 768 % 32 == 0).
            with patch.object(self.mod, "_ensure_lightop_w4a8_marlin_available", lambda: None):
                self.assertIsMethod(
                    cfg.get_quant_method(kimi_moe, "model.layers.0.mlp.experts"), Marlin
                )

            # Kimi ignore list never matches the routed-expert MoE module; an
            # explicit ignore entry keeps those experts BF16.
            ignored = self.mod.SlimQuantW4A8Int8MarlinConfig(ignore=["mlp.experts"])
            self.assertIsInstance(
                ignored.get_quant_method(kimi_moe, "model.layers.0.mlp.experts"), UnquantMoe
            )

            envs_triton = SimpleNamespace(SGLANG_W4A8_TPMOE_BACKEND=_EnvVar("triton"))
            with patch.object(self.mod, "envs", envs_triton), \
                    patch.object(self.mod, "_ensure_lightop_w4a8_triton_available", lambda: None):
                self.assertIsMethod(
                    cfg.get_quant_method(kimi_moe, "model.layers.1.mlp.experts"), Triton
                )
            envs_aiter = SimpleNamespace(SGLANG_W4A8_TPMOE_BACKEND=_EnvVar("aiter"))
            with patch.object(self.mod, "envs", envs_aiter), \
                    patch.object(self.mod, "_ensure_aiter_w4a8_marlin_available", lambda: None):
                self.assertIsMethod(
                    cfg.get_quant_method(kimi_moe, "model.layers.1.mlp.experts"), Aiter
                )

            # Incompatible TP shapes (Qwen inter=640 TP=8 -> 2*80 % 64 != 0)
            # fall back from lightop marlin to the Triton backend.
            cramped = FusedMoE()
            cramped.hidden_size = 2048
            cramped.intermediate_size_per_partition = 80
            with patch.object(self.mod, "_ensure_lightop_w4a8_marlin_available", lambda: None), \
                    patch.object(self.mod, "_ensure_lightop_w4a8_triton_available", lambda: None), \
                    patch.object(self.mod, "logger"):
                self.assertIsMethod(
                    cfg.get_quant_method(cramped, "model.layers.4.mlp.experts"), Triton
                )

    # -- pure helpers --------------------------------------------------------

    def test_marlin_repack_tile_constraints_for_kimi_k31_tp_shards(self):
        ok = self.mod._marlin_w4a8_tp_shapes_ok
        self.assertTrue(ok(KIMI_K31_HIDDEN, KIMI_K31_MOE_INTER // 4))  # TP4: 1536/768
        self.assertTrue(ok(KIMI_K31_HIDDEN, KIMI_K31_MOE_INTER // 8))  # TP8: 768/384
        self.assertFalse(ok(2048, 80))   # Qwen inter=640 TP8: w13 N=160
        self.assertFalse(ok(7168, 40))   # K%32 violated
        self.assertFalse(ok(7167, 768))  # N%64 violated

    def test_w8a8_glob_and_language_model_flattening(self):
        match = self.mod._w8a8_glob_match
        self.assertTrue(match("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj"))
        self.assertTrue(match("model.layers.0.mlp.gate_proj", "model.layers.*.mlp.gate_proj"))
        # '*' matches exactly one dotted component
        self.assertFalse(match("model.layers.0.mlp.experts.gate_proj", "model.layers.*.mlp.gate_proj"))
        # checkpoint language_model form matches sglang's flattened prefix
        self.assertTrue(match(
            "model.layers.0.mlp.gate_proj",
            "model.language_model.layers.*.mlp.gate_proj"))
        self.assertTrue(match("model.layers.9.mlp.up_proj", "re:.*mlp\\.(up|gate)_proj"))

    def test_layer_in_w8a8_include_requires_all_fused_shards(self):
        include = self.mod._layer_in_w8a8_include
        # Patterns are anchored full dotted names (checkpoint form, '*' = one
        # component); bare suffixes do not match.
        self.assertFalse(include(None, ["model.layers.0.mlp.down_proj"]))
        self.assertFalse(include("model.layers.0.mlp.down_proj", []))
        self.assertFalse(include("model.layers.0.mlp.down_proj", ["mlp.down_proj"]))
        self.assertTrue(include(
            "model.layers.0.mlp.down_proj", ["model.layers.*.mlp.down_proj"]))
        mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        self.assertTrue(include(
            "model.layers.0.mlp.gate_up_proj",
            ["model.layers.*.mlp.gate_proj", "model.layers.*.mlp.up_proj"],
            mapping))
        self.assertFalse(include(
            "model.layers.0.mlp.gate_up_proj",
            ["model.layers.*.mlp.gate_proj"],
            mapping))


if __name__ == "__main__":
    unittest.main()
