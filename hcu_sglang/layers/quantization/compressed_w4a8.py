# Copyright 2026 Hygon Information Technology Co., Ltd.
#
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

from __future__ import annotations

import logging
from functools import partial
import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.moe.utils import get_deepep_mode
try:
    from sglang.srt.layers.utils.common import set_weight_attrs
except ImportError:
    from sglang.srt.utils import set_weight_attrs
from hcu_sglang.layers.quantization.w4a8_utils import w4a8_weight_repack_impl

logger = logging.getLogger(__name__)


def _load_transposed_checkpoint_weight(
    original_loader, param, loaded_weight, *args, **kwargs
):
    return original_loader(param, loaded_weight.t().contiguous(), *args, **kwargs)


class HCUW4A8Int8DeepEPMoEMethod(FusedMoEMethodBase):
    # DeepEPMoE selects this method independently of the global CT config,
    # which still resolves mixed W8A8 Linear layers.
    hcu_w4a8_hipc = True

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Retain checkpoint names and CT loader partitioning until post-load.
        # No actorder indices or zero points are needed for this qualified path.
        extra_weight_attrs.update(is_transposed=True, quant_method="channel")
        if "weight_loader" in extra_weight_attrs:
            extra_weight_attrs["weight_loader"] = partial(
                _load_transposed_checkpoint_weight, extra_weight_attrs["weight_loader"]
            )
        shapes = {
            "w13_weight_packed": (
                num_experts,
                hidden_size // 8,
                2 * intermediate_size_per_partition,
            ),
            "w2_weight_packed": (
                num_experts,
                intermediate_size_per_partition // 8,
                hidden_size,
            ),
            "w13_weight_scale": (num_experts, 1, 2 * intermediate_size_per_partition),
            "w2_weight_scale": (num_experts, 1, hidden_size),
            "w13_weight_shape": (num_experts, 2),
            "w2_weight_shape": (num_experts, 2),
        }
        for name, shape in shapes.items():
            dtype = params_dtype if name.endswith("scale") else torch.int32
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def create_moe_runner(self, layer, moe_runner_config):
        if moe_runner_config.activation != "silu" or not moe_runner_config.is_gated:
            raise ValueError("HCU W4A8 HIPC supports gated SiLU only")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError("HCU W4A8 HIPC applies router weights on output only")
        self.moe_runner_config = moe_runner_config
        self.runner = None  # DeepEPMoE owns contiguous/masked compute.

    @staticmethod
    def _convert_packed_weight(weight):
        weight = weight.transpose(1, 2).contiguous().view(torch.uint8)
        # CT stores offset-8, low nibble first; HIPC reads signed, high first.
        high = weight >> 4
        weight <<= 4
        weight |= high
        weight ^= 0x88
        return weight.view(torch.int8)

    @staticmethod
    def _channel_scale(scale, *, compact=False):
        scale = scale.transpose(1, 2).squeeze(-1).contiguous().float().unsqueeze(-1)
        # CT has true scales; the reference SlimQuant checkpoint has scale/16.
        scale.div_(16)
        if compact:
            return scale
        # Preserve the tested HCU vector-read mapping guard.
        storage = torch.empty(
            scale.numel() + (2 * 1024 * 1024) // 4,
            dtype=torch.float32,
            device=scale.device,
        )
        storage[: scale.numel()].copy_(scale.reshape(-1))
        return storage[: scale.numel()].view_as(scale)

    def process_weights_after_loading(self, layer):
        import deepgemm
        from hcu_sglang.layers.moe.ct_w4a8 import (
            _use_w4a8_contiguous_hipc,
            _use_w4a8_masked_hipc,
            register_ops,
        )

        register_ops()

        from sglang.srt.configs.model_config import is_glm_moe_dsa
        from sglang.srt.layers.attention.dsa.utils import process_model_config

        compact_scales = is_glm_moe_dsa(process_model_config().hf_config)

        mode = get_deepep_mode()
        normal_layout = "hipc" if _use_w4a8_contiguous_hipc else "marlin"
        ll_layout = "hipc" if _use_w4a8_masked_hipc else "marlin_masked"
        layout = normal_layout if mode.enable_normal() else ll_layout
        layouts = {layout}
        if mode.enable_low_latency():
            layouts.add(ll_layout)
        if mode.enable_normal() and normal_layout == "marlin":
            if self.moe_runner_config.swiglu_limit is not None:
                raise RuntimeError(
                    "LightOp Marlin apply_ep does not support V4.1 swiglu_limit; "
                    "set SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1"
                )
        for name in (
            ()
            if "hipc" not in layouts
            else (
                "pack_w4a8_moe_hipc_weight",
                "m_grouped_w4a8_gemm_nt_contiguous_hipc",
                "m_grouped_w4a8_gemm_nt_masked_hipc",
            )
        ):
            if not hasattr(deepgemm, name):
                raise RuntimeError(f"HCU W4A8 DeepEP requires deepgemm.{name}")
        w13 = self._convert_packed_weight(layer.w13_weight_packed.data)
        w2 = self._convert_packed_weight(layer.w2_weight_packed.data)
        intermediate_size = w2.shape[-1] * 2
        padded_size = ((intermediate_size + 127) // 128) * 128
        layer.w4a8_intermediate_size = intermediate_size
        layer.w4a8_padded_intermediate_size = (
            padded_size if layout == "hipc" else intermediate_size
        )
        layer.w4a8_ll_padded_intermediate_size = (
            padded_size if ll_layout == "hipc" else intermediate_size
        )
        for prefix, weight in (("w13", w13), ("w2", w2)):
            packed_layouts = {}
            for target_layout in layouts:
                # HIPC packing mutates its input. Clone only when auto mode
                # genuinely requires different normal and low-latency layouts.
                current = weight.clone() if len(layouts) > 1 else weight
                if target_layout == "hipc":
                    if prefix == "w2" and padded_size != intermediate_size:
                        current = F.pad(
                            current, (0, (padded_size - intermediate_size) // 2)
                        )
                    packed = deepgemm.pack_w4a8_moe_hipc_weight(current)
                else:
                    packed = w4a8_weight_repack_impl(
                        current, use_deepep=target_layout == "marlin_masked"
                    )
                packed_layouts[target_layout] = packed
            layer.register_parameter(
                prefix + "_weight",
                torch.nn.Parameter(packed_layouts[layout], requires_grad=False),
            )
            if mode.is_auto() and ll_layout != layout:
                layer.register_parameter(
                    prefix + "_weight_low_latency",
                    torch.nn.Parameter(packed_layouts[ll_layout], requires_grad=False),
                )
            delattr(layer, prefix + "_weight_packed")
            scale_name = prefix + "_weight_scale"
            scale_data = getattr(layer, scale_name).data
            compact = (
                compact_scales
                and scale_data.ndim == 3
                and tuple(scale_data.shape) in ((16, 1, 4096), (16, 1, 6144))
                and layouts <= {"hipc", "marlin_masked"}
            )
            scale = self._channel_scale(scale_data, compact=compact)
            setattr(layer, scale_name, torch.nn.Parameter(scale, requires_grad=False))
        layer.dispatcher.set_quant_config(
            {
                "normal_dispatcher_output_dtype": "int8",
                "low_latency_dispatcher_output_dtype": "int8",
                "normal_expert_alignment": 256,
                "hcu_w4a8_int8_dispatch": True,
                "hcu_w4a8_deepep_ll_producer_event": True,
            }
        )
        logger.info_once(
            "HCU W4A8 reference DeepEP enabled: normal=INT8, low_latency=INT8; "
            f"contiguous_hipc={_use_w4a8_contiguous_hipc}, masked_hipc={_use_w4a8_masked_hipc}"
        )

    def apply(self, layer, dispatch_output):
        return layer.run_moe_core(dispatch_output)

    def apply_ep(self, **kwargs):
        # Keep the reference Marlin fallback without recreating the CT scheme.
        if self.moe_runner_config.swiglu_limit is not None:
            raise RuntimeError(
                "LightOp Marlin apply_ep does not support V4.1 swiglu_limit; "
                "set SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1"
            )
        from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import (
            SlimQuantW4A8Int8MarlinMoEMethod,
        )

        return SlimQuantW4A8Int8MarlinMoEMethod.apply_ep(self, **kwargs)


def around_get_quant_method(original, config, layer, prefix):
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import QuantizationStrategy, QuantizationType
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from hcu_sglang.utils.detection import is_hcu

    if not (
        is_hcu()
        and isinstance(layer, FusedMoE)
        and get_moe_runner_backend().is_deep_gemm()
        and get_moe_a2a_backend().is_deepep()
        and config.quant_format == CompressionFormat.pack_quantized.value
    ):
        return original(config, layer, prefix)
    config._add_fused_moe_to_target_scheme_map()
    try:
        schemes = [
            config.get_scheme_dict(layer, prefix + suffix)
            for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
    except ValueError:
        return original(config, layer, prefix)
    scheme = schemes[0]
    if not all(item == scheme for item in schemes):
        raise ValueError(
            "All MoE projections need to have the same quantization scheme"
        )
    if scheme is not None:
        weight = scheme.get("weights")
        activation = scheme.get("input_activations")
        if (
            weight is not None
            and activation is not None
            and weight.type == QuantizationType.INT
            and activation.type == QuantizationType.INT
            and config._is_dynamic_token_w4a8(weight, activation)
            and weight.strategy == QuantizationStrategy.CHANNEL
            and weight.symmetric
            and activation.symmetric
            and weight.actorder is None
        ):
            return HCUW4A8Int8DeepEPMoEMethod(config)
    return original(config, layer, prefix)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.CompressedTensorsConfig.get_quant_method",
        around_get_quant_method,
        HookType.AROUND,
    )
