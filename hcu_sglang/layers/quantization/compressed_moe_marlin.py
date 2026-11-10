from __future__ import annotations

import logging
from typing import List, Optional
import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn.parameter import Parameter
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from hcu_sglang.layers.moe.deepep_fp8 import get_deepgemm_marlin_weight_pack_fn
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
try:
    from sglang.srt.layers.utils.common import set_weight_attrs
except ImportError:
    from sglang.srt.utils import set_weight_attrs
from hcu_sglang.layers.quantization.lightop_int8_marlin import get_w8a8_int8_marlin_weights

logger = logging.getLogger(__name__)

class CompressedTensorsMarlinMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "SlimQuantCompressedTensorsMarlinConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> CompressedTensorsMarlinMoEMethod:
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        if quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MarlinMoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Slimquant_marlin does not support the FusedMoe scheme: {weight_quant}, {input_quant}"
            )

class CompressedTensorsW8A8Int8MarlinMoEMethod(CompressedTensorsMarlinMoEMethod):
    def __init__(
        self,
        quant_config: "CompressedTensorsMarlinConfig",  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations"
        )
        self.use_deepep = get_moe_a2a_backend().is_deepep()
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "use_int8_w8a8_deepgemm", False):
            from hcu_sglang.layers.quantization.hcu_deepgemm_w8a8_utils import (
                prepare_w8a8_int8_deepgemm_weights,
            )

            prepare_w8a8_int8_deepgemm_weights(layer)
            return

        if self.use_deepep:
            pack_fn = get_deepgemm_marlin_weight_pack_fn("int8")

            layer._dsv4_w13_weight_shape = tuple(layer.w13_weight.shape)
            layer._dsv4_w2_weight_shape = tuple(layer.w2_weight.shape)
            w13_weight = layer.w13_weight.contiguous()
            w13_packed = pack_fn(w13_weight)
            layer.w13_weight = Parameter(w13_packed, requires_grad=False)
            layer.register_buffer(
                "w13_weight_deepgemm", layer.w13_weight.data, persistent=False
            )
            del w13_packed
            torch.cuda.empty_cache()

            w2_weight = layer.w2_weight.contiguous()
            w2_packed = pack_fn(w2_weight)
            layer.w2_weight = Parameter(w2_packed, requires_grad=False)
            layer.register_buffer(
                "w2_weight_deepgemm", layer.w2_weight.data, persistent=False
            )
            del w2_packed
            torch.cuda.empty_cache()
            return

        w1_marlin_list = []
        for ii in range(layer.w13_weight.shape[0]):
            w1_marlin_in = get_w8a8_int8_marlin_weights(layer.w13_weight[ii])
            w1_marlin_list.append(w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)

        w2_marlin_list = []
        for ii in range(layer.w2_weight.shape[0]):
            w2_marlin_in = get_w8a8_int8_marlin_weights(layer.w2_weight[ii])
            w2_marlin_list.append(w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)

        layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_marlin, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    # def apply(
    #     self,
    #     layer: torch.nn.Module,
    #     x: torch.Tensor,
    #     router_logits: torch.Tensor,
    #     top_k: int,
    #     renormalize: bool,
    #     use_grouped_topk: bool = False,
    #     topk_group: Optional[int] = None,
    #     num_expert_group: Optional[int] = None,
    #     global_num_experts: int = -1,
    #     expert_map: Optional[torch.Tensor] = None,
    #     custom_routing_function: Optional[Callable] = None,
    #     scoring_func: str = "softmax",
    #     e_score_correction_bias: Optional[torch.Tensor] = None,
    #     apply_router_weight_on_input: bool = False,
    #     activation: str = "silu",
    #     enable_eplb: bool = False,
    #     use_nn_moe: Optional[bool] = False,
    #     routed_scaling_factor: Optional[float] = None,
    #     use_fused_gate: Optional[bool] = False,
    #     expert_load_view: Optional[torch.Tensor] = None,
    #     logical_to_physical_map: Optional[torch.Tensor] = None,
    #     logical_replica_count: Optional[torch.Tensor] = None,
    #     shared_output: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    #     if enable_eplb:
    #         raise NotImplementedError(
    #             "EPLB not supported for "
    #             "`CompressedTensorsW8A8Int8MoEMethod` yet.")

    #     topk_weights, topk_ids = FusedMoE.select_experts(
    #         hidden_states=x,
    #         router_logits=router_logits,
    #         use_grouped_topk=use_grouped_topk,
    #         top_k=top_k,
    #         renormalize=renormalize,
    #         topk_group=topk_group,
    #         num_expert_group=num_expert_group,
    #         custom_routing_function=custom_routing_function,
    #         scoring_func=scoring_func,
    #         routed_scaling_factor=routed_scaling_factor,
    #         use_fused_gate=use_fused_gate,
    #         e_score_correction_bias=e_score_correction_bias)

    #     return fused_experts_impl_int8_marlin(
    #         hidden_states=x,
    #         w1=layer.w13_weight,
    #         w2=layer.w2_weight,
    #         topk_weights=topk_weights,
    #         topk_ids=topk_ids,
    #         inplace=True,
    #         activation=activation,
    #         apply_router_weight_on_input=apply_router_weight_on_input,
    #         use_int8_w8a8=True,
    #         per_channel_quant=True,
    #         global_num_experts=global_num_experts,
    #         expert_map=expert_map,
    #         w1_scale=layer.w13_weight_scale,
    #         w2_scale=layer.w2_weight_scale,
    #         a1_scale=layer.w13_input_scale,
    #         a2_scale=layer.w2_input_scale,
    #         use_nn_moe=False,
    #         shared_output=shared_output,
    #         routed_scaling_factor=routed_scaling_factor)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
        # local_expert_mapping,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )

        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )

        output = torch.ops.sglang.fused_experts_impl_int8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.moe_runner_config.activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=float(routed_scaling_factor),
            # expert_map=local_expert_mapping,
        )

        return StandardCombineInput(hidden_states=output)

    def apply_with_shared_output(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: str = "silu",
        shared_output: Optional[torch.Tensor] = None,
        topk_output=None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )

        output = torch.ops.sglang.fused_experts_impl_int8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=float(routed_scaling_factor),
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s,
        )

        return output
