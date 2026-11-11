from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator as NativeKVCacheConfigurator
from sglang.srt.runtime_context import get_parallel

from hcu_sglang.utils.detection import is_hcu

_scratch_source = ContextVar("hcu_dsa_layer_split_scratch_source", default=None)


def before_draft_alloc(self, *args, **kwargs):
    if not is_hcu():
        return
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from hcu_sglang.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool

    target_pool = self.target_worker.model_runner.token_to_kv_pool
    self.draft_runner.dsa_layer_split_scratch_source = None
    if (
        self.server_args.disaggregation_mode == DisaggregationMode.PREFILL.value
        and get_parallel().pp_size == 1
        and self.speculative_algorithm.is_eagle()
        and not self.speculative_algorithm.is_eagle3()
        and self.draft_runner.layer_info.num_effective_layers == 1
        and self.draft_runner.model_config.num_nextn_predict_layers == 1
        and isinstance(target_pool, LayerSplitDSATokenToKVPool)
    ):
        self.draft_runner.dsa_layer_split_scratch_source = target_pool


def around_init_configurator(original, self, *args, **kwargs):
    source = getattr(self, "dsa_layer_split_scratch_source", None)
    token = _scratch_source.set(source)
    try:
        return original(self, *args, **kwargs)
    finally:
        _scratch_source.reset(token)


@dataclass(slots=True, kw_only=True)
class HCUKVCacheConfigurator(NativeKVCacheConfigurator):
    dsa_layer_split_scratch_source: Any = field(default_factory=_scratch_source.get)

    def _build_dsa_kv_pool(
        self, *, max_total_num_tokens, max_running_requests, dsa_pool_class,
    ):
        source = self.dsa_layer_split_scratch_source
        if source is None:
            return super(HCUKVCacheConfigurator, self)._build_dsa_kv_pool(
                max_total_num_tokens=max_total_num_tokens,
                max_running_requests=max_running_requests,
                dsa_pool_class=dsa_pool_class,
            )
        from sglang.srt.configs.model_config import (
            dsa_layer_skips_topk, get_dsa_index_head_dim,
            get_dsa_index_kpool, get_dsa_index_kpool_compress,
        )
        from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info
        from sglang.srt.mem_cache.kv_cache_configurator import (
            _should_elide_dsa_index_k, calculate_mla_kv_cache_dim,
        )
        from sglang.srt.runtime_context import get_exec, max_speculative_num_draft_tokens
        from hcu_sglang.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool

        rank, size = get_glm_dsa_cp_layer_shard_info(self)
        if not self.is_draft_worker or rank is None or size <= 1:
            raise ValueError("Shared DSA scratch requires a layer-split draft pool")
        pool_kwargs = {}
        if self.loc_space_scale > 1:
            pool_kwargs["dcp_replicated"] = True
        if _should_elide_dsa_index_k(is_draft_worker=self.is_draft_worker):
            pool_kwargs["skip_topk_layers"] = [
                dsa_layer_skips_topk(self.model_config.hf_config, layer_id)
                for layer_id in range(self.layer_info.start_layer, self.layer_info.end_layer)
            ]
        return LayerSplitDSATokenToKVPool(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config, kv_cache_dtype=self.kv_cache_dtype,
            ),
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
            index_kpool=get_dsa_index_kpool(self.model_config.hf_config),
            index_kpool_compress=get_dsa_index_kpool_compress(self.model_config.hf_config),
            tail_extra_slots=(max_speculative_num_draft_tokens() or 0),
            max_running_requests=max_running_requests,
            layer_shard_rank=rank,
            layer_shard_size=size,
            layer_shard_rank_offset=size - 1,
            layer_split_scratch_source=source,
            **pool_kwargs,
        )


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.alloc_memory_pool",
        before_draft_alloc, HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.init_kv_cache_configurator",
        around_init_configurator, HookType.AROUND,
    )
    for target in (
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator",
        "sglang.srt.model_executor.model_runner.KVCacheConfigurator",
    ):
        HookRegistry.register(target, HCUKVCacheConfigurator, HookType.REPLACE)
