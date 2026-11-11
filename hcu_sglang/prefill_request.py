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
from sglang.srt.managers import schedule_batch as N
from sglang.srt.mem_cache.storage_prefetch import StagedPrefetchPlan as _LegacyStagedPrefetchPlan
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from hcu_sglang.utils.detection import is_hcu


_LEGACY_PLAN = [_LegacyStagedPrefetchPlan]


class Req(N.Req):
    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: N.array[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: N.Optional[DllmConfig] = None,
        token_ids_logprob: N.List[int] = None,
        return_sampling_mask: bool = False,
        sampling_logprobs_mode: SamplingLogprobsMode = "selected",
        return_flat_raw_top_logprobs: bool = False,
        stream: bool = False,
        origin_input_ids_unpadded: N.Optional[N.array[int]] = None,
        lora_id: N.Optional[str] = None,
        input_embeds: N.Optional[N.List[N.List[float]]] = None,
        positional_embed_overrides: N.Optional[PositionalEmbeds] = None,
        token_type_ids: N.List[int] = None,
        session: N.Optional[Session] = None,
        custom_logit_processor: N.Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: ReturnHiddenStatesMode = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        return_indexer_topk: bool = False,
        eos_token_ids: N.Optional[Set[int]] = None,
        bootstrap_host: N.Optional[str] = None,
        bootstrap_port: N.Optional[int] = None,
        bootstrap_room: N.Optional[int] = None,
        disagg_mode: N.Optional[DisaggregationMode] = None,
        routed_dp_rank: N.Optional[int] = None,
        disagg_prefill_dp_rank: N.Optional[int] = None,
        kv_hints: N.Optional[N.KvHintsEnvelope] = None,
        vocab_size: N.Optional[int] = None,
        priority: N.Optional[int] = None,
        metrics_collector: N.Optional[SchedulerMetricsCollector] = None,
        extra_key: N.Optional[str] = None,
        routing_key: N.Optional[str] = None,
        dimensions: N.Optional[int] = None,
        http_worker_ipc: N.Optional[str] = None,
        time_stats: N.Optional[
            N.Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: N.Optional[N.List[int]] = None,
        token_indices_to_pool: N.Optional[N.List[int]] = None,
        decision_layout: N.Optional[N.List[int]] = None,
        session_id: N.Optional[str] = None,
        cache_salt: N.Optional[str] = None,
    ):
        # Input and output info
        self.rid = rid
        self.cache_request_handle = N.CacheRequestHandle(rid=rid, attempt_id=0)
        self.origin_input_ids = origin_input_ids
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else self.origin_input_ids
        )  # Before image padding
        # Each decode stage's output ids. Append-only by contract:
        # refresh_fill_ids infers how many output tokens are already in
        # full_untruncated_fill_ids from lengths alone, so in-place rewrites
        # that preserve length would silently corrupt fill_ids.
        self.output_ids = N.array("q")
        # Full untruncated sequence: origin + output (+ DLLM mask block).
        # Kept in sync by refresh_fill_ids; admission only updates
        # extend_end, never mutates this array's length.
        self.full_untruncated_fill_ids = N.array("q")
        # This round extends fill ids [prefix_len, extend_end).
        self.extend_end: N.Optional[int] = None
        self.dllm_initialized: bool = False

        self.session = session
        # media added by this turn, excluding the session's borrowed history
        self._session_mm_items: N.List[N.MultimodalDataItem] = []
        self.session_id = session_id
        # Used by the session radix cache to reject registration after a close/reopen.
        self.session_generation: N.Optional[int] = None
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices
        self.token_indices_to_pool = token_indices_to_pool
        self.decision_layout = decision_layout

        # For req-level memory management
        self.kv = N.ReqKvInfo()

        # Full-KV-derived boundary whose SWA window should be inserted after
        # the current prefill pass.
        self.swa_branching_seqlen: N.Optional[int] = None

        # for cross-encoder model
        self.token_type_ids = token_type_ids

        # The index of the extend / decode batch
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # For multi-http worker
        self.http_worker_ipc = http_worker_ipc

        # Require reasoning for the request
        self.require_reasoning = require_reasoning

        # State indicating whether the reasoning phase has finished (only meaningful when require_reasoning is True)
        self._is_reasoning_over = False
        self.reasoning_tokens = 0
        self._think_end_matcher: N.Optional[N.TokenSequenceMatcher] = None
        self._think_end_match_len = 0

        # Only custom logit processors need the "__req__" reference cycle.
        if custom_logit_processor is not None and isinstance(
            sampling_params.custom_params, dict
        ):
            sampling_params = N.copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states
        self.return_hidden_states_mode = N.get_return_hidden_states_mode(
            return_hidden_states
        )

        # Extra key for caller-defined request classification.
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.cache_salt = cache_salt or None
        self.lora_id = lora_id
        self.routing_key = routing_key

        # Lazy extra buffer: skip radix cache insert when prealloc failed at
        # boundary — the forward overwrites the only slot, corrupting the state.
        self.mamba_lazy_checkpoint: bool = True

        # Check finish
        self.tokenizer = None
        self.finished_reason: N.Optional[N.BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: N.Optional[N.BaseFinishReason] = None

        # External KV abort cleanup state. Each action is independently
        # retryable so a request can move between scheduler queues without
        # losing progress when an asynchronous cleanup step fails.
        self.external_kv_abort_requested = False
        self.external_kv_pending_chunk_cleared = False
        self.external_kv_linker_released = False
        self.external_kv_sender_abort_requested = False
        self.external_kv_metadata_released = False
        self.external_kv_cache_released = False
        self.external_kv_finish_state_applied = False
        self.external_kv_response_sent = False
        self.external_kv_abort_response_via_chunked = False
        self.external_kv_cleanup_done = False
        self.external_kv_cleanup_steps = {}
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # For multimodal inputs
        self.multimodal_inputs: N.Optional[N.MultimodalInputs] = None
        # Pre-computed multimodal prompt token counts; populated on the prefill
        # node and transferred to decode via the metadata buffer in disagg (PD) mode.
        self.mm_image_tokens: int = 0
        self.mm_audio_tokens: int = 0
        self.mm_video_tokens: int = 0
        self.mm_token_counts_ready: bool = False

        # Prefix info
        # Tokens [0, prefix_len) are already in the row, or, before allocation,
        # on the path to last_node; the next extend starts at prefix_len.
        self.prefix_len: int = 0
        # TODO(ispobock): rename to last_device_node
        self.last_node: N.Any = None
        self.last_host_node: N.Any = None
        self.best_match_node: N.Any = None
        # Per-component host hit lengths split off from host_hit_length:
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        # The branching point seqlen to track mamba state. If set, given by prefix
        # match, it will be the tracked seqlen in the ping pong buffer for the
        # right prefill pass.
        self.mamba_branching_seqlen: N.Optional[int] = None
        # Total cached prefix length (on-device prefix_len + host_hit_length),
        # capped at the max allowed prefix. Set during prefix matching at schedule
        # time and used to estimate uncached tokens / sort by longest prefix for
        # load reporting.
        self.num_matched_prefix_tokens = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0
        self.storage_hit_start: N.Optional[int] = None
        # FULL host-hit tokens actually spliced to device by init_load_back at
        # admission; less than host_hit_length when the load-back was declined
        # or the staged splice dropped (that shortfall is re-prefilled).
        self.host_loaded_length = 0
        # Absolute external-cache hit boundary shared by PP0; None outside PP.
        self.external_cache_hit_length: N.Optional[int] = None
        # Buffer-mode host memory is transport staging, not an L2 cache tier.
        self.host_hit_is_storage = False
        self.storage_prefetch_retry_attempts = 0
        self.staged_prefetch_plan: N.Optional[StagedPrefetchPlan] = None
        # The tree lock this request holds; None while it runs on a session
        # slot's record or holds no lock.
        self.lock: N.Optional[N.TreeLock] = None
        # Device/host prefix used to plan the latest L3 lookup. Admission uses
        # it to detect newly exposed storage demand after queue-time eviction.
        self.storage_prefetch_last_match_len: N.Optional[int] = None
        # Logical-page KV sharding: rotation base of the chain this request
        # extends (owner of position-page P is (base + P) % shard_size).
        # Refreshed at every sharded alloc — read through last_node, or drawn
        # least-full for a new chain — and consumed by the radix insert to
        # stamp new tree nodes. Allocation itself must NOT read it back when
        # a tree node is available (the checkpoint dedup rebind
        # would make it stale); the only allocation-time reader is the
        # radix-disabled fallback, which keeps no nodes past the root and has
        # no rebind.
        self.kv_rotation_base: N.Optional[int] = None

        # 1be97a0: launched steps whose results are not processed yet,
        # read by ScheduleBatch.add_inflight_step / step accounting.
        self.inflight_steps = 0

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.inflight_middle_chunks = 0

        # For retraction
        self.is_retracted = False
        # Indicates if the req has ever been retracted.
        self.retracted_stain = False

        self.weight_version_events: N.List[N.WeightVersionEvent] = []

        # 1be97a0-only Req fields (the fork's __init__ predates them; the
        # native readers below crash on absence — e.g. the output streamer's
        # detokenize header handshake).
        self._think_start_match_len = 0
        self.page_guard_state = None
        self.uocr_doc = None
        self._user_cache_salt = getattr(self, "cache_salt", None)
        self.exit_via = None
        self.send_via = None
        self.defer_output = False
        self.matched_window_host_hit_lengths = {}
        self.prefill_weight_versions = None
        self.detokenize_header_sent = False
        self.request_stats_retraction_count = None
        self.linker_offload_skip_prefix_len = 0

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.logprob = N.ReqLogprob(
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False
        self.return_sampling_mask = return_sampling_mask
        self.sampling_logprobs_mode = sampling_logprobs_mode
        self.return_flat_raw_top_logprobs = return_flat_raw_top_logprobs

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: N.Optional[N.List[N.Tuple[int]]] = None
        self.temp_input_top_logprobs_val: N.Optional[N.List[N.torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: N.Optional[N.List[int]] = None
        self.temp_input_token_ids_logprobs_val: N.Optional[N.List[float]] = None
        self.temp_input_token_ids_logprobs_idx: N.Optional[N.List[int]] = None

        if return_logprob:
            # shape: (bs, 1)
            self.logprob.output_token_logprobs_val = []
            self.logprob.output_token_logprobs_idx = []
            # shape: (bs, k)
            self.logprob.output_top_logprobs_val = []
            self.logprob.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            self.logprob.output_token_ids_logprobs_val = []
            self.logprob.output_token_ids_logprobs_idx = []
        self.sampling_mask_rows: N.Optional[N.SamplingMaskRows] = (
            N.SamplingMaskRows() if return_sampling_mask else None
        )
        self.hidden_states: N.List[N.List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        self.output_topk_p = None
        self.output_topk_index = None
        self.output_dsa_topk_indices = None

        # capture routed experts
        self.return_routed_experts = return_routed_experts
        self.routed_experts_start_len = routed_experts_start_len
        self.routed_experts: N.Optional[N.torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, topk)
        )

        self.return_indexer_topk = return_indexer_topk
        self.indexer_topk: N.Optional[N.torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, num_indexer_layers, index_topk)
        )
        # Customized info
        self.customized_info: N.Optional[N.Dict[str, N.List[N.Any]]] = None

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar_key: N.Optional[N.Tuple[str, str]] = None
        self.grammar: N.Optional[N.Union[N.BaseGrammarObject, N.Future[N.BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0
        # GLM compile may overlap PD bootstrap/KV transfer, but must finish
        # before this request enters a prebuilt batch.

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # Detailed breakdown of cached tokens by source (for HiCache)
        self.cached_tokens_device = 0  # Tokens from device cache (GPU)
        self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)
        self.cached_tokens_storage = 0  # Tokens from L3 storage backend
        self._cache_breakdown_computed = (
            False  # Track if breakdown was already computed
        )

        # Per-request count of verification forward passes.
        self.spec_verify_ct = 0

        # Per-request count of accepted draft tokens (excludes the bonus token).
        self.spec_num_correct_drafts = 0

        self.spec_num_block_accept_tokens = 0

        self.spec_num_cap_tokens = 0

        # Acceptance histogram for speculative decoding.
        # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
        # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
        self.spec_correct_drafts_histogram: N.List[int] = []

        self.spec_cap_lens_histogram: N.List[int] = []

        # The number of times this request has been retracted / preempted.
        self.retraction_count = 0
        self.retraction_mb_id = None

        # For observability
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = N.SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = N.SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False

        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: N.Optional[int] = bootstrap_port
        self.bootstrap_room: N.Optional[int] = bootstrap_room
        # Decode-local: the already-emitted boundary token to replay when a
        # retracted request is rebootstrapped. Set in pause_generation(retract)
        # and consumed in the decode transfer commit; never plumbed to prefill.
        self.pd_rebootstrap_forced_output_id: N.Optional[int] = None
        # Override embeds give KV that the token ids alone do not determine.
        self.skip_radix_cache_insert = (
            bootstrap_host == N.FAKE_BOOTSTRAP_HOST
            or positional_embed_overrides is not None
        )
        # Set when the request's in-flight forward ran on KV that never landed:
        # that forward's output is dropped and the request finishes with it.
        self.discard_output_reason: N.Optional[N.FINISH_ABORT] = None
        self.disagg_kv_sender: N.Optional[N.BaseKVSender] = None

        self.routed_dp_rank: N.Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: N.Optional[int] = disagg_prefill_dp_rank
        # Orchestrator KV hints, forwarded to the HiCache storage backends.
        self.kv_hints: N.Optional[N.KvHintsEnvelope] = kv_hints

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:req.extend_end])
        # start_send_idx = req.extend_end
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        # Decode-side cached-prefix length; base of the staging chunk grid
        # (start_send_idx starts here but advances with every send).
        # None until bootstrap pops it off the sender: the prefill bootstrap
        # treats None as "not resolved yet", so a 0 here silently pins every
        # request's decode prefix to 0.
        self.disagg_decode_prefix_len: N.Optional[int] = None
        # At-rest device-resident prefix end, snapshotted on the request's
        # first prefill batch; the cached-prefix early-send never goes past it.
        self.early_send_prefix_end: N.Optional[int] = None
        # Upper bound on radix prefix reuse for DSpark PD-hidden requests.
        # Set at bootstrap to the decode-committed prefix; see
        # _compute_max_prefix_len.
        self.pd_hidden_max_prefix_len: N.Optional[int] = None
        self.metadata_buffer_index: int = -1
        self.expected_kv_checksum: int = 0
        # Used in overlap sequence to signal that an optimistic request should
        # abort chunking. Set in create_sender, consumed in process_batch_result.
        self.pending_bootstrap = False
        # Number of optimistic prefill forward passes started. preserved across retracts.
        self.prefill_attempt_count = 0

        # For Matryoshka embeddings
        self.dimensions = dimensions

        # Beam search overlay: leader and internal members share one BeamGroup.
        self.beam_group = None

        # Whether to return pooled hidden states (pre-head transformer output)
        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None

        # For diffusion LLM
        self.init_diffusion_llm(dllm_config)

        # For hisparse
        self.hisparse_staging = False

        # Snapshot of the scheduler prefill-token counter taken at waiting_queue entry; used by HRRN aging.
        self.arrival_processed_tokens: int = 0




    def get_mm_token_counts(self) -> Tuple[int, int, int]:
        """Return per-modality prompt counts, scanning media at most once."""
        if not self.mm_token_counts_ready and self.multimodal_inputs is not None:
            (
                self.mm_image_tokens,
                self.mm_audio_tokens,
                self.mm_video_tokens,
            ) = self.multimodal_inputs.compute_mm_token_counts()
            self.mm_token_counts_ready = True
        return self.mm_image_tokens, self.mm_audio_tokens, self.mm_video_tokens




    def inherit_mm_token_counts(self, other: Req) -> None:
        """Reuse counts from the request whose media history is shared."""
        (
            self.mm_image_tokens,
            self.mm_audio_tokens,
            self.mm_video_tokens,
        ) = other.get_mm_token_counts()
        self.mm_token_counts_ready = True




    def extend_image_inputs(self, image_inputs):
        if self.mm_token_counts_ready:
            image_tokens, audio_tokens, video_tokens = (
                image_inputs.compute_mm_token_counts()
            )
            self.mm_image_tokens += image_tokens
            self.mm_audio_tokens += audio_tokens
            self.mm_video_tokens += video_tokens
        if self.session is not None:
            self._extend_session_image_inputs(image_inputs)
        elif self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)




    def init_next_round_input(
        self,
        tree_cache: N.Optional[BasePrefixCache] = None,
        cow_mamba: N.Optional[bool] = None,
    ):
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self.refresh_fill_ids()

        input_len = len(self.full_untruncated_fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            N.logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1

        # Pass the full array with a raw-token cap (limit) instead of slicing,
        # avoiding an O(context) copy per prefill-batch build.
        token_ids_to_match = self.full_untruncated_fill_ids
        key_limit: N.Optional[int] = self._compute_max_prefix_len(input_len)

        # PP0 owns the canonical external-cache boundary. Apply it only while
        # admitting the request; cache insertion performs its own rematch and
        # must be allowed to see KV computed after that original boundary.
        if (
            tree_cache is not None
            and self.external_cache_hit_length is not None
            and getattr(tree_cache, "linker", None) is not None
            and getattr(tree_cache, "pp_size", 1) > 1
        ):
            capped = min(input_len, self.external_cache_hit_length)
            key_limit = capped if key_limit is None else min(key_limit, capped)

        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        if self.positional_embed_overrides is not None:
            token_ids_to_match = N.array("q")
            key_limit = None

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            match_result = N.match_kv_cache(
                self,
                tree_cache,
                token_ids_to_match,
                cow_mamba=cow_mamba,
                max_prefix_len=key_limit,
            )
            self.mamba_branching_seqlen = match_result.mamba_branching_seqlen
            if match_result.cache_protected_len is None:
                self.kv.cache_protected_len = self.prefix_len

            if self.is_dllm():
                self._update_block_offset_for_dllm()

        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )




    def _compute_max_prefix_len(self, input_len: int) -> int:
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if (
            self.dllm_config is not None
            and self.dllm_config.requires_separate_context_encoding
        ):
            max_prefix_len = min(max_prefix_len, self.dllm_block_offset)
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        # DSpark PD ships hidden states for [decode prefix, end). A reused
        # prefix is never forwarded, so it yields no hidden rows; matching past
        # the decode-committed prefix would leave holes that abort the transfer.
        if self.pd_hidden_max_prefix_len is not None:
            max_prefix_len = min(max_prefix_len, self.pd_hidden_max_prefix_len)
        return max(max_prefix_len, 0)




    @property
    def auxiliary_swa_host_hit_length(self) -> int:
        plan = self.staged_prefetch_plan
        if isinstance(plan, _LEGACY_PLAN[0]):
            return 0
        return (
            plan.window_tokens(ComponentType.AUXILIARY_SWA) if plan is not None else 0
        )

    def needs_host_load_back(self) -> bool:
        """Whether any cache layer has a host hit that needs L2 H2D load_back."""
        return (
            self.host_hit_length > 0
            or self.swa_host_hit_length > 0
            or self.mamba_host_hit_length > 0
            or self.auxiliary_swa_host_hit_length > 0
        )


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.schedule_batch.Req", Req, HookType.REPLACE)
