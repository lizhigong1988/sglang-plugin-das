from __future__ import annotations

import json

from sglang.srt.disaggregation.mooncake import conn as N
from hcu_sglang.disagg.common_utils import TransferKVChunk, FastQueue, pack_string_list, unpack_string_list
from hcu_sglang.disagg.protocol import StateType, TransferInfo, KVArgsRegisterInfo, resolve_state_component_dst_index
from hcu_sglang.disagg.transport_hidden_events import PDHiddenEventManager
from hcu_sglang.disagg.transport_source_drain import SourceDrainTracker
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import encode_v4_transfer_metadata, decode_v4_transfer_metadata
from hcu_sglang.utils.detection import is_hcu
try:
    from sglang.srt.utils import get_bool_env_var
except ImportError:
    from sglang.srt.utils.common import get_bool_env_var

try:
    from sglang.srt.disaggregation.mooncake.conn import zmq
except ImportError:
    from sglang.srt.disaggregation.common.conn import zmq
from sglang.srt.disaggregation.mooncake.conn import (ABORT_ACK_TAG, AbortNotification, DecodeStagingContext, Dict, DisaggregationMode, KVArgs, KVPoll, List, MooncakeKVManager, MooncakeRequestStage, NetworkAddress, Optional, PrefillStagingContext, ServerArgs, Set, TraceNullContext, TraceReqContext, Tuple, Union, _validate_efa_allocator_compatibility, check_mooncake_custom_mem_pool_enabled, concurrent, defaultdict, envs, get_observability, handle_staging_rsp, handle_watermark_msg, logger, mooncake_trace_func, np, npt, os, pack_int_lists, struct, threading, time,)
from hcu_sglang.disagg.utils import build_transfer_entry_pairs
_is_hcu = is_hcu()
# default mirrors hcu_sglang.environ (single source): official layout unless
# the HCU FA perf-A/B path is explicitly requested.
_kv_layout_hcu_fa = _is_hcu and get_bool_env_var("SGLANG_KV_LAYOUT_HCU_FA", default="false")

class MooncakeKVManagerMethods:
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        enable_custom_mem_pool, custom_mem_pool_type = (
            check_mooncake_custom_mem_pool_enabled()
        )
        _validate_efa_allocator_compatibility(
            enable_custom_mem_pool, custom_mem_pool_type
        )
        super(N.MooncakeKVManager, self).__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.init_engine()
        self.register_buffer_to_engine()
        self.pd_hidden_pool = None
        self.pd_hidden_events = PDHiddenEventManager(self)
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.max_transfer_batch_indices = (
            envs.SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES.get()
        )
        self.enable_trace = get_observability().enable_trace
        # Set by teardown() to make worker threads exit (P<->D role switch).
        self._stopped = False
        self._worker_threads: List[threading.Thread] = []
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            # Separate from failed_sessions, which _run_one_probe_pass clears
            # on a successful probe: an incompatible decode is still reachable,
            # so only a re-registration may lift this.
            self.state_layout_rejections: Dict[str, str] = {}
            # Peers already cleared at registration; skip the per-transfer check.
            self.state_strides_validated: Set[str] = set()
            self.session_lock = threading.Lock()
            # Per-room count of chunks not yet transferred; teardown waits for
            # zero so a deferred chunk is not dropped by an early conclude.
            self._staging_outstanding = defaultdict(int)
            self._source_drain = SourceDrainTracker()
            self.session_lock = threading.Lock()
            self._transfer_completion_condition = threading.Condition()
            self.pd_hidden_events.init_prefill_state()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = (
                envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
            )
            if transfer_thread_pool_size is None:
                transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            self.enable_custom_mem_pool, self.custom_mem_pool_type = (
                check_mooncake_custom_mem_pool_enabled()
            )
            self._staging_ctx = PrefillStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_buffers(len(self.transfer_queues))
            for i, (transfer_queue, executor) in enumerate(
                zip(self.transfer_queues, self.executors)
            ):
                # Track the thread so teardown() can join it: otherwise every
                # P->D->P flip that re-enters PREFILL leaks threads
                # (each parked forever in FastQueue.get()).
                t = threading.Thread(
                    target=self.transfer_worker,
                    args=(
                        transfer_queue,
                        executor,
                        (
                            self._staging_ctx.buffers[i]
                            if self.enable_staging and self._staging_ctx.buffers
                            else None
                        ),
                        i,
                    ),
                    daemon=True,
                )
                t.start()
                self._worker_threads.append(t)
            self.enable_failed_session_probe = (
                envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.get()
            )
            if self.enable_failed_session_probe:
                self.failed_session_probe_interval = (
                    envs.SGLANG_FAILED_SESSION_PROBE_INTERVAL_S.get()
                )
                self._failed_session_probe_shutdown = threading.Event()
                t = threading.Thread(
                    target=self._failed_session_probe_loop,
                    name="MooncakeFailedSessionProbe",
                    daemon=True,
                )
                t.start()
                self._worker_threads.append(t)
            self.start_prefill_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.pd_hidden_events.init_decode_state()
            self._staging_ctx = DecodeStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_allocator()
            self.start_decode_thread()

    def supports_pd_hidden_streaming(self) -> bool:
        return self.pd_hidden_events.supports_streaming()

    def mark_pd_hidden_request_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        self.pd_hidden_events.mark_request_done(
            bootstrap_room,
            state_indices,
        )

    def pop_pd_hidden_request_done(self, bootstrap_room: int) -> bool:
        return self.pd_hidden_events.pop_request_done(bootstrap_room)

    def mark_pd_hidden_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        self.mark_pd_hidden_request_done(bootstrap_room, state_indices)

    def pop_pd_hidden_done(self, bootstrap_room: int) -> bool:
        return self.pop_pd_hidden_request_done(bootstrap_room)

    def park_pd_hidden_chunk_for_ack(
        self,
        *,
        transfer_queue: FastQueue,
        kv_chunk: TransferKVChunk,
        prefill_rank: int,
        expected_count: int,
        timeout_s: float = 300.0,
    ) -> bool:
        return self.pd_hidden_events.park_chunk_for_ack(
            transfer_queue=transfer_queue,
            kv_chunk=kv_chunk,
            prefill_rank=prefill_rank,
            expected_count=expected_count,
            timeout_s=timeout_s,
        )

    def _wake_pd_hidden_ack_waiters(self, room: int) -> None:
        self.pd_hidden_events.wake_ack_waiters(room)

    def _park_pd_hidden_chunk_behind_room(
        self, transfer_queue: FastQueue, kv_chunk: TransferKVChunk
    ) -> None:
        self.pd_hidden_events.park_chunk_behind_room(transfer_queue, kv_chunk)

    def _wake_next_pd_hidden_room_waiter(self, room: int) -> None:
        self.pd_hidden_events.wake_next_room_waiter(room)

    def _handle_pd_hidden_chunk_ack(
        self, room: int, prefill_rank: int, hidden_start: int
    ) -> None:
        self.pd_hidden_events.handle_chunk_ack(room, prefill_rank, hidden_start)

    def pop_pd_hidden_ready_chunks(self, room: int) -> List[dict]:
        return self.pd_hidden_events.pop_ready_chunks(room)

    def submit_pd_hidden_chunk_ack(
        self,
        *,
        event,
        remote: str,
        dst_port: int,
        room: int,
        prefill_rank: int,
        hidden_start: int,
        is_last_hidden_chunk: bool,
    ) -> None:
        self.pd_hidden_events.submit_chunk_ack(
            event=event,
            remote=remote,
            dst_port=dst_port,
            room=room,
            prefill_rank=prefill_rank,
            hidden_start=hidden_start,
            is_last_hidden_chunk=is_last_hidden_chunk,
        )

    def _drain_pd_hidden_ack_completions(self) -> None:
        self.pd_hidden_events.drain_ack_completions()

    def pop_pd_hidden_acked_chunks(self, room: int) -> List[dict]:
        return self.pd_hidden_events.pop_acked_chunks(room)

    def wait_pd_hidden_ack_completions(
        self, room: int, timeout_s: float = 300.0
    ) -> bool:
        return self.pd_hidden_events.wait_ack_completions(room, timeout_s)

    def _begin_pd_hidden_transfer(self, room: int) -> None:
        self.pd_hidden_events.begin_transfer(room)

    def _end_pd_hidden_transfer(self, room: int) -> None:
        self.pd_hidden_events.end_transfer(room)

    def _wait_pd_hidden_transfers_quiesced(
        self, room: int, timeout_s: float = 300.0
    ) -> bool:
        return self.pd_hidden_events.wait_transfers_quiesced(room, timeout_s)

    def _pd_hidden_state_index(self) -> Optional[int]:
        return self.pd_hidden_events.state_index()

    def _has_pd_hidden_state(self, state_indices: Optional[List]) -> bool:
        return self.pd_hidden_events.has_state(state_indices)

    def _without_pd_hidden_state(self, state_indices: Optional[List]) -> Optional[List]:
        return self.pd_hidden_events.without_state(state_indices)

    def _pd_hidden_release_state_indices(
        self, kv_chunk: TransferKVChunk
    ) -> Optional[List]:
        return self.pd_hidden_events.release_state_indices(kv_chunk)

    def _free_pd_hidden_state_indices(self, state_indices: Optional[List]) -> None:
        self.pd_hidden_events.free_state_indices(state_indices)

    def _free_pd_hidden_chunk_rows(self, kv_chunk: TransferKVChunk) -> None:
        self.pd_hidden_events.free_chunk_rows(kv_chunk)

    def _release_or_mark_pd_hidden_done(self, kv_chunk: TransferKVChunk) -> None:
        self.pd_hidden_events.release_or_mark_done(kv_chunk)

    def _finish_pd_hidden_streaming_chunk(
        self,
        kv_chunk: TransferKVChunk,
        hidden_inflight_key: Optional[Tuple[int, int]],
    ) -> None:
        self.pd_hidden_events.finish_streaming_chunk(kv_chunk, hidden_inflight_key)

    def _mark_session_failed_and_sync(
        self,
        *,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
        prefill_unique_rank: int,
        failure_reason: str,
    ) -> None:
        self.pd_hidden_events.mark_session_failed_and_sync(
            kv_chunk=kv_chunk,
            req=req,
            prefill_unique_rank=prefill_unique_rank,
            failure_reason=failure_reason,
        )

    def notify_pd_hidden_chunk_ready(
        self,
        *,
        remote: str,
        dst_port: int,
        room: int,
        prefill_rank: int,
        hidden_start: int,
        row_len: int,
        is_last_hidden_chunk: bool,
        dst_indices: List[int],
    ) -> None:
        na = NetworkAddress(remote, dst_port)
        self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(
            [
                self.PD_HIDDEN_CHUNK_READY_HEADER,
                str(room).encode("ascii"),
                str(prefill_rank).encode("ascii"),
                str(hidden_start).encode("ascii"),
                str(row_len).encode("ascii"),
                b"1" if is_last_hidden_chunk else b"0",
                struct.pack(f"<{len(dst_indices)}i", *[int(x) for x in dst_indices])
                if dst_indices
                else b"",
                self.local_ip.encode("ascii"),
                str(self.rank_port).encode("ascii"),
            ]
        )

    def ack_pd_hidden_chunk(
        self,
        *,
        remote: str,
        dst_port: int,
        room: int,
        prefill_rank: int,
        hidden_start: int,
    ) -> None:
        na = NetworkAddress(remote, dst_port)
        self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(
            [
                self.PD_HIDDEN_CHUNK_ACK_HEADER,
                str(room).encode("ascii"),
                str(prefill_rank).encode("ascii"),
                str(hidden_start).encode("ascii"),
            ]
        )

    def _notify_transfer_waiters(self) -> None:
        condition = getattr(self, "_transfer_completion_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def update_status(self, bootstrap_room: int, status: KVPoll):
        super(N.MooncakeKVManager, self).update_status(bootstrap_room, status)
        if status in (KVPoll.Success, KVPoll.Failed):
            self._notify_transfer_waiters()

    def wait_for_transfer_rooms(
        self, bootstrap_rooms: Set[int], timeout_s: float
    ) -> bool:
        """Wait briefly for final rooms so the scheduler can poll before forward."""
        if not bootstrap_rooms or timeout_s <= 0:
            return False

        def all_terminal() -> bool:
            # Mirror MooncakeKVSender.poll(): Success is set before the worker
            # finishes the chunk, so it only counts once nothing is outstanding.
            for room in bootstrap_rooms:
                status = self.request_status.get(room)
                if status not in (None, KVPoll.Success, KVPoll.Failed):
                    return False
                if status == KVPoll.Success and (
                    self._staging_outstanding.get(room, 0) > 0
                    or (
                        self.defer_decode_allocation
                        and room in self._staging_outstanding
                    )
                ):
                    return False
            return True

        with self._transfer_completion_condition:
            return self._transfer_completion_condition.wait_for(
                all_terminal, timeout=timeout_s
            )

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
        )

        # Extract configuration
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * self.attn_tp_size

        dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        (
            src_head_start_offset,
            num_heads_to_send,
            dst_head_start_offset,
            _,
        ) = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            self.kv_args.engine_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        src_data_ptrs = self.kv_args.kv_data_ptrs
        src_layer_ids = self.kv_args.kv_layer_ids
        if src_layer_ids or dst_layer_ids:
            # Draft buffers break the flat [K block, V block] layout, so pair by
            # layer ID instead of the half-split used by get_mha_kv_ptrs_with_pp.
            if any(
                item_len != src_kv_item_len for item_len in self.kv_args.kv_item_lens
            ):
                logger.error(
                    f"[{mooncake_session_id}] head-sliced transfer assumes one item "
                    f"length for every KV entry, got {set(self.kv_args.kv_item_lens)}"
                )
                return -1
            layer_ptr_pairs = [
                (src_data_ptrs[i], dst_kv_ptrs[j])
                for i, j in build_transfer_entry_pairs(
                    src_layer_ids,
                    dst_layer_ids or [],
                    len(src_data_ptrs),
                    len(dst_kv_ptrs),
                    allow_positional_fallback=self.pp_size == 1,
                )
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_kv_ptrs)
            )
            layer_ptr_pairs = [
                (src_k_ptrs[i], dst_k_ptrs[i]) for i in range(layers_current_pp_stage)
            ] + [(src_v_ptrs[i], dst_v_ptrs[i]) for i in range(layers_current_pp_stage)]

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        prefill_page_indices = prefill_kv_indices.reshape(-1, 1).astype(np.int64)
        decode_page_indices = dst_kv_indices.reshape(-1, 1).astype(np.int64)
        tokens_per_page = np.arange(page_size, dtype=np.int64).reshape(1, -1)
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size
        kv_cache_layout = getattr(self, "kv_cache_layout", None)
        if kv_cache_layout is None:
            # Compatibility with older CommonKVManager versions that do not
            # propagate the pool layout through KVArgs yet.
            kv_cache_layout = "hnd" if envs.SGLANG_USE_HND_KVCACHE.get() else "nhd"
        is_hnd = kv_cache_layout == "hnd"
        is_hcu_legacy = (
            _kv_layout_hcu_fa and not is_hnd and not envs.SGLANG_USE_HND_KVCACHE.get()
        )
        if is_hnd:
            # HND/BHSD pages are laid out as [page, head, token, dim]. Heads
            # belonging to one token are therefore not contiguous: adjacent
            # heads are page_size * head_bytes apart. Build one contiguous
            # dim-slice per (page, head, token) instead of treating the page as
            # token-major, which would silently corrupt asymmetric-TP KV data.
            src_head_indices = np.arange(
                src_head_start_offset,
                src_head_start_offset + num_heads_to_send,
                dtype=np.int64,
            ).reshape(1, -1, 1)
            dst_head_indices = np.arange(
                dst_head_start_offset,
                dst_head_start_offset + num_heads_to_send,
                dtype=np.int64,
            ).reshape(1, -1, 1)
            token_offsets = np.arange(page_size, dtype=np.int64).reshape(1, 1, -1)
            src_token_slot_offsets = (
                src_head_indices * page_size + token_offsets
            ) * bytes_per_head_slice_to_send
            dst_token_slot_offsets = (
                dst_head_indices * page_size + token_offsets
            ) * bytes_per_head_slice_to_send
            transfer_slice_len = bytes_per_head_slice_to_send
        elif is_hcu_legacy:
            # The legacy HCU FA cache is head-major within each page:
            # K is [page, head, token, dim] and V is [page, head, dim, token].
            # Although K and V differ inside a head, one whole head remains a
            # contiguous page-sized byte range in both buffers. Transfer each
            # selected head as one slice so asymmetric TP preserves the exact
            # physical layout without assuming token-major NHD storage.
            src_head_indices = np.arange(
                src_head_start_offset,
                src_head_start_offset + num_heads_to_send,
                dtype=np.int64,
            ).reshape(1, -1)
            dst_head_indices = np.arange(
                dst_head_start_offset,
                dst_head_start_offset + num_heads_to_send,
                dtype=np.int64,
            ).reshape(1, -1)
            head_page_bytes = page_size * bytes_per_head_slice_to_send
            src_token_slot_offsets = src_head_indices * head_page_bytes
            dst_token_slot_offsets = dst_head_indices * head_page_bytes
            transfer_slice_len = head_page_bytes
        else:
            src_token_slot_offsets = (
                tokens_per_page * bytes_per_token_on_prefill + src_head_slice_offset
            )
            dst_token_slot_offsets = (
                tokens_per_page * bytes_per_token_on_decode + dst_head_slice_offset
            )
            transfer_slice_len = heads_bytes_per_token_to_send

        def process_layer_tp_aware(src_layer_ptr, dst_layer_ptr):
            src_page_base_addrs = src_layer_ptr + prefill_page_indices * src_kv_item_len
            dst_page_base_addrs = dst_layer_ptr + decode_page_indices * dst_kv_item_len
            src_slice_addrs = src_page_base_addrs + src_token_slot_offsets
            dst_slice_addrs = dst_page_base_addrs + dst_token_slot_offsets

            src_addr_list = src_slice_addrs.reshape(-1).tolist()
            if not src_addr_list:
                # Nothing to transfer for this layer.
                return 0
            dst_addr_list = dst_slice_addrs.reshape(-1).tolist()
            total_slices = len(src_addr_list)
            length_list = [transfer_slice_len] * total_slices
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = [
            executor.submit(process_layer_tp_aware, src_layer_ptr, dst_layer_ptr)
            for src_layer_ptr, dst_layer_ptr in layer_ptr_pairs
        ]

        return self._await_transfer_futures(futures)

    def _send_pd_hidden_packet(
        self,
        req: TransferInfo,
        prefill_state_indices: List,
        packet_idx: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info=None,
    ) -> Tuple[int, bool]:
        state_idx = self._pd_hidden_state_index()
        if state_idx is None or state_idx >= len(prefill_state_indices):
            return 0, True
        indices = prefill_state_indices[state_idx]
        if indices is None:
            return 0, True
        src_indices = np.asarray(indices, dtype=np.int32)
        if target_rank_registration_info is None:
            target_rank_registration_info = self.decode_kv_args_table[req.mooncake_session_id]
        dst_state_idx = resolve_state_component_dst_index(self.kv_args.state_types, target_rank_registration_info.dst_state_types, state_idx)
        dynamic_dst = (req.spec_metadata or {}).get("pp_slice", {}).get("dynamic_dst")
        if not dynamic_dst:
            if packet_idx > 0:
                return 0, True
            if dst_state_idx >= len(req.dst_state_indices):
                return 0, True
            dst_indices = np.asarray(req.dst_state_indices[dst_state_idx], dtype=np.int32)
            if len(src_indices) == 0 and len(dst_indices) == 0:
                return 0, True
            if (req.spec_metadata or {}).get("streaming_hidden"):
                dst_indices = dst_indices[: len(src_indices)]
            if len(src_indices) != len(dst_indices):
                raise RuntimeError(
                    "PD_HIDDEN state index length mismatch: "
                    f"room={req.room}, prefill={len(src_indices)}, "
                    f"dst={len(dst_indices)}"
                )
            dst_data_ptrs = (
                target_rank_registration_info.dst_state_data_ptrs[dst_state_idx]
                if dst_state_idx < len(target_rank_registration_info.dst_state_data_ptrs)
                else []
            )
            rc = self._send_kvcache_generic(
                mooncake_session_id=req.mooncake_session_id,
                src_data_ptrs=self.kv_args.state_data_ptrs[state_idx],
                dst_data_ptrs=dst_data_ptrs,
                item_lens=self.kv_args.state_item_lens[state_idx],
                prefill_data_indices=src_indices,
                dst_data_indices=dst_indices,
                executor=executor,
                state_type=StateType.PD_HIDDEN,
                force_flat=True,
            )
            return rc, True
        row_chunks = dynamic_dst.get("row_chunks") or []
        if packet_idx >= len(row_chunks):
            return 0, True

        row_chunk = row_chunks[packet_idx]
        row_start = int(row_chunk.get("row_start", 0))
        row_len = int(row_chunk.get("row_len", 0))
        if row_len <= 0:
            return 0, packet_idx + 1 >= len(row_chunks)
        row_end = row_start + row_len
        if row_start < 0 or row_end > len(src_indices):
            raise RuntimeError(
                "Invalid PD hidden packet: "
                f"room={req.room}, packet_idx={packet_idx}, "
                f"row_start={row_start}, row_len={row_len}, "
                f"row_count={len(src_indices)}"
            )

        src_data_ptrs = self.kv_args.state_data_ptrs[state_idx]
        dst_data_ptrs = [int(row_chunk.get("ptr", dynamic_dst.get("ptr", 0)))]
        item_lens = [int(dynamic_dst["item_len"])]
        rc = self._send_kvcache_generic(
            mooncake_session_id=req.mooncake_session_id,
            src_data_ptrs=src_data_ptrs,
            dst_data_ptrs=dst_data_ptrs,
            item_lens=item_lens,
            prefill_data_indices=src_indices[row_start:row_end],
            dst_data_indices=np.arange(row_len, dtype=np.int32),
            executor=executor,
            state_type=StateType.PD_HIDDEN,
                force_flat=True,
        )
        return rc, packet_idx + 1 >= len(row_chunks)

    def _handle_deferred_abort_notification(
        self, notification: AbortNotification, room_active: bool
    ) -> None:
        room = notification.room
        if room_active or self._staging_outstanding.get(room, 0) > 0:
            self.update_status(room, KVPoll.Failed)
            self._wake_pd_hidden_ack_waiters(room)

        ack_target = notification.deferred_ack_target()
        if ack_target is None:
            return
        self.register_deferred_ack_target(room, ack_target)
        self._maybe_ack_drained_abort(room)
        if room_active:
            logger.debug(
                "Received abort notification for room %s, marked as Failed; "
                "ACK deferred until transfer drains",
                room,
            )

    def _handle_legacy_abort_notification(
        self, notification: AbortNotification, room_active: bool
    ) -> None:
        room = notification.room
        if room_active:
            self.update_status(room, KVPoll.Failed)
            self._wake_pd_hidden_ack_waiters(room)
            logger.debug(
                "Received abort notification for room %s, marked as Failed", room
            )
        else:
            logger.debug(
                "Received abort notification for room %s, ignoring "
                "(already completed or unknown)",
                room,
            )

        self._wait_pd_hidden_transfers_quiesced(room)
        if notification.decode_ip is None or notification.decode_port is None:
            return
        try:
            na = NetworkAddress(notification.decode_ip, notification.decode_port)
            # Deferred ACKs carry a rank; the feature-off wire format does not.
            self._send_multipart_locked(
                na.to_tcp(),
                [ABORT_ACK_TAG, str(room).encode("ascii")],
                is_ipv6=na.is_ipv6,
            )
            logger.debug(
                "Sent ABORT_ACK for room %s to %s:%s",
                room,
                notification.decode_ip,
                notification.decode_port,
            )
        except Exception as exc:
            logger.debug("Failed to send ABORT_ACK for room %s: %s", room, exc)

    def start_prefill_thread(self):
        recv = self._make_worker_recv(self.server_socket)

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while not self._stopped:
                waiting_req_bytes = recv()
                if waiting_req_bytes is None:
                    continue
                if waiting_req_bytes[0] == MooncakeKVManager.PD_HIDDEN_CHUNK_ACK_HEADER:
                    room = int(waiting_req_bytes[1].decode("ascii"))
                    prefill_rank = int(waiting_req_bytes[2].decode("ascii"))
                    hidden_start = int(waiting_req_bytes[3].decode("ascii"))
                    self._handle_pd_hidden_chunk_ack(room, prefill_rank, hidden_start)
                    continue
                room = waiting_req_bytes[0].decode("ascii")
                # Staging: decode reports consumption watermark back to prefill
                if room == "WATERMARK":
                    handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
                    continue
                # Staging: decode replies with allocated staging offset
                if room == "STAGING_RSP":
                    handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
                    continue
                if self._handle_abort_notification(waiting_req_bytes):
                    continue
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    decode_kv_args = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    try:
                        from hcu_sglang.disagg.validation import validate_remote_peer_state_abis
                        validate_remote_peer_state_abis(self, decode_kv_args)
                    except RuntimeError as error:
                        decode_kv_args.registration_error = str(error)
                        logger.error(
                            "Decode peer %s registered an incompatible state ABI: %s",
                            mooncake_session_id,
                            error,
                        )
                    decode_kv_args.requires_dcp_relayout = self.requires_dcp_relayout(
                        decode_kv_args.dst_dcp_size,
                        decode_kv_args.dst_dcp_rank,
                    )
                    if decode_kv_args.requires_dcp_relayout:
                        num_entries = len(self.kv_args.kv_item_lens)
                        num_draft = self.kv_args.num_draft_entries
                        dst_item_lens: List[Optional[int]] = [
                            decode_kv_args.dst_kv_item_len
                        ] * (num_entries - num_draft) + [None] * num_draft
                        decode_kv_args.dcp_token_item_lens = (
                            self.prepare_dcp_token_item_lens(
                                dst_item_lens,
                                decode_kv_args.dst_dcp_size,
                            )
                        )
                        self._init_dcp_pack_buffers_once(
                            decode_kv_args.dst_dcp_size, include_draft=True
                        )
                    layout_mismatch = self._publish_peer_registration(
                        mooncake_session_id, decode_kv_args
                    )
                    if layout_mismatch is not None:
                        logger.error(
                            "Rejecting state transfers to decode session %s: %s",
                            mooncake_session_id,
                            layout_mismatch,
                        )
                    else:
                        logger.debug(
                            f"Register KVArgs from {mooncake_session_id} successfully"
                        )
                    continue
                else:
                    if len(waiting_req_bytes) < 8:
                        logger.warning(
                            "Ignoring malformed Mooncake bootstrap message: "
                            "room=%s frames=%d",
                            room,
                            len(waiting_req_bytes),
                        )
                        continue
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    transfer_info = TransferInfo.from_zmq(waiting_req_bytes)
                    self.transfer_infos[room][mooncake_session_id] = transfer_info
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.resolve_kv_replica_factor(self.transfer_infos[room])
                        self.req_to_decode_prefix_len[room] = next(
                            (
                                info.decode_prefix_len
                                for info in self.transfer_infos[room].values()
                                if info.decode_prefix_len is not None
                            ),
                            0,
                        )
                        pd_hidden_meta = next(
                            (
                                info.spec_metadata
                                for info in self.transfer_infos[room].values()
                                if info.spec_metadata
                                and info.spec_metadata.get("pd_hidden")
                            ),
                            None,
                        )
                        if pd_hidden_meta:
                            self.req_to_pd_hidden_meta[room] = pd_hidden_meta
                        self.update_status(room, KVPoll.WaitingForInput)

        t = threading.Thread(target=bootstrap_thread, daemon=True)
        t.start()
        self._worker_threads.append(t)

    def start_decode_thread(self):
        recv = self._make_worker_recv(self.server_socket)

        def decode_thread():
            poller = zmq.Poller()
            poller.register(self.server_socket, zmq.POLLIN)
            poller.register(self.pd_hidden_events.ack_wakeup_receiver, zmq.POLLIN)
            while not self._stopped:
                events = dict(
                    poller.poll(500 if self.server_args.enable_pd_role_switch else None)
                )
                if self.pd_hidden_events.ack_wakeup_receiver in events:
                    while True:
                        try:
                            self.pd_hidden_events.ack_wakeup_receiver.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                    self._drain_pd_hidden_ack_completions()
                if self.server_socket not in events:
                    continue
                msg = self.server_socket.recv_multipart()
                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue
                if msg[0] == MooncakeKVManager.PD_HIDDEN_CHUNK_READY_HEADER:
                    room = int(msg[1].decode("ascii"))
                    prefill_rank = int(msg[2].decode("ascii"))
                    hidden_start = int(msg[3].decode("ascii"))
                    row_len = int(msg[4].decode("ascii"))
                    is_last_hidden_chunk = msg[5] == b"1"
                    dst_indices = (
                        list(np.frombuffer(msg[6], dtype=np.int32).astype(np.int64))
                        if len(msg[6]) > 0
                        else []
                    )
                    ack_host = msg[7].decode("ascii")
                    ack_port = int(msg[8].decode("ascii"))
                    self.pd_hidden_events.append_ready_chunk(
                        room,
                        {
                            "room": room,
                            "prefill_rank": prefill_rank,
                            "hidden_start": hidden_start,
                            "row_len": row_len,
                            "is_last_hidden_chunk": is_last_hidden_chunk,
                            "dst_indices": [int(x) for x in dst_indices],
                            "ack_host": ack_host,
                            "ack_port": ack_port,
                        },
                    )
                    continue
                # Staging: prefill notifies a chunk written to staging buffer
                if msg[0] == b"CHUNK_READY":
                    room = int(msg[1].decode("ascii"))
                    chunk_idx = int(msg[2].decode("ascii"))
                    page_start = int(msg[3].decode("ascii"))
                    num_pages = int(msg[4].decode("ascii"))
                    session_id = msg[5].decode("ascii")
                    handler = self._staging_handler
                    assert handler is not None, (
                        "CHUNK_READY received before staging handler initialized"
                    )
                    handler.handle_chunk_arrived(
                        room,
                        chunk_idx,
                        page_start,
                        num_pages,
                        session_id,
                    )
                    continue

                # Staging: prefill pre-requests staging allocation before forward
                if msg[0] == b"STAGING_REQ":
                    self._handle_staging_req(msg)
                    continue

                if self.handle_abort_ack_message(msg):
                    continue

                parsed = self.parse_kv_status_message(msg)
                if parsed is None:
                    continue
                room, status, prefill_rank, reason = parsed
                self.apply_prefill_status(
                    bootstrap_room=room,
                    status=status,
                    prefill_rank=prefill_rank,
                    failure_reason=reason,
                )

        t = threading.Thread(target=decode_thread, daemon=True)
        t.start()
        self._worker_threads.append(t)
        t = self._start_heartbeat_checker_thread()
        self._worker_threads.append(t)

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        trace_ctx: Optional[Union[TraceReqContext, TraceNullContext]] = None,
        source_event=None,
        pd_hidden_start: Optional[int] = None,
        pd_hidden_row_len: int = 0,
        pd_hidden_is_last_chunk: bool = False,
        pd_hidden_release_indices: Optional[List[int]] = None,
        wait_event: Optional[object] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        if trace_ctx is None:
            trace_ctx = TraceNullContext()

        self._publish_source_chunk(
            self.transfer_queues[shard_idx],
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                source_event=source_event,
                pd_hidden_start=pd_hidden_start,
                pd_hidden_row_len=pd_hidden_row_len,
                pd_hidden_is_last_chunk=pd_hidden_is_last_chunk,
                pd_hidden_release_indices=pd_hidden_release_indices,
                trace_ctx=trace_ctx,
                wait_event=wait_event,
            ),
        )

    def _publish_source_chunk(self, queue, chunk):
        self._source_drain.publish(
            chunk,
            queue,
            lambda: (
                chunk.room in self.request_status
                and self.check_status(chunk.room) != KVPoll.Failed
            ),
        )

    def source_transfers_drained(self, room):
        if self.request_status.get(room) in (None, KVPoll.Failed):
            # failure_exception() may already have removed request_status.
            # Also covers abort racing with a worker parking after its status
            # check. Each scheduler poll wakes any newly parked chunks.
            self._wake_pd_hidden_ack_waiters(room)
        return self._source_drain.drained(room)

class MooncakeKVSenderMethods:
    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super(N.MooncakeKVSender, self).__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            req_has_disagg_prefill_dp_rank,
        )
        self.init_time = time.time()
        self._source_event = None
        self._pd_hidden_chunk_meta = None
        self._init_trace_ctx()

    def set_source_event(self, source_event) -> None:
        self._source_event = source_event

    def set_pd_hidden_chunk_meta(
        self,
        hidden_start: int,
        row_len: int,
        is_last_hidden_chunk: bool,
        release_indices: Optional[List[int]] = None,
    ) -> None:
        self._pd_hidden_chunk_meta = (
            int(hidden_start),
            int(row_len),
            bool(is_last_hidden_chunk),
            [int(x) for x in release_indices] if release_indices else None,
        )

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_SEND)
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            self._source_event = None
            return

        pd_hidden_chunk_meta = self._pd_hidden_chunk_meta
        self._pd_hidden_chunk_meta = None
        wait_event = getattr(self, "_early_send_wait_event", None)
        self._early_send_wait_event = None
        if not is_last_chunk:
            source_event = self._source_event
            self._source_event = None
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                num_kv_tokens=num_kv_tokens,
                state_indices=state_indices,
                source_event=source_event,
                trace_ctx=self.trace_ctx.copy_for_thread(),
                pd_hidden_start=(
                    pd_hidden_chunk_meta[0] if pd_hidden_chunk_meta else None
                ),
                pd_hidden_row_len=(
                    pd_hidden_chunk_meta[1] if pd_hidden_chunk_meta else 0
                ),
                pd_hidden_is_last_chunk=(
                    pd_hidden_chunk_meta[2] if pd_hidden_chunk_meta else False
                ),
                pd_hidden_release_indices=(
                    pd_hidden_chunk_meta[3] if pd_hidden_chunk_meta else None
                ),
                wait_event=wait_event,
            )
        else:
            source_event = self._source_event
            self._source_event = None
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                source_event=source_event,
                trace_ctx=self.trace_ctx.copy_for_thread(),
                pd_hidden_start=(
                    pd_hidden_chunk_meta[0] if pd_hidden_chunk_meta else None
                ),
                pd_hidden_row_len=(
                    pd_hidden_chunk_meta[1] if pd_hidden_chunk_meta else 0
                ),
                pd_hidden_is_last_chunk=(
                    pd_hidden_chunk_meta[2] if pd_hidden_chunk_meta else False
                ),
                pd_hidden_release_indices=(
                    pd_hidden_chunk_meta[3] if pd_hidden_chunk_meta else None
                ),
                wait_event=wait_event,
            )
        self._record_transfer_indices(kv_indices, state_indices)

    def is_transfer_drained(self):
        return self.kv_mgr.source_transfers_drained(self.bootstrap_room)

    def abort(self):
        # Serialize Failed publication against the final enqueue.
        with self.kv_mgr._source_drain.lock:
            super(N.MooncakeKVSender, self).abort()
        self.kv_mgr._wake_pd_hidden_ack_waiters(self.bootstrap_room)
        self.trace_ctx.abort(abort_info={"reason": "Aborted"})
        self.trace_ctx.trace_req_finish()

class MooncakeKVReceiverMethods:
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        spec_metadata: Optional[dict] = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        self.chunk_staging_infos = []
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info in self.bootstrap_infos:
            is_dummy = bootstrap_info["is_dummy"]
            local_state_indices = state_indices
            local_spec_metadata = spec_metadata
            if spec_metadata and spec_metadata.get("pp_slices"):
                pp_rank_value = bootstrap_info.get(
                    "target_pp_rank", bootstrap_info.get("pp_rank")
                )
                if pp_rank_value is None:
                    raise RuntimeError(
                        "PD hidden metadata requires target_pp_rank in "
                        f"bootstrap_info, got keys={sorted(bootstrap_info.keys())}"
                    )
                pp_rank = int(pp_rank_value)
                pp_slice = spec_metadata["pp_slices"].get(str(pp_rank))
                if pp_slice is None:
                    raise RuntimeError(
                        "PD hidden metadata is missing slice for "
                        f"target_pp_rank={pp_rank}, available_pp_slices="
                        f"{sorted(spec_metadata['pp_slices'].keys())}"
                    )
                local_spec_metadata = {
                    **spec_metadata,
                    "target_pp_rank": int(pp_rank),
                    "pp_slice": pp_slice,
                }
                local_state_indices = list(
                    state_indices
                    if state_indices is not None
                    else [None] * len(self.kv_mgr.kv_args.state_types)
                )
                for idx, state_type in enumerate(self.kv_mgr.kv_args.state_types):
                    if state_type == StateType.PD_HIDDEN:
                        local_state_indices[idx] = pp_slice.get("dst_indices", [])
                        break

            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.session_id.encode("ascii"),
                            kv_indices.tobytes() if not is_dummy else b"",
                            str(aux_index).encode("ascii") if not is_dummy else b"",
                            (
                                pack_int_lists(local_state_indices, "i")
                                if not is_dummy and local_state_indices
                                else b""
                            ),
                            str(self.required_dst_info_num).encode("ascii"),
                            str(decode_prefix_len or 0).encode("ascii"),
                            (
                                np.asarray(device_kv_indices, dtype=np.int32).tobytes()
                                if not is_dummy and device_kv_indices is not None
                                else b""
                            ),
                            (
                                json.dumps(local_spec_metadata).encode("utf-8")
                                if local_spec_metadata
                                else b""
                            ),
                        ]
                    )
            except zmq.ZMQError:
                self.invalidate_cached_bootstrap_infos()
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
        self.init_time = time.time()

def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    N.MooncakeKVManager.PD_HIDDEN_CHUNK_READY_HEADER = b"PD_HIDDEN_CHUNK_READY"
    N.MooncakeKVManager.PD_HIDDEN_CHUNK_ACK_HEADER = b"PD_HIDDEN_CHUNK_ACK"
    if hasattr(N.MooncakeKVManager, '__init__'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.__init__", MooncakeKVManagerMethods.__dict__['__init__'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '__init__', MooncakeKVManagerMethods.__dict__['__init__'])
    if hasattr(N.MooncakeKVManager, 'supports_pd_hidden_streaming'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.supports_pd_hidden_streaming", MooncakeKVManagerMethods.__dict__['supports_pd_hidden_streaming'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'supports_pd_hidden_streaming', MooncakeKVManagerMethods.__dict__['supports_pd_hidden_streaming'])
    if hasattr(N.MooncakeKVManager, 'mark_pd_hidden_request_done'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.mark_pd_hidden_request_done", MooncakeKVManagerMethods.__dict__['mark_pd_hidden_request_done'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'mark_pd_hidden_request_done', MooncakeKVManagerMethods.__dict__['mark_pd_hidden_request_done'])
    if hasattr(N.MooncakeKVManager, 'pop_pd_hidden_request_done'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.pop_pd_hidden_request_done", MooncakeKVManagerMethods.__dict__['pop_pd_hidden_request_done'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'pop_pd_hidden_request_done', MooncakeKVManagerMethods.__dict__['pop_pd_hidden_request_done'])
    if hasattr(N.MooncakeKVManager, 'mark_pd_hidden_done'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.mark_pd_hidden_done", MooncakeKVManagerMethods.__dict__['mark_pd_hidden_done'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'mark_pd_hidden_done', MooncakeKVManagerMethods.__dict__['mark_pd_hidden_done'])
    if hasattr(N.MooncakeKVManager, 'pop_pd_hidden_done'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.pop_pd_hidden_done", MooncakeKVManagerMethods.__dict__['pop_pd_hidden_done'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'pop_pd_hidden_done', MooncakeKVManagerMethods.__dict__['pop_pd_hidden_done'])
    if hasattr(N.MooncakeKVManager, 'park_pd_hidden_chunk_for_ack'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.park_pd_hidden_chunk_for_ack", MooncakeKVManagerMethods.__dict__['park_pd_hidden_chunk_for_ack'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'park_pd_hidden_chunk_for_ack', MooncakeKVManagerMethods.__dict__['park_pd_hidden_chunk_for_ack'])
    if hasattr(N.MooncakeKVManager, '_wake_pd_hidden_ack_waiters'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._wake_pd_hidden_ack_waiters", MooncakeKVManagerMethods.__dict__['_wake_pd_hidden_ack_waiters'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_wake_pd_hidden_ack_waiters', MooncakeKVManagerMethods.__dict__['_wake_pd_hidden_ack_waiters'])
    if hasattr(N.MooncakeKVManager, '_park_pd_hidden_chunk_behind_room'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._park_pd_hidden_chunk_behind_room", MooncakeKVManagerMethods.__dict__['_park_pd_hidden_chunk_behind_room'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_park_pd_hidden_chunk_behind_room', MooncakeKVManagerMethods.__dict__['_park_pd_hidden_chunk_behind_room'])
    if hasattr(N.MooncakeKVManager, '_wake_next_pd_hidden_room_waiter'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._wake_next_pd_hidden_room_waiter", MooncakeKVManagerMethods.__dict__['_wake_next_pd_hidden_room_waiter'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_wake_next_pd_hidden_room_waiter', MooncakeKVManagerMethods.__dict__['_wake_next_pd_hidden_room_waiter'])
    if hasattr(N.MooncakeKVManager, '_handle_pd_hidden_chunk_ack'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._handle_pd_hidden_chunk_ack", MooncakeKVManagerMethods.__dict__['_handle_pd_hidden_chunk_ack'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_handle_pd_hidden_chunk_ack', MooncakeKVManagerMethods.__dict__['_handle_pd_hidden_chunk_ack'])
    if hasattr(N.MooncakeKVManager, 'pop_pd_hidden_ready_chunks'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.pop_pd_hidden_ready_chunks", MooncakeKVManagerMethods.__dict__['pop_pd_hidden_ready_chunks'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'pop_pd_hidden_ready_chunks', MooncakeKVManagerMethods.__dict__['pop_pd_hidden_ready_chunks'])
    if hasattr(N.MooncakeKVManager, 'submit_pd_hidden_chunk_ack'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.submit_pd_hidden_chunk_ack", MooncakeKVManagerMethods.__dict__['submit_pd_hidden_chunk_ack'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'submit_pd_hidden_chunk_ack', MooncakeKVManagerMethods.__dict__['submit_pd_hidden_chunk_ack'])
    if hasattr(N.MooncakeKVManager, '_drain_pd_hidden_ack_completions'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._drain_pd_hidden_ack_completions", MooncakeKVManagerMethods.__dict__['_drain_pd_hidden_ack_completions'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_drain_pd_hidden_ack_completions', MooncakeKVManagerMethods.__dict__['_drain_pd_hidden_ack_completions'])
    if hasattr(N.MooncakeKVManager, 'pop_pd_hidden_acked_chunks'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.pop_pd_hidden_acked_chunks", MooncakeKVManagerMethods.__dict__['pop_pd_hidden_acked_chunks'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'pop_pd_hidden_acked_chunks', MooncakeKVManagerMethods.__dict__['pop_pd_hidden_acked_chunks'])
    if hasattr(N.MooncakeKVManager, 'wait_pd_hidden_ack_completions'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.wait_pd_hidden_ack_completions", MooncakeKVManagerMethods.__dict__['wait_pd_hidden_ack_completions'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'wait_pd_hidden_ack_completions', MooncakeKVManagerMethods.__dict__['wait_pd_hidden_ack_completions'])
    if hasattr(N.MooncakeKVManager, '_begin_pd_hidden_transfer'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._begin_pd_hidden_transfer", MooncakeKVManagerMethods.__dict__['_begin_pd_hidden_transfer'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_begin_pd_hidden_transfer', MooncakeKVManagerMethods.__dict__['_begin_pd_hidden_transfer'])
    if hasattr(N.MooncakeKVManager, '_end_pd_hidden_transfer'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._end_pd_hidden_transfer", MooncakeKVManagerMethods.__dict__['_end_pd_hidden_transfer'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_end_pd_hidden_transfer', MooncakeKVManagerMethods.__dict__['_end_pd_hidden_transfer'])
    if hasattr(N.MooncakeKVManager, '_wait_pd_hidden_transfers_quiesced'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._wait_pd_hidden_transfers_quiesced", MooncakeKVManagerMethods.__dict__['_wait_pd_hidden_transfers_quiesced'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_wait_pd_hidden_transfers_quiesced', MooncakeKVManagerMethods.__dict__['_wait_pd_hidden_transfers_quiesced'])
    if hasattr(N.MooncakeKVManager, '_pd_hidden_state_index'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._pd_hidden_state_index", MooncakeKVManagerMethods.__dict__['_pd_hidden_state_index'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_pd_hidden_state_index', MooncakeKVManagerMethods.__dict__['_pd_hidden_state_index'])
    if hasattr(N.MooncakeKVManager, '_has_pd_hidden_state'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._has_pd_hidden_state", MooncakeKVManagerMethods.__dict__['_has_pd_hidden_state'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_has_pd_hidden_state', MooncakeKVManagerMethods.__dict__['_has_pd_hidden_state'])
    if hasattr(N.MooncakeKVManager, '_without_pd_hidden_state'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._without_pd_hidden_state", MooncakeKVManagerMethods.__dict__['_without_pd_hidden_state'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_without_pd_hidden_state', MooncakeKVManagerMethods.__dict__['_without_pd_hidden_state'])
    if hasattr(N.MooncakeKVManager, '_pd_hidden_release_state_indices'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._pd_hidden_release_state_indices", MooncakeKVManagerMethods.__dict__['_pd_hidden_release_state_indices'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_pd_hidden_release_state_indices', MooncakeKVManagerMethods.__dict__['_pd_hidden_release_state_indices'])
    if hasattr(N.MooncakeKVManager, '_free_pd_hidden_state_indices'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._free_pd_hidden_state_indices", MooncakeKVManagerMethods.__dict__['_free_pd_hidden_state_indices'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_free_pd_hidden_state_indices', MooncakeKVManagerMethods.__dict__['_free_pd_hidden_state_indices'])
    if hasattr(N.MooncakeKVManager, '_free_pd_hidden_chunk_rows'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._free_pd_hidden_chunk_rows", MooncakeKVManagerMethods.__dict__['_free_pd_hidden_chunk_rows'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_free_pd_hidden_chunk_rows', MooncakeKVManagerMethods.__dict__['_free_pd_hidden_chunk_rows'])
    if hasattr(N.MooncakeKVManager, '_release_or_mark_pd_hidden_done'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._release_or_mark_pd_hidden_done", MooncakeKVManagerMethods.__dict__['_release_or_mark_pd_hidden_done'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_release_or_mark_pd_hidden_done', MooncakeKVManagerMethods.__dict__['_release_or_mark_pd_hidden_done'])
    if hasattr(N.MooncakeKVManager, '_finish_pd_hidden_streaming_chunk'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._finish_pd_hidden_streaming_chunk", MooncakeKVManagerMethods.__dict__['_finish_pd_hidden_streaming_chunk'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_finish_pd_hidden_streaming_chunk', MooncakeKVManagerMethods.__dict__['_finish_pd_hidden_streaming_chunk'])
    if hasattr(N.MooncakeKVManager, '_mark_session_failed_and_sync'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._mark_session_failed_and_sync", MooncakeKVManagerMethods.__dict__['_mark_session_failed_and_sync'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_mark_session_failed_and_sync', MooncakeKVManagerMethods.__dict__['_mark_session_failed_and_sync'])
    if hasattr(N.MooncakeKVManager, 'notify_pd_hidden_chunk_ready'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.notify_pd_hidden_chunk_ready", MooncakeKVManagerMethods.__dict__['notify_pd_hidden_chunk_ready'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'notify_pd_hidden_chunk_ready', MooncakeKVManagerMethods.__dict__['notify_pd_hidden_chunk_ready'])
    if hasattr(N.MooncakeKVManager, 'ack_pd_hidden_chunk'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.ack_pd_hidden_chunk", MooncakeKVManagerMethods.__dict__['ack_pd_hidden_chunk'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'ack_pd_hidden_chunk', MooncakeKVManagerMethods.__dict__['ack_pd_hidden_chunk'])
    if hasattr(N.MooncakeKVManager, '_notify_transfer_waiters'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._notify_transfer_waiters", MooncakeKVManagerMethods.__dict__['_notify_transfer_waiters'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_notify_transfer_waiters', MooncakeKVManagerMethods.__dict__['_notify_transfer_waiters'])
    if hasattr(N.MooncakeKVManager, 'update_status'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.update_status", MooncakeKVManagerMethods.__dict__['update_status'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'update_status', MooncakeKVManagerMethods.__dict__['update_status'])
    if hasattr(N.MooncakeKVManager, 'wait_for_transfer_rooms'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.wait_for_transfer_rooms", MooncakeKVManagerMethods.__dict__['wait_for_transfer_rooms'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'wait_for_transfer_rooms', MooncakeKVManagerMethods.__dict__['wait_for_transfer_rooms'])
    if hasattr(N.MooncakeKVManager, 'send_kvcache_slice'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.send_kvcache_slice", MooncakeKVManagerMethods.__dict__['send_kvcache_slice'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'send_kvcache_slice', MooncakeKVManagerMethods.__dict__['send_kvcache_slice'])
    if hasattr(N.MooncakeKVManager, '_send_pd_hidden_packet'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._send_pd_hidden_packet", MooncakeKVManagerMethods.__dict__['_send_pd_hidden_packet'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_send_pd_hidden_packet', MooncakeKVManagerMethods.__dict__['_send_pd_hidden_packet'])
    if hasattr(N.MooncakeKVManager, '_handle_deferred_abort_notification'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._handle_deferred_abort_notification", MooncakeKVManagerMethods.__dict__['_handle_deferred_abort_notification'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_handle_deferred_abort_notification', MooncakeKVManagerMethods.__dict__['_handle_deferred_abort_notification'])
    if hasattr(N.MooncakeKVManager, '_handle_legacy_abort_notification'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._handle_legacy_abort_notification", MooncakeKVManagerMethods.__dict__['_handle_legacy_abort_notification'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_handle_legacy_abort_notification', MooncakeKVManagerMethods.__dict__['_handle_legacy_abort_notification'])
    if hasattr(N.MooncakeKVManager, 'start_prefill_thread'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.start_prefill_thread", MooncakeKVManagerMethods.__dict__['start_prefill_thread'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'start_prefill_thread', MooncakeKVManagerMethods.__dict__['start_prefill_thread'])
    if hasattr(N.MooncakeKVManager, 'start_decode_thread'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.start_decode_thread", MooncakeKVManagerMethods.__dict__['start_decode_thread'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'start_decode_thread', MooncakeKVManagerMethods.__dict__['start_decode_thread'])
    if hasattr(N.MooncakeKVManager, 'add_transfer_request'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.add_transfer_request", MooncakeKVManagerMethods.__dict__['add_transfer_request'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'add_transfer_request', MooncakeKVManagerMethods.__dict__['add_transfer_request'])
    if hasattr(N.MooncakeKVManager, '_publish_source_chunk'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._publish_source_chunk", MooncakeKVManagerMethods.__dict__['_publish_source_chunk'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, '_publish_source_chunk', MooncakeKVManagerMethods.__dict__['_publish_source_chunk'])
    if hasattr(N.MooncakeKVManager, 'source_transfers_drained'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.source_transfers_drained", MooncakeKVManagerMethods.__dict__['source_transfers_drained'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVManager, 'source_transfers_drained', MooncakeKVManagerMethods.__dict__['source_transfers_drained'])
    if hasattr(N.MooncakeKVSender, '__init__'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVSender.__init__", MooncakeKVSenderMethods.__dict__['__init__'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVSender, '__init__', MooncakeKVSenderMethods.__dict__['__init__'])
    if hasattr(N.MooncakeKVSender, 'set_source_event'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVSender.set_source_event", MooncakeKVSenderMethods.__dict__['set_source_event'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVSender, 'set_source_event', MooncakeKVSenderMethods.__dict__['set_source_event'])
    if hasattr(N.MooncakeKVSender, 'set_pd_hidden_chunk_meta'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVSender.set_pd_hidden_chunk_meta", MooncakeKVSenderMethods.__dict__['set_pd_hidden_chunk_meta'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVSender, 'set_pd_hidden_chunk_meta', MooncakeKVSenderMethods.__dict__['set_pd_hidden_chunk_meta'])
    if hasattr(N.MooncakeKVSender, 'send'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVSender.send", MooncakeKVSenderMethods.__dict__['send'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVSender, 'send', MooncakeKVSenderMethods.__dict__['send'])
    if hasattr(N.MooncakeKVSender, 'is_transfer_drained'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVSender.is_transfer_drained", MooncakeKVSenderMethods.__dict__['is_transfer_drained'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVSender, 'is_transfer_drained', MooncakeKVSenderMethods.__dict__['is_transfer_drained'])
    if hasattr(N.MooncakeKVSender, 'abort'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVSender.abort", MooncakeKVSenderMethods.__dict__['abort'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVSender, 'abort', MooncakeKVSenderMethods.__dict__['abort'])
    if hasattr(N.MooncakeKVReceiver, 'send_metadata'):
        HookRegistry.register("sglang.srt.disaggregation.mooncake.conn.MooncakeKVReceiver.send_metadata", MooncakeKVReceiverMethods.__dict__['send_metadata'], HookType.REPLACE)
    else:
        setattr(N.MooncakeKVReceiver, 'send_metadata', MooncakeKVReceiverMethods.__dict__['send_metadata'])
