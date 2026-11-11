# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations


class TokenMajor:
    def build_mamba_entry_views(
        raw: torch.Tensor,
        *,
        layer_num: int,
        conv_state_shapes: Sequence[Sequence[int]],
        conv_dtype: torch.dtype,
        temporal_state_shape: Sequence[int],
        temporal_dtype: torch.dtype,
        max_slots: int,
        anchor_bytes: int = 0,
        entry_bytes: Optional[int] = None,
        offset_bytes: int = 0,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-slot envelope views over ``raw`` for Mamba state.

        Layout of one block: ``[conv[0] rows × layers][conv[1] rows × layers]...
        [temporal rows × layers]``, starting ``offset_bytes`` into a slot's entry
        of ``entry_bytes`` (the block's own size when omitted, i.e. a standalone
        envelope). Each returned view has shape ``(num_layers, max_slots,
        *inner_shape)`` matching ``MambaPool.State.conv[i]`` / ``.temporal``.
        Mamba state is always token-granular (page_size == 1).
        """
        if entry_bytes is None:
            entry_bytes = mamba_entry_bytes(
                layer_num=layer_num,
                conv_state_shapes=conv_state_shapes,
                conv_dtype=conv_dtype,
                temporal_state_shape=temporal_state_shape,
                temporal_dtype=temporal_dtype,
            )

        conv_itemsize = conv_dtype.itemsize
        assert entry_bytes % conv_itemsize == 0, (
            f"misaligned mamba spec: per-slot entry_bytes={entry_bytes} is not a "
            f"multiple of the conv-state itemsize {conv_itemsize} B"
        )
        assert anchor_bytes % conv_itemsize == 0, (
            f"misaligned mamba spec: anchor_bytes={anchor_bytes} is not a multiple "
            f"of the conv-state itemsize {conv_itemsize} B"
        )
        as_conv_dtype = raw.view(conv_dtype)
        conv_slot_stride_elems = entry_bytes // conv_itemsize

        offset_bytes_within_entry = offset_bytes
        conv_views: List[torch.Tensor] = []
        for shape in conv_state_shapes:
            inner_shape_bytes = _prod(shape) * conv_itemsize
            assert inner_shape_bytes % conv_itemsize == 0
            offset_elems = (anchor_bytes + offset_bytes_within_entry) // conv_itemsize
            stride = (
                inner_shape_bytes // conv_itemsize,
                conv_slot_stride_elems,
            ) + _contiguous_strides(shape)
            conv_views.append(
                torch.as_strided(
                    as_conv_dtype,
                    size=(layer_num, max_slots) + tuple(shape),
                    stride=stride,
                    storage_offset=offset_elems,
                )
            )
            offset_bytes_within_entry += layer_num * inner_shape_bytes

        # The temporal view's storage_offset is computed in temporal-dtype elements
        # by integer-dividing a byte offset by itemsize, so every term of that byte
        # offset (entry stride, anchor, the conv region) must be a whole multiple of
        # itemsize or the offset truncates and mis-places the view.
        itemsize = temporal_dtype.itemsize
        assert entry_bytes % itemsize == 0, (
            f"misaligned mamba spec: per-slot entry_bytes={entry_bytes} is not a "
            f"multiple of the temporal-state itemsize {itemsize} B; the temporal "
            f"view's storage_offset would truncate and mis-place the state"
        )
        assert anchor_bytes % itemsize == 0, (
            f"misaligned mamba spec: anchor_bytes={anchor_bytes} is not a multiple "
            f"of the temporal-state itemsize {itemsize} B"
        )
        inner_shape_bytes = _prod(temporal_state_shape) * itemsize
        assert inner_shape_bytes % itemsize == 0, (
            f"misaligned mamba spec: temporal inner_shape_bytes={inner_shape_bytes} "
            f"is not a multiple of the temporal-state itemsize {itemsize} B"
        )
        assert (anchor_bytes + offset_bytes_within_entry) % itemsize == 0, (
            f"misaligned mamba spec: temporal region byte offset "
            f"{anchor_bytes + offset_bytes_within_entry} is not a multiple of the "
            f"temporal-state itemsize {itemsize} B"
        )
        offset_elems = (anchor_bytes + offset_bytes_within_entry) // itemsize
        as_temporal_dtype = raw.view(temporal_dtype)
        stride = (
            inner_shape_bytes // itemsize,
            entry_bytes // itemsize,
        ) + _contiguous_strides(temporal_state_shape)
        temporal_view = torch.as_strided(
            as_temporal_dtype,
            size=(layer_num, max_slots) + tuple(temporal_state_shape),
            stride=stride,
            storage_offset=offset_elems,
        )
        return conv_views, temporal_view



class MultiModalStaticCache:
    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True
        tensor = embedding.embedding
        storage_size = _get_tensor_size(tensor)
        data_size = min(storage_size, tensor.element_size() * tensor.numel())
        # An uncacheable video must not evict useful entries on every chunk.
        if data_size > self.max_size:
            return False
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding.embedding)

        if storage_size > data_size:
            # Clone admitted slices so one cached item cannot retain a whole batch.
            embedding = replace(embedding, embedding=tensor.clone())
        data_size = _get_tensor_size(embedding.embedding)
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True



class HiCacheHF3FS:
    def __init__(
        self,
        rank: int,
        file_path: str,
        file_size: int,
        numjobs: int,
        bytes_per_page: int,
        entries: int,
        client_timeout: int,
        dtype: torch.dtype,
        metadata_client: Hf3fsMetadataInterface,
        is_mla_model: bool = False,
        is_page_first_layout: bool = False,
        use_mock_client: bool = False,
        enable_storage_metrics: bool = False,
        tp_size: int = 1,
    ):
        self.rank = rank
        self.tp_rank = rank
        self.tp_size = tp_size
        self.file_path = file_path
        self.file_size = file_size
        self.numjobs = numjobs
        self.bytes_per_page = bytes_per_page
        self.gb_per_page = bytes_per_page / (1 << 30)
        self.entries = entries
        self.client_timeout = client_timeout
        self.dtype = dtype
        self.metadata_client = metadata_client
        self.is_mla_model = is_mla_model
        self.is_page_first_layout = is_page_first_layout
        self.enable_storage_metrics = enable_storage_metrics
        self.use_mock_client = use_mock_client
        self.numel = self.bytes_per_page // self.dtype.itemsize
        self.num_pages = self.file_size // self.bytes_per_page
        self.skip_backup = False
        if self.is_mla_model and self.rank != 0:
            self.skip_backup = True
            self.rank = 0

        self.is_zero_copy = False

        logger.info(
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "
            f"file_path={self.file_path}, "
            f"file_size={self.file_size / (2**30):.2f} GB, "
            f"num_pages={self.num_pages}, "
            f"is_mla_model={self.is_mla_model}"
        )

        self.ac = AtomicCounter(self.numjobs)
        self.clients = [
            create_hf3fs_client(
                self.file_path,
                self.file_size,
                self.bytes_per_page,
                self.entries,
                self.client_timeout,
                use_mock_client,
            )
            for _ in range(numjobs)
        ]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs, thread_name_prefix=f"HiCacheHF3FS-Rank{self.rank}"
        )

        self.metadata_client.initialize(self.rank, self.num_pages)
        self.lock = threading.RLock()
        self._pool_storage_ctx: dict = {}

        atexit.register(self.close)

        signal.signal(signal.SIGINT, lambda sig, frame: self.close())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.close())
        signal.signal(signal.SIGQUIT, lambda sig, frame: self.close())

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

    @staticmethod
    def from_env_config(
        bytes_per_page: int,
        dtype: torch.dtype,
        storage_config: HiCacheStorageConfig = None,
    ) -> "HiCacheHF3FS":
        """Create a HiCacheHF3FS instance from environment configuration.

        Environment:
            - Uses env var stored in `HiCacheHF3FS.default_env_var` to locate a JSON config.
            - Falls back to a local single-machine config when the env var is not set.

        Raises:
            ValueError: If MLA Model is requested without global metadata server or required keys are missing.
        """
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            Hf3fsGlobalMetadataClient,
            Hf3fsLocalMetadataClient,
        )

        use_mock_client = False
        if storage_config is not None:
            rank, is_mla_model, is_page_first_layout = (
                storage_config.tp_rank,
                storage_config.is_mla_model,
                storage_config.is_page_first_layout,
            )

            if storage_config.extra_config is not None:
                use_mock_client = storage_config.extra_config.get(
                    "use_mock_hf3fs_client", False
                )
        else:
            rank, is_mla_model, is_page_first_layout = (
                0,
                False,
                False,
            )

        mla_unsupported_msg = f"MLA model is not supported without global metadata server, please refer to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/hf3fs/docs/deploy_sglang_3fs_multinode.md"

        config_path = os.getenv(HiCacheHF3FS.default_env_var)
        if not config_path:
            if is_mla_model:
                raise ValueError(mla_unsupported_msg)

            return HiCacheHF3FS(
                rank=rank,
                file_path=f"/data/hicache.{rank}.bin",
                file_size=1 << 40,
                numjobs=16,
                bytes_per_page=bytes_per_page,
                entries=8,
                client_timeout=5,
                dtype=dtype,
                metadata_client=Hf3fsLocalMetadataClient(),
                is_page_first_layout=is_page_first_layout,
                use_mock_client=use_mock_client,
            )

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")

        # Check required keys (metadata_server_url is now optional)
        required_keys = {
            "file_path_prefix",
            "file_size",
            "numjobs",
            "entries",
        }
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")

        # Choose metadata client based on configuration
        if config.get("metadata_server_url"):
            # Use global metadata client to connect to metadata server
            metadata_server_url = config["metadata_server_url"]
            metadata_client = Hf3fsGlobalMetadataClient(metadata_server_url)

            logger.info(
                f"Using global metadata client with server url: {metadata_server_url}"
            )
        else:
            # Enable MLA optimization only when using the global metadata client
            if is_mla_model:
                raise ValueError(mla_unsupported_msg)

            # Use local metadata client for single-machine deployment
            metadata_client = Hf3fsLocalMetadataClient()

        rank_for_path = 0 if is_mla_model else rank
        return HiCacheHF3FS(
            rank=rank,
            # Let all ranks use the same file path for MLA model
            file_path=f"{config['file_path_prefix']}.{rank_for_path}.bin",
            file_size=int(config["file_size"]),
            numjobs=int(config["numjobs"]),
            bytes_per_page=bytes_per_page,
            entries=int(config["entries"]),
            client_timeout=config.get("client_timeout", 5),
            dtype=dtype,
            metadata_client=metadata_client,
            is_mla_model=is_mla_model,
            is_page_first_layout=is_page_first_layout,
            use_mock_client=use_mock_client,
            enable_storage_metrics=storage_config.enable_storage_metrics,
            tp_size=storage_config.tp_size,
        )

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break

            pool_name = transfer.name
            ctx = self._pool_storage_ctx.get(pool_name)
            if ctx is None:
                final_pages = 0
                break

            component_keys = [
                self._pool_component_key(key, pool_name) for key in keys[:kv_pages]
            ]
            exists_results = self.metadata_client.exists(
                self.rank, component_keys, namespace=ctx.namespace
            )

            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = exists_results.index(False)
                except ValueError:
                    boundary = kv_pages
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        exists_results[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[pool_name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _pool_batch_get(self, transfer: PoolTransfer) -> List[bool]:
        pool_name = transfer.name
        ctx = self._pool_storage_ctx[pool_name]
        host_pool = self.registered_pools[pool_name]
        keys = transfer.keys
        host_indices = transfer.host_indices
        page_size = getattr(host_pool, "page_size", 1) or 1
        page_num = len(keys)

        component_keys = [self._pool_component_key(key, pool_name) for key in keys]
        page_indices = self.metadata_client.get_page_indices(
            self.rank, component_keys, namespace=ctx.namespace
        )

        batch_indices, file_offsets, values = [], [], []
        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * ctx.bytes_per_page)
                values.append(host_pool.get_dummy_flat_data_page())

        if not batch_indices:
            return [False] * page_num

        start_time = time.perf_counter()
        futures = [
            self.executor.submit(
                ctx.clients[self.ac.next()].batch_read,
                file_offsets[j : j + self.entries],
                values[j : j + self.entries],
            )
            for j in range(0, len(batch_indices), self.entries)
        ]
        read_results = [r for f in futures for r in f.result()]
        end_time = time.perf_counter()
        ionum = len(batch_indices)

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(ionum)
            self.prefetch_bandwidth.append(
                ionum / (end_time - start_time) * ctx.gb_per_page
            )

        results = [False] * page_num
        for idx, (batch_idx, read_result) in enumerate(
            zip(batch_indices, read_results)
        ):
            if read_result == ctx.bytes_per_page:
                host_idx = host_indices[batch_idx * page_size].item()
                host_pool.set_from_flat_data_page(host_idx, values[idx])
                results[batch_idx] = True
            else:
                logger.error(
                    f"[Rank {self.rank}][Pool {pool_name.upper()}] HiCacheHF3FS get {keys[batch_idx]} failed"
                )

        return results

    def _pool_batch_set(self, transfer: PoolTransfer) -> List[bool]:
        pool_name = transfer.name
        ctx = self._pool_storage_ctx[pool_name]
        host_pool = self.registered_pools[pool_name]
        keys = transfer.keys
        host_indices = transfer.host_indices
        page_size = getattr(host_pool, "page_size", 1) or 1
        page_num = len(keys)

        component_keys = [self._pool_component_key(key, pool_name) for key in keys]
        key_with_prefix = [(k, "") for k in component_keys]
        indices = self.metadata_client.reserve_and_allocate_page_indices(
            self.rank, key_with_prefix, namespace=ctx.namespace
        )

        if len(indices) != page_num:
            logger.error(
                f"[Rank {self.rank}] Pool {pool_name}: mismatched indices length"
            )
            if indices:
                self.metadata_client.confirm_write(
                    self.rank, [], [idx[1] for idx in indices], namespace=ctx.namespace
                )
            return [False] * page_num

        batch_indices, file_offsets, file_values = [], [], []
        for i, (is_written, page_index) in enumerate(indices):
            if is_written or page_index == -1:
                continue
            batch_indices.append(i)
            file_offsets.append(page_index * ctx.bytes_per_page)
            host_idx = host_indices[i * page_size].item()
            data = host_pool.get_data_page(host_idx, flat=True)
            assert data.is_contiguous()
            file_values.append(data)

        start_time = time.perf_counter()
        futures = [
            self.executor.submit(
                ctx.clients[self.ac.next()].batch_write,
                file_offsets[j : j + self.entries],
                file_values[j : j + self.entries],
            )
            for j in range(0, len(batch_indices), self.entries)
        ]
        write_results = [r == ctx.bytes_per_page for f in futures for r in f.result()]
        end_time = time.perf_counter()
        ionum = len(batch_indices)

        if self.enable_storage_metrics:
            self.backup_pgs.append(ionum)
            self.backup_bandwidth.append(
                ionum / (end_time - start_time) * ctx.gb_per_page
            )

        written_keys_to_confirm = []
        pages_to_release = []
        results = [idx[0] for idx in indices]
        for batch_idx, write_ok in zip(batch_indices, write_results):
            key = component_keys[batch_idx]
            page_index = indices[batch_idx][1]
            if write_ok:
                written_keys_to_confirm.append((key, page_index))
            else:
                logger.error(
                    f"[Rank {self.rank}][Pool {pool_name.upper()}] HiCacheHF3FS set {keys[batch_idx]} failed"
                )
                pages_to_release.append(page_index)
            results[batch_idx] = write_ok

        if written_keys_to_confirm or pages_to_release:
            self.metadata_client.confirm_write(
                self.rank,
                written_keys_to_confirm,
                pages_to_release,
                namespace=ctx.namespace,
            )

        return results

    def _pool_component_key(self, key: str, pool_name: PoolName) -> str:
        # Mamba state is TP-sharded even when the model's MLA KV is replicated.
        if self.is_mla_model and pool_name == PoolName.MAMBA:
            return f"{key}_{pool_name}_{self.tp_rank}_{self.tp_size}"
        return f"{key}_{pool_name}"



class SeaweedFSStore:
    def __init__(
        self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None
    ) -> None:
        self.config = SeaweedFSConfig.from_extra_config(storage_config.extra_config)
        scope = [self.config.prefix.strip("/")] if self.config.prefix else []
        self.key_prefix = "/".join(scope + key_scope(storage_config))
        self._mamba_rank_tag = (
            f".tp{storage_config.tp_rank}of{storage_config.tp_size}"
            if storage_config.is_mla_model
            else ""
        )
        self._s3 = _make_s3_client(self.config)
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._ensure_bucket()
        logger.info(
            "SeaweedFS HiCache backend: endpoint=%s bucket=%s prefix=%s",
            self.config.endpoint,
            self.config.bucket,
            self.key_prefix,
        )

    def _component_key(self, key: str, pool_name) -> str:
        name = getattr(pool_name, "value", pool_name)
        if name == PoolName.KV.value:
            return key
        # Mamba state is TP-sharded even when the model's MLA KV is replicated.
        if name == PoolName.MAMBA.value:
            return f"{key}.{name}{self._mamba_rank_tag}"
        return f"{key}.{name}"



class Tensorcast:
    def _validate_storage_config(
        mem_pool_host: HostKVCache,
        *,
        family: _PoolFamily,
        storage_config: HiCacheStorageConfig,
    ) -> None:
        """Check that the storage configuration matches the implemented data path.

        The current checks cover only FULL v1. Extend this validation as support for
        additional pool families and storage API versions is implemented.
        """
        if mem_pool_host.layout not in _SUPPORTED_FULL_LAYOUTS:
            raise NotImplementedError(
                "TensorCast FULL v1 supports only page_first and page_first_direct, "
                f"got layout={mem_pool_host.layout!r}"
            )
        if mem_pool_host.layer_num != mem_pool_host.target_layer_num:
            raise NotImplementedError(
                "TensorCast FULL v1 does not support packed MTP draft pools"
            )
        if mem_pool_host.dcp_size != 1 or mem_pool_host.dcp_rank != 0:
            raise NotImplementedError(
                "TensorCast FULL v1 does not support DCP-aware HostPool layouts"
            )
        if storage_config.attn_cp_size != 1 or storage_config.attn_cp_rank != 0:
            raise NotImplementedError(
                "TensorCast FULL v1 requires attention context parallel size 1"
            )
        if storage_config.tp_lcm_size is not None:
            raise NotImplementedError("TensorCast FULL v1 does not support TP-LCM")
        if storage_config.should_split_heads:
            raise NotImplementedError("TensorCast FULL v1 does not support split heads")
        expected_mla = family is _PoolFamily.MLA
        if storage_config.is_mla_model is not expected_mla:
            raise ValueError(
                "TensorCast HostPool family conflicts with is_mla_model: "
                f"pool_family={family.value}, is_mla_model={storage_config.is_mla_model}"
            )
        if mem_pool_host.page_size <= 0 or mem_pool_host.page_num <= 0:
            raise ValueError(
                "TensorCast FULL HostPool requires positive page_size and page_num"
            )



class StorageAttachment:
    def attach(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Enable the storage backend at runtime.

        Starts the storage threads inside the cache controller and turns on the
        prefetch / backup paths. The caller must ensure there are no running or
        queued requests, so this cannot race the scheduler thread.
        """
        cache = self._cache

        # Validate first: a rejected request must have no side effects.
        invalid = self._validate_policies(
            hicache_storage_prefetch_policy, hicache_write_policy
        )
        if invalid is not None:
            return False, invalid

        controller = cache.cache_controller
        if controller is None:
            return (
                False,
                "HiCache is not initialized (no cache controller); "
                "launch with --enable-hierarchical-cache to attach a backend.",
            )

        # Write-back can retain auxiliary host rows without FULL, or backed
        # descendants below an unbacked FULL ancestor. Those states are invalid
        # for write-through insertion and eviction. Check before changing any
        # policy, including on the same-backend update and re-attach paths.
        if (
            cache.is_write_back
            and cache.host_memory_mode != "buffer_only"
            and hicache_write_policy in ("write_through", "write_through_selective")
            and not cache.tree_core.is_write_through_compatible()
        ):
            return (
                False,
                "Cannot switch HiCache from write_back to write_through while "
                "the cache has host data without its FULL prefix. "
                "Flush the cache before changing the write policy.",
            )

        if cache.enable_storage:
            current_backend = controller.storage_backend_type
            if current_backend != storage_backend:
                return (
                    False,
                    f"HiCache storage backend is already enabled with backend "
                    f"'{current_backend}'. Cannot attach different backend "
                    f"'{storage_backend}'. Detach first.",
                )
            # Same backend: the request degenerates to a policy update.
            self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)
            return (
                True,
                "HiCache storage backend already enabled with same backend; "
                "policies updated.",
            )

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json "
                f"'{storage_backend_extra_config_json}': {e}",
            )

        original_policies = (
            cache.prefetch_stop_policy,
            controller.write_policy,
            cache.write_through_threshold,
            cache.is_write_back,
        )
        try:
            prefetch_threshold = self.resolve_prefetch_threshold(prefetch_threshold)
            if cache.host_memory_mode == "buffer_only":
                validate_buffer_only_stack(
                    sidecar_pool_specs=cache.sidecar_pool_specs,
                    host_pool_group=cache.host_pool_group,
                    swa_component=cache.components.get(ComponentType.SWA),
                    auxiliary_swa_component=cache.components.get(
                        ComponentType.AUXILIARY_SWA
                    ),
                    storage_prefetch_threshold=prefetch_threshold,
                )
            # New workers must see the requested policy from their first operation.
            self._apply_policies(hicache_storage_prefetch_policy, hicache_write_policy)
            controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=controller.mem_pool_host.entries,
            )
        except Exception as e:
            (
                cache.prefetch_stop_policy,
                controller.write_policy,
                cache.write_through_threshold,
                cache.is_write_back,
            ) = original_policies
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self.apply_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=cache._enable_metrics_flag,
            extra_metric_labels=cache.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def resolve_prefetch_threshold(self, configured: int) -> int:
        """Use the same complete-window minimum for every buffer-mode anchor."""
        cache = self._cache
        if cache.host_memory_mode == "buffer_only":
            for ct in (ComponentType.SWA, ComponentType.AUXILIARY_SWA):
                component = cache.components.get(ct)
                if component is not None:
                    configured = max(
                        configured, component.full_window_pages * cache.page_size
                    )
        return configured

    def _release_pending_storage_ops(self) -> None:
        """Release the host pages and locks that still-tracked ops hold.

        A safety net only: the scheduler refuses a detach unless `ongoing_prefetch`
        and `ongoing_backup` are already empty, so this normally finds nothing. It
        earns its keep on the atexit path, which has no such guard.

        Two deliberate departures from `release_aborted_request`, which does the
        same bookkeeping while the storage threads are still running:

        * no cross-rank barrier -- at process exit the other ranks may already be
          gone, and a barrier there would hang;
        * only the completed prefix of a prefetch is freed. The prefetch IO thread
          owns ``host_indices[completed_tokens:]`` and frees it as it drains, so
          freeing the whole range here would double-free whatever it already
          returned. Leaking the tail of an interrupted prefetch is the safer
          failure of the two.
        """
        cache = self._cache
        controller = cache.cache_controller

        for handle in list(cache.ongoing_prefetch):
            info = cache.ongoing_prefetch[handle]
            try:
                cache.discard_storage_prefetch_accounting(handle)
                if info.host_indices is None:
                    # Host pages were never allocated for this operation.
                    cache.revoke_pending_prefetch(handle)
                    continue
                completed_tokens, _ = controller.terminate_prefetch(info.operation)
                cache._retire_ongoing_prefetch(
                    handle,
                    info.host_indices[:completed_tokens],
                    (
                        [x for xfers in info.comp_xfers.values() for x in xfers]
                        if info.operation.pool_transfers_done
                        else None
                    ),
                )
            except Exception:
                logger.exception("Failed to release pending prefetch %s", handle.rid)
                cache.ongoing_prefetch.pop(handle, None)

        for ack_id in list(cache.ongoing_backup):
            node_id, lock_params = cache.ongoing_backup.pop(ack_id)
            try:
                cache.dec_host_lock_ref(node_id, lock_params)
            except Exception:
                logger.exception("Failed to release host lock for backup op %s", ack_id)

        for handle in list(cache._storage_prefetch_hit_remaining_by_reqid):
            cache.discard_storage_prefetch_accounting(handle)
        cache.prefetch_loaded_tokens_by_reqid.clear()
        cache.prefetch_loaded_storage_start_by_reqid.clear()
        cache.storage_prefetch_retries.clear()
        cache._storage_prefetch_wait.clear()



class MambaComponent:
    def _alloc_mamba_slot(self, req: Optional[Req] = None) -> torch.Tensor:
        """Allocate one mamba pool slot, evicting if necessary."""
        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            if req is not None and get_exec().mamba.enable_mamba_extra_buffer_lazy:
                lock = req.lock
                ct = self.component_type
                if (
                    lock is not None
                    and not lock.swa_released
                    and ct not in lock.receipt.skipped_lock_components
                ):
                    # Prefill has consumed the COW source; retain only its KV locks.
                    self.cache.dec_lock_ref(
                        lock.node,
                        replace(
                            lock.receipt,
                            skipped_lock_components=tuple(
                                c for c in self.cache.tree_components if c != ct
                            ),
                        ),
                    )
                    lock.receipt.skipped_lock_components += (ct,)
            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            assert slot is not None, "Can not alloc mamba cache"
        return slot

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        if self.cache.enable_mamba_extra_buffer:
            cache_len = req.kv.mamba_last_track_seqlen
        else:
            cache_len = token_ids_len
            # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by the
            # slot's unflushed ring depth (`write_pos`), so on request finish cap
            # the donate to the last flush boundary (where temporal is current)
            # and reset the cursor, keeping the donated checkpoint consistent with
            # its key length. page_size is asserted == 1, so no realign.
            if is_finished:
                write_pos_buf = (
                    self.cache.req_to_token_pool.mamba_pool.replayssm_write_pos
                )
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.kv.mamba_pool_idx].item())
                    write_pos_buf[req.kv.mamba_pool_idx] = 0

        if is_finished:
            if cache_len is None:
                cache_len = 0
            if self.cache.enable_mamba_extra_buffer:
                checkpoint = self._select_finished_checkpoint(req, token_ids_len)
                if checkpoint is None:
                    return 0
                cache_len, keep_idx = checkpoint
                insert_params.mamba_keep_idx = keep_idx
                active_value = (
                    req.kv.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                active_value = req.kv.mamba_pool_idx.unsqueeze(-1).clone()
            # fork-only int8 checkpoint pool; 1be97a0's MambaComponent has none.
            if getattr(self, "int8_ckpt_pool", None) is not None:
                insert_params.mamba_value = self._commit_int8_checkpoint(active_value)
            else:
                insert_params.mamba_value = active_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            # Donate the mamba index to the radix cache instead of copying.
            if getattr(self, "int8_ckpt_pool", None) is not None:
                if self.cache.enable_mamba_extra_buffer:
                    new_slot = self._alloc_mamba_slot(req)
                    src_active = (
                        self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                            req, new_slot
                        )
                    )
                    mamba_value_donated = self._commit_int8_checkpoint(src_active)
                    self.cache.req_to_token_pool.mamba_allocator.free(src_active)
                else:
                    mamba_value_donated = self._commit_int8_checkpoint(
                        req.kv.mamba_pool_idx.view(-1)
                    )
            elif self.cache.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot(req)
                mamba_value_donated = (
                    self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                        req, new_slot
                    )
                )
            else:
                mamba_value_donated = self._alloc_mamba_slot(req)
                # mamba_pool is a pure PHYSICAL store; translate both slot ids
                # virtual->physical (identity for the non-unified memory pool) first.
                translate = self.cache.req_to_token_pool.translate_mamba_indices
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    translate(req.kv.mamba_pool_idx.unsqueeze(0)),
                    translate(mamba_value_donated),
                )
            insert_params.mamba_value = mamba_value_donated
            return cache_len

    def build_external_linker_transfer(
        self,
        phase: LinkerTransferPhase,
        node: Optional[UnifiedTreeNode],
        keys: Optional[Sequence[str]],
    ) -> Optional[PoolTransfer]:
        if phase == LinkerTransferPhase.OFFLOAD:
            if node is None or not node.hash_value:
                return None
            value = node.component_data[self.component_type].value
            if value is None:
                return None
            # One state per node, valid only at the node's end boundary.
            return PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=value,
                keys=[node.hash_value[-1]],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )

        if not keys:
            return None

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=[keys[-1]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        if phase == LinkerTransferPhase.LOAD:
            transfer.device_indices = self._alloc_mamba_slot()
        return transfer

    def update_external_linker_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        full_transfer: PoolTransfer,
        transfer: PoolTransfer,
        prefix_len: int,
    ) -> Optional[PoolTransfer]:
        pool = self.cache.req_to_token_pool
        if phase == ExternalLinkerLoadPhase.ABORT:
            pool.mamba_allocator.free(transfer.device_indices)
            return None

        assert phase == ExternalLinkerLoadPhase.PREPARE
        if req.kv.holds_mamba:
            pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        # The layer-gated load writes the request's private state directly.
        req.kv.mamba_pool_idx = transfer.device_indices[0]
        self._adopt_loaded_state(req)
        return transfer

    def _adopt_loaded_state(self, req: Req) -> None:
        # A loaded state supersedes the replay cursor and the deferred CoW/clear.
        pool = self.cache.req_to_token_pool
        write_pos = pool.mamba_pool.replayssm_write_pos
        if write_pos is not None:
            slot = req.kv.mamba_pool_idx.unsqueeze(0)
            write_pos[pool.translate_mamba_indices(slot)] = 0
        req.kv.mamba_cow_src_index = None
        req.kv.mamba_needs_clear = False



class MambaBufferLoadBackContext:
    def finalize_allocation(self, success: bool) -> None:
        req = self.req
        if not success:
            self.component.finalize_load_back(req, self.prep, success=False)
            return
        self.component._adopt_loaded_state(req)



class UnifiedCacheLinkerWrapper:
    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full, SWA and Mamba tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        swa = cache.components.get(ComponentType.SWA)
        self._skip_swa = swa is not None and is_swa_req_ring(
            cache.token_to_kv_pool_allocator
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        self.host_prefetch_hits: dict[str, ExternalCacheHitMarker] = {}
        self.pending_host_prefetch_submissions: dict[
            str, tuple[PreparedHostPrefetch, ExternalCacheHitMarker]
        ] = {}
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        self.inflight_load_index = -1
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

class MooncakeGlue:
    def _get_mooncake_client_http_port(dp_rank: Optional[int] = None) -> int:
        """Return a metrics port unique to this physical scheduler process."""
        base_port = envs.MOONCAKE_CLIENT_METRICS_PORT_BASE.get()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
            if dp_rank is not None:
                global_rank += dp_rank * torch.distributed.get_world_size()
            return base_port + global_rank
        return base_port
    def _get_mooncake_client_http_setup_kwargs(
        enable_client_http_server: bool, dp_rank: Optional[int] = None
    ) -> dict[str, Any]:
        if not enable_client_http_server:
            return {}
        return {
            "enable_client_http_server": True,
            "client_http_port": _get_mooncake_client_http_port(dp_rank),
        }

class MooncakeStoreConfig:
    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.get()
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        if (
            "master_server_address" not in config
            and "client_server_address" not in config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in config file"
            )

        return MooncakeStoreConfig(
            local_hostname=MooncakeStoreConfig._resolve_local_hostname(config),
            metadata_server=config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            # Fall back to MOONCAKE_PROTOCOL (not its default) so an image or
            # deployment that sets it, e.g. MOONCAKE_PROTOCOL=efa, is honored
            # when the config file omits "protocol".
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.get()),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
            standalone_storage=config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
            tenant_id=_normalize_tenant_id(
                config.get("tenant_id", envs.MOONCAKE_TENANT_ID.default)
            ),
        )
    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """Load config from extra_config dictionary."""
        if (
            "master_server_address" not in extra_config
            and "client_server_address" not in extra_config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in extra_config"
            )

        return MooncakeStoreConfig(
            local_hostname=MooncakeStoreConfig._resolve_local_hostname(extra_config),
            metadata_server=extra_config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            # See from_file(): honor MOONCAKE_PROTOCOL when "protocol" is omitted.
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.get()),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=extra_config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=extra_config.get(
                "check_server", envs.MOONCAKE_CHECK_SERVER.default
            ),
            standalone_storage=extra_config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=extra_config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=extra_config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=extra_config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
            tenant_id=_normalize_tenant_id(
                extra_config.get("tenant_id", envs.MOONCAKE_TENANT_ID.default)
            ),
        )

class MooncakeBaseStore:
    def register_buffer(self, tensor: torch.Tensor):
        if self.store is None:
            raise RuntimeError("Mooncake store is not initialized.")
        if isinstance(tensor, (tuple, list)):
            for buffer in tensor:
                self.register_buffer(buffer)
            return

        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        if size == 0:
            # conv-only models have a 0-element ssm state; nothing to register
            return
        ret_code = self.store.register_buffer(ptr, size)
        if ret_code != 0:
            logger.error(f"Failed to register buffer, error code: {ret_code}")
            raise RuntimeError(
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
            )

class MooncakeStore:
    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool: HostKVCache = None,
        enable_client_http_server: Optional[bool] = None,
    ):
        MooncakeBaseStore.__init__(self)
        MooncakeDistributedStore = self._import_mooncake_store()
        self._replicate_config_cls, self._supports_group_ids = (
            self._import_mooncake_group_semantics()
        )
        try:
            self.store = MooncakeDistributedStore()

            self.config = self._load_config(storage_config)
            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            if enable_client_http_server is None:
                enable_client_http_server = bool(
                    getattr(storage_config, "enable_storage_metrics", False)
                )
            self.enable_group_semantics = bool(
                extra_config.get("enable_group_semantics", False)
                if extra_config
                else False
            )
            self._use_group_semantics = (
                self.enable_group_semantics
                and self._supports_group_ids
                and self._replicate_config_cls is not None
            )
            if self.enable_group_semantics and not self._supports_group_ids:
                logger.warning(
                    "Mooncake group semantics is enabled, but the installed "
                    "Mooncake package does not support ReplicateConfig.group_ids. "
                    "Falling back to the existing batch_put_from path."
                )
            rank_scale_factor = (
                1
                if storage_config is None
                else (storage_config.tp_size * storage_config.pp_size)
            )

            per_rank_global_segment_size = (
                self.config.global_segment_size // rank_scale_factor
            )

            # Use the backend tag and model name as a prefix to isolate tenants
            # and models sharing one store.
            self.config_prefix = None
            config_prefix_parts = []
            if extra_config and extra_config.get("extra_backend_tag") is not None:
                config_prefix_parts.append(str(extra_config["extra_backend_tag"]))
            if storage_config is not None and storage_config.model_name:
                model_name = "-".join(storage_config.model_name.split("/"))
                config_prefix_parts.append(model_name)
            if config_prefix_parts:
                self.config_prefix = "_".join(config_prefix_parts)
                logger.info(f"Using Mooncake config prefix: {self.config_prefix}")

            # Check server status
            if self.config.check_server:
                self.check_server()

            # Handle JSON device_name configuration
            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        # Try both integer and string keys since JSON parsing may convert keys
                        device_name = device_config.get(tp_rank, "")
                        if not device_name:
                            device_name = device_config.get(str(tp_rank), "")
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"Failed to parse device_name as JSON: {device_name}"
                    )
                    device_name = ""
            if self.config.standalone_storage:
                if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):
                    raise RuntimeError(
                        "MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator. "
                        "Please set standalone_storage=False "
                        "or upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'."
                    )
                required_bytes = self._standalone_required_bytes(mem_pool)
                ret_code = self.store.setup_dummy(
                    required_bytes,
                    DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                    self.config.client_server_address,
                )
            else:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_mooncake_transfer_engine,
                    )

                    self._shared_mooncake_transfer_engine = (
                        get_mooncake_transfer_engine()
                    )
                except Exception:
                    self._shared_mooncake_transfer_engine = None
                    logger.debug("Failed to reuse initialized mooncake transfer engine")

                # Only reuse the shared MooncakeTransferEngine when its
                # configuration matches the one used by MooncakeStore.
                if (
                    self._shared_mooncake_transfer_engine is not None
                    and device_name
                    == self._shared_mooncake_transfer_engine.get_ib_device()
                    and self.config.metadata_server == "P2PHANDSHAKE"
                    and self.config.protocol == "rdma"
                ):
                    client_hostname = (
                        self._shared_mooncake_transfer_engine.get_session_id()
                    )
                    transfer_engine = self._shared_mooncake_transfer_engine.get_engine()
                    logger.info(
                        f"Reuse initialized mooncake transfer engine: {self._shared_mooncake_transfer_engine}"
                    )
                else:
                    client_hostname = self.config.local_hostname
                    transfer_engine = None

                setup_kwargs = _get_mooncake_client_http_setup_kwargs(
                    enable_client_http_server, getattr(storage_config, "dp_rank", None)
                )
                if self.config.enable_ssd_offload:
                    setup_kwargs["enable_ssd_offload"] = True
                if self.config.ssd_offload_path is not None:
                    # Each rank embeds its own Mooncake client. Sharing one
                    # offload directory corrupts silently: bucket ids are
                    # generated per process and resumed from the same startup
                    # scan after a restart, and bucket files are opened with
                    # O_CREAT|O_TRUNC, so a filename collision truncates
                    # another rank's bucket. Give every rank a private subdir.
                    ssd_offload_path = self.config.ssd_offload_path
                    if storage_config is not None:
                        ssd_offload_path = os.path.join(
                            ssd_offload_path,
                            f"rank_{storage_config.dp_rank}"
                            f"_{storage_config.tp_rank}_{storage_config.pp_rank}"
                            f"_{storage_config.attn_cp_rank}",
                        )
                    os.makedirs(ssd_offload_path, exist_ok=True)
                    setup_kwargs["ssd_offload_path"] = ssd_offload_path
                if self.config.tenant_id != DEFAULT_TENANT_ID:
                    setup_kwargs["tenant_id"] = self.config.tenant_id

                while True:
                    try:
                        ret_code = self.store.setup(
                            client_hostname,
                            self.config.metadata_server,
                            per_rank_global_segment_size,
                            DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                            self.config.protocol,
                            device_name,
                            self.config.master_server_address,
                            transfer_engine,
                            **setup_kwargs,
                        )
                        break
                    except TypeError as e:
                        unsupported_kwargs = [
                            key for key in list(setup_kwargs) if key in str(e)
                        ]
                        if not unsupported_kwargs:
                            raise
                        if "tenant_id" in unsupported_kwargs:
                            raise RuntimeError(
                                "The installed Mooncake version does not support "
                                "tenant_id in MooncakeDistributedStore.setup(). "
                                "Please upgrade Mooncake to use non-default "
                                "Mooncake tenants with SGLang."
                            ) from e
                        logger.warning(
                            "The installed Mooncake version does not support the "
                            f"{', '.join(unsupported_kwargs)} parameter(s) in setup(). "
                            f"Retrying without {', '.join(unsupported_kwargs)}. "
                            "Please upgrade Mooncake to enable SSD offload support."
                        )
                        for key in unsupported_kwargs:
                            setup_kwargs.pop(key, None)
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

            self.local_rank = (
                storage_config.tp_rank if storage_config is not None else 0
            )
            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            self.enable_storage_metrics = False
            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.pp_rank = storage_config.pp_rank
                self.pp_size = storage_config.pp_size
                self.attn_cp_rank = storage_config.attn_cp_rank
                self.attn_cp_size = storage_config.attn_cp_size
                self.enable_storage_metrics = storage_config.enable_storage_metrics
            else:
                self.is_mla_backend = False
                self.local_rank = 0
                self.pp_rank = 0
                self.pp_size = 1
                self.attn_cp_rank = 0
                self.attn_cp_size = 1

            self.enable_pp = self.pp_size > 1
            if self.enable_pp:
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
                self.mla_suffix = f"{self.pp_rank}"
            else:
                self.mha_suffix = f"{self.local_rank}"
                self.mla_suffix = ""

            self.storage_config = storage_config
            self.should_split_heads = storage_config.should_split_heads
            self.split_factor = 0
            if self.should_split_heads:
                self.split_factor = (
                    self.storage_config.tp_lcm_size // self.storage_config.tp_size
                )
                base_rank = self.local_rank * self.split_factor
                target_ranks = [base_rank + i for i in range(self.split_factor)]
                if self.enable_pp:
                    self.mha_suffix = [
                        f"{rank}_{self.pp_rank}" for rank in target_ranks
                    ]
                else:
                    self.mha_suffix = [f"{rank}" for rank in target_ranks]

            self.registered_pools = {}

            self.gb_per_page = None
            self.prefetch_pgs = []
            self.backup_pgs = []
            self.prefetch_bandwidth = []
            self.backup_bandwidth = []

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise
    def _get_hybrid_page_component_keys(
        self, page_keys: List[str], transfer: PoolTransfer
    ) -> Tuple[List[str], int]:
        host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
        if host_pool is None:
            raise ValueError(f"Unregistered Mooncake hybrid pool: {transfer.name}")

        if isinstance(host_pool, DevicePoolEntry):
            # A direct-linker entry packs every buffer of a page into one object.
            object_name = "k" if transfer.name == PoolName.KV else transfer.name
            suffix = f"_{self.mla_suffix}_{object_name}"
            return [f"{page_key}{suffix}" for page_key in page_keys], 1

        # Suffix order must match get_page_buffer_meta() for one page, because
        # Mooncake zips object keys with registered buffer pointers.
        pool_name = transfer.name
        suffixes = []
        if pool_name == PoolName.KV:
            suffixes = [f"_{self.mla_suffix}_k"]
        elif pool_name == PoolName.MAMBA:
            # Mamba stores one temporal object plus one object per conv state.
            # conv-only models have no ssm state; drop the 0-element temporal
            # object (mooncake rejects 0-size puts). get_page_buffer_meta drops
            # its temporal pointer under the same condition to stay aligned.
            conv_num = len(getattr(host_pool, "conv_buffer", None) or [])
            suffixes = [f"_{self.mha_suffix}_conv_{i}" for i in range(conv_num)]
            if getattr(host_pool, "temporal_state_elem_size", 1) > 0:
                suffixes = [f"_{self.mha_suffix}_temporal"] + suffixes
        elif pool_name == PoolName.DRAFT:
            # Draft pool's MLA/MHA layout is independent from the target
            # (e.g. EAGLE-MHA draft on top of an MLA target), so pick the
            # suffix scheme from the draft pool's own class. The `_draft`
            # tag is what keeps these keys from colliding with target's
            # `{rank}_k` / `{rank}_k` + `{rank}_v` keys.
            draft_pool = self.registered_pools.get(PoolName.DRAFT)
            if isinstance(draft_pool, MLATokenToKVPoolHost):
                suffixes = [f"_{self.mla_suffix}_{PoolName.DRAFT}_k"]
            else:
                suffixes = [
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_k",
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_v",
                ]
        elif pool_name == PoolName.DRAFT_SWA:
            from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
            from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

            if isinstance(
                host_pool,
                (DeepSeekV4PagedHostPool, MLATokenToKVPoolHost),
            ):
                suffixes = [f"_{self.mla_suffix}_{pool_name}"]
            elif isinstance(host_pool, MHATokenToKVPoolHost):
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
        elif pool_name in (
            PoolName.INDEXER,
            PoolName.DRAFT_INDEXER,
            PoolName.DEEPSEEK_V4_C1,
            PoolName.DEEPSEEK_V4_C1_INDEXER,
            PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE,
            PoolName.DEEPSEEK_V4_C2,
            PoolName.DEEPSEEK_V4_C2_INDEXER,
            PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE,
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_ROPE,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.DEEPSEEK_V4_C128_ROPE,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            # DSA indexer and DeepSeek V4 side pools are page-packed
            # single-object pools.
            suffixes = [f"_{self.mla_suffix}_{pool_name}"]
        elif pool_name == PoolName.SWA:
            if not self.is_mla_backend and hasattr(host_pool, "v_buffer"):
                # Ordinary MHA SWA mirrors a K/V pool.
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
            elif self.is_mla_backend:
                suffixes = [f"_{self.mla_suffix}_{pool_name}"]

        if not suffixes:
            raise ValueError(
                f"Unsupported Mooncake hybrid pool name: {pool_name}, "
                f"host_pool={type(host_pool)}"
            )
        key_multiplier = len(suffixes)
        component_keys = [
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes
        ]
        return component_keys, key_multiplier
