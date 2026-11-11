from __future__ import annotations

import types
from sglang.srt.mem_cache.unified_cache import unified_tree_core as N
from hcu_sglang.utils.detection import is_hcu

class _TreeMethods:
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        ) = self._match_prefix_helper(
            key, return_full_match=params.return_full_match
        )
        _post_kwargs = {}
        # 1be97a0 added last_visited_node_id + device_validators; the plugin
        # walker keeps no last-node id, so report root-only (None) and mirror
        # the native device-validator construction.
        import inspect as _inspect

        if "last_visited_node_id" in _inspect.signature(
            self._match_post_processor
        ).parameters:
            _post_kwargs["last_visited_node_id"] = None
            _post_kwargs["device_validators"] = self._create_match_validators(
                params, match_device_only=True
            )
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
            **_post_kwargs,
        )

    def _match_prefix_helper(
        self, key: RadixKey, *, return_full_match: bool = False
    ) -> tuple[
        list[torch.Tensor],
        UnifiedTreeNode,
        UnifiedTreeNode,
        int,
        int,
        Optional[CacheAction | ComponentAction],
    ]:
        # Non-HiCache mode has only device-resident matches, so the scheduler
        # device anchor follows the best match. In HiCache mode, host-backed
        # nodes can also match, so we separately track the best device-resident
        # match for scheduler prefix indices and locking.
        node = self.root_node
        key_offset = 0
        child_key = key.child_key_at(key_offset, self.page_size)
        value: list[torch.Tensor] = []
        best_match_node = node
        best_match_device_node = node
        best_match_device_value_len = 0
        full_kv_hit_length = 0
        action: Optional[CacheAction | ComponentAction] = None
        separate_device_match = self.enable_hicache
        match_components = (
            tuple(
                component
                for component in self.components
                if component.component_type == BASE_COMPONENT_TYPE
            )
            if return_full_match
            else self.components
        )
        if separate_device_match:
            validators = tuple(
                comp.create_match_validator() for comp in match_components
            )
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in match_components
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in match_components
            )

        def _all_valid(validators, node):
            return all([v(node) for v in validators])

        def _update_best_if_valid(node):
            nonlocal best_match_node
            nonlocal best_match_device_value_len, best_match_device_node
            matched = _all_valid(validators, node)
            if matched:
                best_match_node = node

            if not separate_device_match:
                if matched:
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return
            if _all_valid(device_validators, node):
                best_match_device_value_len = len(value)
                best_match_device_node = node

        while key_offset < len(key) and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match_at(key, key_offset, page_size=self.page_size)
            full_kv_hit_length += prefix_len
            if prefix_len < len(child.key):
                node, action = self._split_node(child.key, child, prefix_len)
                if not node.evicted:
                    value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(node)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key_offset += prefix_len
            if key_offset < len(key):
                child_key = key.child_key_at(key_offset, self.page_size)

        return (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )

_REGISTERED = False


def register_hooks():
    global _REGISTERED
    if _REGISTERED or not is_hcu():
        return
    for name, raw in vars(_TreeMethods).items():
        if not isinstance(raw, types.FunctionType):
            continue
        fn = types.FunctionType(raw.__code__, vars(N), name,
                                raw.__defaults__, raw.__closure__)
        fn.__kwdefaults__ = raw.__kwdefaults__
        fn.__annotations__ = raw.__annotations__
        fn.__qualname__ = N.UnifiedTreeCore.__qualname__ + "." + name
        setattr(N.UnifiedTreeCore, name, fn)
    _REGISTERED = True
