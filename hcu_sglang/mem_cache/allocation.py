"""HCU request-table and paged-KV allocation hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


_LIGHTOP_ALLOC_EXTEND_KERNEL = None
_LIGHTOP_ALLOC_EXTEND_KERNEL_CHECKED = False


def _get_lightop_alloc_extend_kernel():
    global _LIGHTOP_ALLOC_EXTEND_KERNEL, _LIGHTOP_ALLOC_EXTEND_KERNEL_CHECKED

    if _LIGHTOP_ALLOC_EXTEND_KERNEL_CHECKED:
        return _LIGHTOP_ALLOC_EXTEND_KERNEL

    _LIGHTOP_ALLOC_EXTEND_KERNEL_CHECKED = True
    try:
        from lightop import op as lightop_op

        _LIGHTOP_ALLOC_EXTEND_KERNEL = getattr(
            lightop_op, "hcu_alloc_extend_kernel", None
        )
    except Exception:
        _LIGHTOP_ALLOC_EXTEND_KERNEL = None

    return _LIGHTOP_ALLOC_EXTEND_KERNEL


def _enabled(name: str) -> bool:
    from sglang.srt.utils import get_bool_env_var

    return get_bool_env_var(name, default="true")


def _load_kvcacheio():
    from sgl_kernel import kvcacheio

    return kvcacheio


def around_get_last_loc(
    original: Callable[..., torch.Tensor],
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if not is_hcu() or not _enabled("SGLANG_GET_LAST_LOC"):
        return original(
            req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
        )
    return _load_kvcacheio().hcu_get_last_loc(
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
    )


def around_assign_req_to_token_pool(
    original: Callable[..., Any],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> Any:
    if not is_hcu() or not _enabled("SGLANG_ASSIGN_REQ_TO_TOKEN_POOL"):
        return original(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        )
    _load_kvcacheio().hcu_assign_req_to_token_pool(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        allocate_lens=start_offset,
        new_allocate_lens=end_offset,
        out_cache_loc=out_cache_loc,
        shape=req_to_token.shape[1],
        bs=batch_size,
    )
    return None


def after_paged_allocator_init(result: Any, allocator: Any, *args, **kwargs) -> Any:
    if is_hcu():
        from sglang.srt.environ import envs

        allocator.sglang_kvalloc_kernel = _enabled("SGLANG_KVALLOC_KERNEL")
        allocator.lightop_kvalloc_kernel = envs.SGLANG_LIGHTOP_KVALLOC_KERNEL.get()
    return result


def around_alloc_extend(
    original: Callable[..., torch.Tensor | None],
    allocator: Any,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    num_new_pages: int | None = None,
) -> torch.Tensor | None:
    if not is_hcu() or not getattr(allocator, "sglang_kvalloc_kernel", False):
        return original(
            allocator,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages,
        )

    # 1be97a0 moved it to mem_cache.utils.
    try:
        from sglang.srt.mem_cache.utils import get_num_new_pages
    except ImportError:
        from sglang.srt.utils import get_num_new_pages

    if allocator.debug_mode:
        assert torch.all(
            (last_loc + 1) % allocator.page_size == prefix_lens % allocator.page_size
        )

    batch_size = len(prefix_lens)
    max_new_pages = extend_num_tokens // allocator.page_size + batch_size + 1
    if max_new_pages > len(allocator.free_pages):
        allocator.merge_and_sort_free()
    if max_new_pages > len(allocator.free_pages):
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=allocator.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(allocator.free_pages):
            return None

    out_indices = torch.empty(
        (extend_num_tokens,),
        dtype=torch.int64,
        device=allocator.device,
    )
    kernel = None
    if getattr(allocator, "lightop_kvalloc_kernel", False):
        kernel = _get_lightop_alloc_extend_kernel()
    if kernel is None:
        kernel = _load_kvcacheio().hcu_alloc_extend_kernel
    kernel(
        pre_lens_ptr=prefix_lens.to(torch.int64),
        seq_lens_ptr=seq_lens.to(torch.int64),
        last_loc_ptr=last_loc.to(torch.int64),
        free_page_ptr=allocator.free_pages.to(torch.int64),
        out_indices=out_indices,
        bs=batch_size,
        page_size=allocator.page_size,
    )

    if allocator.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    if num_new_pages is None:
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=allocator.page_size,
            prefix_lens=prefix_lens_cpu,
        )
    if num_new_pages > len(allocator.free_pages):
        return None

    allocator.free_pages = allocator.free_pages[num_new_pages:]
    return out_indices


def around_alloc_decode(
    original: Callable[..., torch.Tensor | None],
    allocator: Any,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
) -> torch.Tensor | None:
    if not is_hcu() or not getattr(allocator, "sglang_kvalloc_kernel", False):
        return original(
            allocator,
            seq_lens,
            seq_lens_cpu,
            last_loc,
        )

    # 1be97a0 moved it to mem_cache.utils.
    try:
        from sglang.srt.mem_cache.utils import get_num_new_pages
    except ImportError:
        from sglang.srt.utils import get_num_new_pages

    if allocator.debug_mode:
        assert torch.all(
            (last_loc + 2) % allocator.page_size == seq_lens % allocator.page_size
        )

    batch_size = len(seq_lens)
    if batch_size > len(allocator.free_pages):
        allocator.merge_and_sort_free()

    num_new_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=allocator.page_size,
        decode=True,
    )
    if num_new_pages > len(allocator.free_pages):
        return None

    out_indices = torch.empty(
        (batch_size,),
        dtype=torch.int64,
        device=allocator.device,
    )
    _load_kvcacheio().hcu_alloc_decode_kernel(
        seq_lens_ptr=seq_lens,
        last_loc_ptr=last_loc.to(dtype=torch.int32, copy=False),
        free_page_ptr=allocator.free_pages,
        out_indices=out_indices,
        bs=batch_size,
        page_size=allocator.page_size,
    )

    if allocator.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    allocator.free_pages = allocator.free_pages[num_new_pages:]
    return out_indices


def around_free(
    original: Callable[..., Any],
    allocator: Any,
    free_index: torch.Tensor,
) -> Any:
    if not is_hcu():
        return original(allocator, free_index)
    if free_index.numel() == 0:
        return None

    if allocator.free_group is None:
        free_page_indices = torch.unique(free_index // allocator.page_size)
        free_page_indices = free_page_indices[free_page_indices > 0]
        if free_page_indices.numel() == 0:
            return None
        free_page_indices = free_page_indices[
            ~torch.isin(free_page_indices, allocator.get_all_free_pages())
        ]
        if free_page_indices.numel() == 0:
            return None
        allocator._release_page_ids(free_page_indices)
    else:
        allocator.free_group.append(allocator._copy_for_free_group(free_index))

    if allocator.debug_mode:
        allocator._debug_check_no_duplicate_pages()
    return None


def register_hooks() -> None:
    from hcu_sglang.mem_cache import mamba_pool
    mamba_pool.register_hooks()
    from hcu_sglang.mem_cache import cpu_cache
    cpu_cache.register_hooks()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.allocation.get_last_loc",
        around_get_last_loc,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func",
        around_assign_req_to_token_pool,
        HookType.AROUND,
    )
    target = "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator"
    HookRegistry.register(
        f"{target}.__init__",
        after_paged_allocator_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        f"{target}.alloc_extend",
        around_alloc_extend,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.alloc_decode",
        around_alloc_decode,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.free",
        around_free,
        HookType.AROUND,
    )
