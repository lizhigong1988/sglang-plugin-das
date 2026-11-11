# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dual-source re-export of the unified-KV env gate.

1be97a0 moved ``env_gate`` from
``sglang.kernels.ops.attention.dsv4.unified_kv_kernels`` to
``sglang.srt.layers.attention.dsv4.unified_kv_gate``; keep the fork's
kernels path as the fallback so both upstream pins import cleanly.
"""

from __future__ import annotations

try:
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_fp8,
        is_unified_kv_triton,
    )
except ImportError:
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_fp8,
        is_unified_kv_triton,
    )

__all__ = ["is_unified_kv_fp8", "is_unified_kv_triton"]
