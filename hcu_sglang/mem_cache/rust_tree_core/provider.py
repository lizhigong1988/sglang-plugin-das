from __future__ import annotations

import hashlib
from pathlib import Path
import shlex
import subprocess
import sys

from hcu_sglang.utils.detection import is_hcu

_NATIVE_MODULE = 'sglang.srt.mem_cache.rust_tree_core.mem_cache'
_OWNED_MODULE = 'hcu_sglang.mem_cache.rust_tree_core.mem_cache'
_OWNED_INSPECTION = 'hcu_sglang.mem_cache.rust_tree_core.mem_cache_inspection'
_CRATE = Path(__file__).resolve().parent / 'crate'


def around_load(original, python_module, **kwargs):
    if not is_hcu() or python_module != _NATIVE_MODULE or kwargs.get('workspace') is not None:
        return original(python_module, **kwargs)
    extension = kwargs.get('extension_module')
    if extension not in (None, _NATIVE_MODULE, 'sglang.srt.mem_cache.rust_tree_core.mem_cache_inspection'):
        return original(python_module, **kwargs)
    owned_module = _OWNED_INSPECTION if extension and extension.endswith('_inspection') else _OWNED_MODULE
    header = _CRATE / 'torch_compat.h'
    if not kwargs.get('build_environment'):
        from sglang.srt.rust_extensions.torch_build import torch_build_configuration
        build = torch_build_configuration(compat_header=header, python_module=owned_module)
        kwargs['build_environment'] = build.environment
        fingerprint = dict(build.fingerprint)
    else:
        environment = dict(kwargs['build_environment'])
        flags = shlex.split(environment.get('CXXFLAGS', ''))
        updated = False
        for i, flag in enumerate(flags[:-1]):
            if flag == '-include' and Path(flags[i + 1]).name == 'torch_compat.h':
                flags[i + 1] = str(header)
                updated = True
        if not updated:
            flags.extend(('-include', str(header)))
        environment['CXXFLAGS'] = shlex.join(flags)
        native_rpath = '$ORIGIN/' + '../' * (len(python_module.split('.')) - 1) + 'torch/lib'
        owned_rpath = '$ORIGIN/' + '../' * (len(owned_module.split('.')) - 1) + 'torch/lib'
        environment['RUSTFLAGS'] = environment.get('RUSTFLAGS', '').replace(native_rpath, owned_rpath)
        kwargs['build_environment'] = environment
        fingerprint = dict(kwargs.get('build_fingerprint') or {})
    fingerprint.update(dict(kwargs.get('build_fingerprint') or {}))
    fingerprint['compat_header_sha256'] = hashlib.sha256(header.read_bytes()).hexdigest()
    kwargs['build_fingerprint'] = fingerprint
    kwargs['workspace'] = _CRATE
    kwargs['extension_module'] = owned_module
    return original(python_module, **kwargs)


def _cargo_builds_edition2024() -> bool:
    """The owned crate is edition 2024 (Rust >= 1.85); dtk-era images ship
    cargo 1.75, which refuses the manifest outright."""
    try:
        out = subprocess.run(
            ("cargo", "--version"),
            cwd=_CRATE, check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    head = out.strip().split(maxsplit=1)
    if len(head) < 2:
        return False
    try:
        major = int(head[1].split(".")[0])
    except ValueError:
        return False
    return major >= 85


def around_fallback_reason(original, params):
    reason = original(params)
    if not is_hcu() or reason is None:
        return reason
    try:
        from sglang.srt.environ import envs as _envs

        if _envs.SGLANG_RUST_BUILD_MODE.get() == "never":
            return reason  # explicit opt-out wins over the owned crate
    except AttributeError:
        pass
    availability = reason in (
        "the requested Rust TreeCore source build has no sources",
        "this installation contains neither the Rust TreeCore extension nor its sources",
    ) or reason.startswith("the Rust TreeCore toolchain is unavailable (")
    if not availability:
        return reason
    if any(sys.modules.get(name) is not None for name in (_OWNED_MODULE, _OWNED_INSPECTION)):
        return None
    if not (_CRATE / 'Cargo.toml').is_file():
        return reason
    for command in (("cargo", "--version", "--verbose"), ("rustc", "-vV")):
        try:
            subprocess.run(command, cwd=_CRATE, check=True, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return reason
    if not _cargo_builds_edition2024():
        return reason  # keep the Python TreeCore fallback on stale toolchains
    return None


def attach_event_recorder(core, params):
    from hcu_sglang.mem_cache.rust_tree_core.events import _RustKVCacheEventRecorder
    modules = (sys.modules.get(_OWNED_MODULE), sys.modules.get(_OWNED_INSPECTION))
    if not any(module is not None and core._bindings is module for module in modules):
        raise RuntimeError('Rust TreeCore was loaded before the HCU provider; start a new process before selecting the backend')
    core.kv_events = _RustKVCacheEventRecorder(
        core._binding, enabled=params.enable_kv_cache_events,
        lora_names=core.kv_events.lora_names,
        dynamo_format=params.dynamo_kv_event_format,
    )


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register('sglang.srt.rust_extensions.loader.load_rust_extension', around_load, HookType.AROUND)
    HookRegistry.register('sglang.srt.mem_cache.unified_cache.tree_core_registry._rust_fallback_reason', around_fallback_reason, HookType.AROUND)
