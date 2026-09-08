# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight loader for the CANN key_pool / pool_key_indexer wheel wrappers.

The ``cann_ops_transformer`` wheel root eagerly imports unrelated operators
(and may JIT-build them), so the two target wrapper modules are loaded
directly and registered on ``torch_npu``:

    torch_npu.key_pool(...)
    torch_npu.pool_key_indexer(...)

The matching hardware-specific ``.run`` package must be installed before the
ops are actually invoked.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch_npu


def load_key_pool_and_indexer_from_wheel() -> tuple[object, object]:
    """Load both target wrappers without importing unrelated wheel modules."""
    package_spec = importlib.util.find_spec("cann_ops_transformer")
    if package_spec is None or package_spec.origin is None:
        raise ImportError("cann_ops_transformer wheel is not installed")

    package_root = Path(package_spec.origin).resolve().parent

    # Use a lightweight package namespace so the wheel root __init__.py does
    # not eagerly import (and JIT-build) unrelated operators.
    package = sys.modules.get("cann_ops_transformer")
    if package is None or not hasattr(package, "__path__"):
        package = types.ModuleType("cann_ops_transformer")
        package.__file__ = str(package_root / "__init__.py")
        package.__package__ = "cann_ops_transformer"
        package.__path__ = [str(package_root)]
        sys.modules["cann_ops_transformer"] = package

    ops = sys.modules.get("cann_ops_transformer.ops")
    if ops is None or not hasattr(ops, "__path__"):
        ops = types.ModuleType("cann_ops_transformer.ops")
        ops.__file__ = str(package_root / "ops" / "__init__.py")
        ops.__package__ = "cann_ops_transformer.ops"
        ops.__path__ = [str(package_root / "ops")]
        sys.modules["cann_ops_transformer.ops"] = ops
        package.ops = ops

    key_pool_module = importlib.import_module("cann_ops_transformer.ops.key_pool")
    indexer_module = importlib.import_module(
        "cann_ops_transformer.ops.pool_key_indexer"
    )

    torch_npu.key_pool = key_pool_module.key_pool
    torch_npu.pool_key_indexer = indexer_module.pool_key_indexer
    return torch_npu.key_pool, torch_npu.pool_key_indexer


def register_cann_kpool_ops() -> bool:
    """Register ``torch_npu.key_pool`` / ``torch_npu.pool_key_indexer``.

    Returns True when the wheel is installed and both ops were registered;
    returns False (no-op) when the wheel is missing so vLLM can still import.
    """
    try:
        load_key_pool_and_indexer_from_wheel()
    except ImportError:
        return False
    return True


register_cann_kpool_ops()
