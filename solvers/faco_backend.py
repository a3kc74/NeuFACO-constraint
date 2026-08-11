"""Import adapter for the compiled FACO backend extension."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

BACKEND_SOURCE_DIR = Path(__file__).resolve().parents[1] / "cpp" / "faco" / "src"


def import_backend():
    """Import the compiled `faco_opt` extension with a clear build error."""
    if str(BACKEND_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_SOURCE_DIR))
    try:
        return importlib.import_module("faco_opt")
    except ImportError as exc:
        raise ImportError(
            "C++ backend 'faco_opt' is not available. Build it with: "
            "uv run python cpp/faco/setup.py build_ext --inplace"
        ) from exc


def set_num_threads(n_threads: int) -> None:
    """Set FACO backend OpenMP thread count."""
    import_backend().set_num_threads(int(n_threads))


__all__ = ["BACKEND_SOURCE_DIR", "import_backend", "set_num_threads"]
