"""Canonical build wrapper for the FACO C++ extension."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = REPO_ROOT / "cpp" / "faco" / "src"


def main() -> None:
    setup_path = BACKEND_SRC / "setup.py"
    old_cwd = Path.cwd()
    old_argv = list(sys.argv)
    try:
        os.chdir(BACKEND_SRC)
        sys.argv = [str(setup_path), *old_argv[1:]]
        runpy.run_path(str(setup_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
