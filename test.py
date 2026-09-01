"""Unified evaluation entry point for NeuFACO constraint experiments."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

METHOD_TO_SCRIPT = {
    "faco": ROOT / "evaluation" / "faco_test.py",
    "faco_ib": ROOT / "evaluation" / "faco_test_ib.py",
    "gfacs": ROOT / "evaluation" / "gfacs_test.py",
    "macs": ROOT / "baselines" / "macs_baseline.py",
    "pyvrp": ROOT / "baselines" / "pyvrp_baseline.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified NeuFACO evaluation entry point.")
    parser.add_argument("--problem", default="CVRPTW", choices=["CVRPTW", "VRPTW"], help="Problem family.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_TO_SCRIPT), help="Evaluation method.")
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the method script.")
    args = parser.parse_args()
    forwarded_args = args.legacy_args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    if args.problem == "VRPTW" and "--vrptw" not in forwarded_args:
        forwarded_args = [*forwarded_args, "--vrptw"]
    return args, forwarded_args


def run_script(script: Path, argv: list[str]) -> None:
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    args, forwarded_args = parse_args()
    run_script(METHOD_TO_SCRIPT[args.method], forwarded_args)


if __name__ == "__main__":
    main()

