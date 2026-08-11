"""Unified data-generation entry point for NeuFACO constraint experiments."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Generate NeuFACO CVRPTW/VRPTW datasets.")
    parser.add_argument("--problem", default="CVRPTW", choices=["CVRPTW", "VRPTW"], help="Problem family.")
    parser.add_argument("--format", default="gfacs", choices=["gfacs"], help="Dataset format to generate.")
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the generator.")
    args = parser.parse_args()
    forwarded_args = args.legacy_args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    if args.problem == "VRPTW" and "--vrptw" not in forwarded_args:
        forwarded_args = [*forwarded_args, "--vrptw"]
    return args, forwarded_args


def main() -> None:
    _, forwarded_args = parse_args()
    script = ROOT / "envs" / "gfacs_data.py"
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(script), *forwarded_args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
