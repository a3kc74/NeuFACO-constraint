"""Unified training entry point for NeuFACO constraint experiments."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

METHOD_TO_SCRIPT = {
    "dynaco_ppo": ROOT / "trainers" / "dynaco_ppo_trainer.py",
    "ppo": ROOT / "trainers" / "ppo_trainer.py",
    "gfacs": ROOT / "trainers" / "gfacs_trainer.py",
    "deepaco": ROOT / "trainers" / "deepaco_trainer.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified NeuFACO training entry point.")
    parser.add_argument("--problem", default="CVRPTW", choices=["CVRPTW", "VRPTW"], help="Problem family.")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_TO_SCRIPT), help="Training method.")
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the method script.")
    args = parser.parse_args()
    forwarded_args = args.legacy_args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    if args.problem == "VRPTW" and "--vrptw" not in forwarded_args and args.method in {"gfacs", "deepaco"}:
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
