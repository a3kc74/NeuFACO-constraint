from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


BASE_ARGS = [
    "50",
    "--threads", "8",
    "--batch_size", "1",
    "--val_size", "16",
    "--val_ants", "100",
    "--min_new_edges", "12",
    "--val_n_iter", "10",
    "--val_mini_H", "10",
    "--smooth_mmas",
    "--extend_ls",
    "--use_local_search",
    "--epochs", "40",
    "--steps", "32",
    "--val_infer_mode", "faco_test",
    "--select_metric_mode", "faco_test",
    "--ants", "32",
    "--train_H", "1",
    "--train_mini_H", "10",
    "--ppo_clip", "0.1",
    "--parallel_traced",
    "--disable_wandb",
]


@dataclass(frozen=True)
class VersionSpec:
    version: str
    description: str
    extra_args: tuple[str, ...]
    runnable: bool = True
    note: str = ""


VERSIONS = [
    VersionSpec("V0", "Exact baseline", ("--run_name", "v0_exact_baseline")),
    VersionSpec("V1", "Pheromone decay 0.5", ("--decay", "0.5", "--run_name", "v1_decay05")),
    VersionSpec("V2", "PPO epochs 2", ("--decay", "0.5", "--ppo_epochs", "2", "--run_name", "v2_ppoep2")),
    VersionSpec("V3", "Entropy + tighter grad clip", ("--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--run_name", "v3_reg")),
    VersionSpec("V4", "Replay-state correctness fix control", ("--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--run_name", "v4_replayfix_h1s10")),
    VersionSpec("V5", "Dynamic H=2/S=5", ("--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--train_H", "2", "--train_mini_H", "5", "--run_name", "v5_dynamic_h2s5")),
    VersionSpec("V5B", "Dynamic H=5/S=2", ("--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--train_H", "5", "--train_mini_H", "2", "--run_name", "v5b_dynamic_h5s2")),
    VersionSpec("V6", "Macro-state relative advantage + target KL", ("--decay", "0.5", "--ppo_epochs", "4", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--train_H", "2", "--train_mini_H", "5", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--target_kl", "0.01", "--run_name", "v6_macroadv_targetkl")),
    VersionSpec("V7", "True PPO batch of four instances", ("--batch_size", "4", "--steps", "8", "--decay", "0.5", "--ppo_epochs", "4", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--train_H", "2", "--train_mini_H", "5", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--target_kl", "0.01", "--run_name", "v7_truebatch4")),
    VersionSpec("V8", "Cosine LR + EMA checkpoints", ("--batch_size", "4", "--steps", "8", "--decay", "0.5", "--ppo_epochs", "4", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--train_H", "2", "--train_mini_H", "5", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--target_kl", "0.01", "--lr_scheduler", "cosine", "--ema_decay", "0.995", "--run_name", "v8_cosine_ema")),
    VersionSpec("V9", "LayerNorm network", ("--batch_size", "4", "--steps", "8", "--decay", "0.5", "--ppo_epochs", "4", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--train_H", "2", "--train_mini_H", "5", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--target_kl", "0.01", "--lr_scheduler", "cosine", "--ema_decay", "0.995", "--norm_type", "layer", "--run_name", "v9_layernorm")),
    VersionSpec("V10", "Depot action as policy action", (), False, "Needs C++ action trace/action-space changes before it can be trained."),
    VersionSpec("V11A", "Capacity-aware dynamic features", (), False, "Needs backend exposure of route load/capacity state."),
    VersionSpec("V11B", "Time-window-aware dynamic features", (), False, "Needs backend exposure of arrival/slack/time-window state."),
]


def replace_arg(args: list[str], key: str, value: str) -> list[str]:
    args = list(args)
    if key in args:
        idx = args.index(key)
        args[idx + 1] = value
    else:
        args.extend([key, value])
    return args


def remove_arg_with_value(args: list[str], key: str) -> list[str]:
    out = []
    idx = 0
    while idx < len(args):
        if args[idx] == key:
            idx += 2
        else:
            out.append(args[idx])
            idx += 1
    return out


def build_args(spec: VersionSpec, report_path: Path, output_dir: Path, smoke: bool) -> list[str]:
    args = BASE_ARGS + list(spec.extra_args)
    if smoke:
        for key, value in [
            ("--epochs", "2"),
            ("--steps", "2"),
            ("--val_size", "4"),
            ("--val_ants", "16"),
            ("--val_n_iter", "2"),
            ("--val_mini_H", "2"),
        ]:
            args = replace_arg(args, key, value)
        args = replace_arg(args, "--threads", "2")
    args.extend(["--report_path", str(report_path), "--output", str(output_dir)])
    return args


def parse_report(path: Path, expected_epochs: int = 40) -> dict[str, str | float]:
    if not path.exists():
        return {"status": "missing"}
    text = path.read_text(encoding="utf-8", errors="replace")
    best_epoch = re.search(r"- Best epoch: `([^`]+)`", text)
    best_metric = re.search(r"- Best selected metric: `([^`]+)`", text)
    rows = []
    lines = text.splitlines()
    header = None
    for idx, line in enumerate(lines):
        if line.startswith("| epoch |") or line.startswith("| step |"):
            header = [cell.strip() for cell in line.strip("|").split("|")]
            for row_line in lines[idx + 2:]:
                if not row_line.startswith("|"):
                    break
                cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                if len(cells) == len(header):
                    rows.append(dict(zip(header, cells)))
            break
    def as_float(value: str) -> float:
        try:
            return float(value)
        except Exception:
            return math.nan
    epoch_rows = [row for row in rows if row.get("epoch") not in {"", "-1", "0"}]
    best_t_values = [as_float(row.get("faco_test_best_T", "nan")) for row in epoch_rows]
    best_t_values = [value for value in best_t_values if not math.isnan(value)]
    last10 = best_t_values[-10:]
    status = "done" if len(epoch_rows) >= expected_epochs else "partial"
    return {
        "status": status,
        "best_epoch": best_epoch.group(1) if best_epoch else "",
        "best_metric": float(best_metric.group(1)) if best_metric else math.nan,
        "last10_mean": sum(last10) / len(last10) if last10 else math.nan,
        "last10_std": (sum((x - (sum(last10) / len(last10))) ** 2 for x in last10) / len(last10)) ** 0.5 if last10 else math.nan,
        "auc": sum(best_t_values) / len(best_t_values) if best_t_values else math.nan,
        "epochs_logged": len(epoch_rows),
    }


def format_float(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_summary(summary_path: Path, results: list[dict[str, object]], smoke: bool):
    runnable_results = [row for row in results if row["runnable"] and row.get("status") == "done"]
    best = min(runnable_results, key=lambda row: float(row.get("best_metric", math.inf)), default=None)
    baseline = next((row for row in runnable_results if row["version"] == "V0"), None)

    lines = []
    lines.append("# DyNACO PPO Version Ladder Results")
    lines.append("")
    lines.append(f"- Mode: `{'smoke' if smoke else 'full'}`")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Best version: `{best['version'] if best else 'n/a'}`")
    lines.append(f"- Best metric: `{format_float(best.get('best_metric')) if best else 'n/a'}`")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Version | Description | Status | Best T | Delta vs V0 | Best epoch | Last10 mean | Last10 std | AUC | Epochs | Report |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    base_metric = float(baseline.get("best_metric", math.nan)) if baseline else math.nan
    for row in results:
        best_metric = row.get("best_metric", math.nan)
        delta = float(best_metric) - base_metric if isinstance(best_metric, float) and not math.isnan(base_metric) else math.nan
        report = row.get("report", "")
        lines.append(
            "| {version} | {description} | {status} | {best_metric} | {delta} | {best_epoch} | {last10_mean} | {last10_std} | {auc} | {epochs_logged} | `{report}` |".format(
                version=row["version"],
                description=row["description"],
                status=row.get("status", ""),
                best_metric=format_float(best_metric),
                delta=format_float(delta),
                best_epoch=row.get("best_epoch", ""),
                last10_mean=format_float(row.get("last10_mean", math.nan)),
                last10_std=format_float(row.get("last10_std", math.nan)),
                auc=format_float(row.get("auc", math.nan)),
                epochs_logged=row.get("epochs_logged", ""),
                report=report,
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Lower `Best T`, `Last10 mean`, and `AUC` are better.")
    lines.append("- `Delta vs V0` is paired only by shared validation protocol, not holdout64; negative is better than baseline.")
    lines.append("- V10-V11B are listed as blocked because the current Python code does not yet expose the required C++ policy actions/features.")
    lines.append("- Use the per-version report links for full per-epoch diagnostics.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the ladder with the Stage-0 smoke budget.")
    parser.add_argument("--resume", action="store_true", help="Skip versions whose reports already contain a best metric.")
    parser.add_argument("--only", nargs="*", default=None, help="Only run these version ids, e.g. V0 V1 V2.")
    parser.add_argument("--report-dir", default="experiments/dynaco_version_ladder_runs")
    parser.add_argument("--output-dir", default="experiments/dynaco_version_ladder_ckpt")
    parser.add_argument("--summary", default="dynaco_version_ladder_results.md")
    args = parser.parse_args()
    expected_epochs = 2 if args.smoke else 40

    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    selected = set(args.only) if args.only else None
    for spec in VERSIONS:
        report_path = report_dir / f"{spec.version.lower()}_{spec.extra_args[-1] if spec.extra_args else spec.version.lower()}.md"
        if selected is not None and spec.version not in selected:
            parsed = parse_report(report_path, expected_epochs)
            results.append({"version": spec.version, "description": spec.description, "runnable": spec.runnable, "report": report_path.as_posix(), **parsed})
            continue
        if not spec.runnable:
            results.append({"version": spec.version, "description": spec.description, "runnable": False, "status": "blocked", "report": "", "note": spec.note})
            continue
        if args.resume and parse_report(report_path, expected_epochs).get("status") == "done":
            parsed = parse_report(report_path, expected_epochs)
            results.append({"version": spec.version, "description": spec.description, "runnable": True, "report": report_path.as_posix(), **parsed})
            continue
        cmd = [sys.executable, "train_dynaco_ppo.py", *build_args(spec, report_path, output_dir, args.smoke)]
        print("RUN", spec.version, " ".join(cmd), flush=True)
        status = "failed"
        try:
            subprocess.run(cmd, check=True)
            status = "done"
        except subprocess.CalledProcessError as exc:
            print(f"FAILED {spec.version}: {exc}", file=sys.stderr, flush=True)
        parsed = parse_report(report_path, expected_epochs)
        parsed["status"] = status if parsed.get("status") == "missing" else parsed.get("status", status)
        results.append({"version": spec.version, "description": spec.description, "runnable": True, "report": report_path.as_posix(), **parsed})
        write_summary(Path(args.summary), results, args.smoke)

    write_summary(Path(args.summary), results, args.smoke)
    csv_path = Path(args.summary).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["version", "description", "status", "best_metric", "best_epoch", "last10_mean", "last10_std", "auc", "epochs_logged", "report"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"WROTE {args.summary}")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
