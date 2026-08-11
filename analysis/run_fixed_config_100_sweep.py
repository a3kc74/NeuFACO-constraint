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


FIXED_BASE_ARGS = [
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
    "--epochs", "50",
    "--steps", "32",
    "--val_infer_mode", "faco_test",
    "--select_metric_mode", "faco_test",
    "--ants", "100",
    "--train_H", "1",
    "--train_mini_H", "1",
    "--ppo_clip", "0.1",
    "--parallel_traced",
    "--disable_wandb",
    "--early_stop_patience", "5",
    "--early_stop_min_delta", "0.0",
]

FIXED_KEYS = {
    "nodes", "--threads", "--batch_size", "--val_size", "--val_ants", "--min_new_edges",
    "--val_n_iter", "--val_mini_H", "--smooth_mmas", "--extend_ls", "--use_local_search",
    "--epochs", "--steps", "--val_infer_mode", "--select_metric_mode", "--ants",
    "--train_H", "--train_mini_H", "--ppo_clip", "--parallel_traced",
}


@dataclass(frozen=True)
class VersionSpec:
    name: str
    description: str
    extra_args: tuple[str, ...]


def slugify(text: str) -> str:
    text = text.lower().replace("--", "")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:80]


def add_spec(specs: list[VersionSpec], seen: set[tuple[str, ...]], description: str, args: list[str]):
    key = tuple(args)
    if key in seen:
        return
    seen.add(key)
    specs.append(VersionSpec(f"v{len(specs):03d}_{slugify(description)}", description, tuple(args)))


def build_specs(limit: int = 100) -> list[VersionSpec]:
    specs: list[VersionSpec] = []
    seen: set[tuple[str, ...]] = set()
    add_spec(specs, seen, "baseline_defaults", [])

    lr_values = ["1e-6", "2e-6", "5e-6", "1e-5", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3"]
    decay_values = ["0.3", "0.5", "0.7", "0.85", "0.95"]
    ppo_epoch_values = ["1", "2", "3", "4", "6", "8"]
    entropy_values = ["0.0001", "0.0003", "0.001", "0.003", "0.01"]
    grad_values = ["0.25", "0.5", "0.75", "1.0", "2.0", "5.0"]
    kl_values = ["0.003", "0.005", "0.01", "0.02", "0.05"]
    adv_clip_values = ["1.0", "2.0", "3.0", "5.0"]
    ema_values = ["0.9", "0.99", "0.995", "0.999"]
    alpha_values = ["0.5", "0.75", "1.25", "1.5", "2.0"]
    p_best_values = ["0.01", "0.02", "0.1", "0.2"]
    nls_beta_values = ["0.1", "0.2", "0.4", "0.6", "0.8"]

    for lr in lr_values:
        add_spec(specs, seen, f"lr_{lr}", ["--lr", lr])
    for decay in decay_values:
        add_spec(specs, seen, f"decay_{decay}", ["--decay", decay])
    for ppo_epochs in ppo_epoch_values:
        add_spec(specs, seen, f"ppo_epochs_{ppo_epochs}", ["--ppo_epochs", ppo_epochs])
    for entropy in entropy_values:
        add_spec(specs, seen, f"entropy_{entropy}", ["--entropy_coeff", entropy])
    for grad in grad_values:
        add_spec(specs, seen, f"max_grad_norm_{grad}", ["--max_grad_norm", grad])
    for target_kl in kl_values:
        add_spec(specs, seen, f"target_kl_{target_kl}", ["--target_kl", target_kl])
    for adv_clip in adv_clip_values:
        add_spec(specs, seen, f"macro_adv_clip_{adv_clip}", ["--advantage_mode", "macro_relative", "--adv_clip", adv_clip])
    add_spec(specs, seen, "no_adv_norm", ["--no_adv_norm"])
    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        add_spec(specs, seen, f"cosine_lr_{lr}", ["--lr", lr, "--lr_scheduler", "cosine"])
    for ema in ema_values:
        add_spec(specs, seen, f"ema_{ema}", ["--ema_decay", ema])
    for lr in ["5e-6", "1e-5", "5e-5", "1e-4", "5e-4"]:
        add_spec(specs, seen, f"layernorm_lr_{lr}", ["--norm_type", "layer", "--lr", lr])
    for alpha in alpha_values:
        add_spec(specs, seen, f"alpha_{alpha}", ["--alpha", alpha])
    for p_best in p_best_values:
        add_spec(specs, seen, f"p_best_{p_best}", ["--p_best", p_best])
    for beta in nls_beta_values:
        add_spec(specs, seen, f"nls_beta_{beta}", ["--nls", "--nls_beta", beta])

    # Priority pair/triple refinements around stable PPO controls.
    add_spec(specs, seen, "static_edge_features", ["--edge_feature_mode", "static"])
    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        add_spec(specs, seen, f"static_edge_lr_{lr}", ["--edge_feature_mode", "static", "--lr", lr])
    for alpha in ["0.75", "1.0", "1.25", "1.5"]:
        add_spec(specs, seen, f"static_edge_alpha_{alpha}", ["--edge_feature_mode", "static", "--alpha", alpha])

    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        for target_kl in ["0.005", "0.01", "0.02"]:
            add_spec(specs, seen, f"lr_{lr}_kl_{target_kl}", ["--lr", lr, "--target_kl", target_kl])
    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        for grad in ["0.25", "0.5", "1.0"]:
            add_spec(specs, seen, f"lr_{lr}_grad_{grad}", ["--lr", lr, "--max_grad_norm", grad])
    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        for entropy in ["0.0003", "0.001", "0.003"]:
            add_spec(specs, seen, f"lr_{lr}_entropy_{entropy}", ["--lr", lr, "--entropy_coeff", entropy])
    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        for adv_clip in ["2.0", "3.0", "5.0"]:
            add_spec(specs, seen, f"lr_{lr}_macro_adv_{adv_clip}", ["--lr", lr, "--advantage_mode", "macro_relative", "--adv_clip", adv_clip])
    for lr in ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"]:
        add_spec(
            specs,
            seen,
            f"stable_pack_lr_{lr}",
            ["--lr", lr, "--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--target_kl", "0.01"],
        )
        add_spec(
            specs,
            seen,
            f"v6lite_lr_{lr}",
            ["--lr", lr, "--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--target_kl", "0.01", "--advantage_mode", "macro_relative", "--adv_clip", "3.0"],
        )
        add_spec(
            specs,
            seen,
            f"v6lite_cosine_lr_{lr}",
            ["--lr", lr, "--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--target_kl", "0.01", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--lr_scheduler", "cosine"],
        )
        add_spec(
            specs,
            seen,
            f"v6lite_layernorm_lr_{lr}",
            ["--lr", lr, "--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--target_kl", "0.01", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--norm_type", "layer"],
        )
        add_spec(
            specs,
            seen,
            f"v6lite_ema_lr_{lr}",
            ["--lr", lr, "--decay", "0.5", "--ppo_epochs", "2", "--entropy_coeff", "0.001", "--max_grad_norm", "0.5", "--target_kl", "0.01", "--advantage_mode", "macro_relative", "--adv_clip", "3.0", "--ema_decay", "0.995"],
        )

    for decay in ["0.3", "0.5", "0.7"]:
        for lr in ["1e-5", "5e-5", "1e-4", "2e-4"]:
            add_spec(specs, seen, f"decay_{decay}_lr_{lr}_kl", ["--decay", decay, "--lr", lr, "--target_kl", "0.01"])

    if len(specs) < limit:
        raise RuntimeError(f"only generated {len(specs)} specs")
    return specs[:limit]


def report_status(path: Path, expected_epochs: int) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing"}
    text = path.read_text(encoding="utf-8", errors="replace")
    best_epoch = re.search(r"- Best epoch: `([^`]+)`", text)
    best_metric = re.search(r"- Best selected metric: `([^`]+)`", text)
    stopped = re.search(r"- Stopped early: `([^`]+)`", text)
    rows = []
    header = None
    lines = text.splitlines()
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

    def f(value: str) -> float:
        try:
            return float(value)
        except Exception:
            return math.nan

    epoch_rows = [row for row in rows if row.get("epoch") not in {"", "-1", "0"}]
    val_values = [f(row.get("faco_test_best_T", "nan")) for row in epoch_rows]
    val_values = [value for value in val_values if not math.isnan(value)]
    train_loss = [f(row.get("train_loss", "nan")) for row in epoch_rows]
    approx_kl = [f(row.get("train_approx_kl", "nan")) for row in epoch_rows]
    clip_frac = [f(row.get("train_clip_frac", "nan")) for row in epoch_rows]
    status = "done" if stopped or len(epoch_rows) >= expected_epochs else "partial"
    if not best_metric:
        status = "partial" if epoch_rows else "missing"
    return {
        "status": status,
        "best_epoch": best_epoch.group(1) if best_epoch else "",
        "best_metric": f(best_metric.group(1)) if best_metric else math.nan,
        "epochs": len(epoch_rows),
        "last_metric": val_values[-1] if val_values else math.nan,
        "last5_mean": sum(val_values[-5:]) / min(len(val_values), 5) if val_values else math.nan,
        "auc": sum(val_values) / len(val_values) if val_values else math.nan,
        "last_train_loss": train_loss[-1] if train_loss else math.nan,
        "max_kl": max([x for x in approx_kl if not math.isnan(x)], default=math.nan),
        "max_clip_frac": max([x for x in clip_frac if not math.isnan(x)], default=math.nan),
        "stopped_reason": stopped.group(1) if stopped else "",
    }


def fmt(value: object) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def write_summary(path: Path, results: list[dict[str, object]], total_specs: int):
    done = [row for row in results if row.get("status") == "done"]
    best = min(done, key=lambda row: float(row.get("best_metric", math.inf)), default=None)
    baseline = next((row for row in results if row["name"].startswith("v000_")), None)
    base_metric = float(baseline.get("best_metric", math.nan)) if baseline else math.nan

    lines = [
        "# Fixed Config Train Sweep Results",
        "",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Planned versions: `{total_specs}`",
        f"- Completed versions: `{len(done)}`",
        f"- Best version: `{best['name'] if best else 'n/a'}`",
        f"- Best metric: `{fmt(best.get('best_metric')) if best else 'n/a'}`",
        "",
        "## Fixed Base Config",
        "",
        "```powershell",
        "uv run .\\train_dynaco_ppo.py " + " ".join(FIXED_BASE_ARGS[:-4]),
        "```",
        "",
        "## Results",
        "",
        "| # | Version | Status | Best T | Delta vs baseline | Best epoch | Epochs | Last T | Last5 mean | AUC | Max KL | Max clip | Extra args | Report |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(results):
        metric = row.get("best_metric", math.nan)
        delta = float(metric) - base_metric if isinstance(metric, float) and not math.isnan(base_metric) else math.nan
        extra = " ".join(row.get("extra_args", ()))
        lines.append(
            f"| {idx} | `{row['name']}` | {row.get('status', '')} | {fmt(metric)} | {fmt(delta)} | {row.get('best_epoch', '')} | {row.get('epochs', '')} | {fmt(row.get('last_metric', math.nan))} | {fmt(row.get('last5_mean', math.nan))} | {fmt(row.get('auc', math.nan))} | {fmt(row.get('max_kl', math.nan))} | {fmt(row.get('max_clip_frac', math.nan))} | `{extra}` | `{row.get('report', '')}` |"
        )
    lines.extend([
        "",
        "## Stop Rule",
        "",
        "- Every runnable version adds `--early_stop_patience 5 --early_stop_min_delta 0.0`.",
        "- A run is marked done when it either reaches 50 epochs or stops because the selected validation metric did not improve for 5 epochs.",
        "- Lower `Best T`, `Last T`, `Last5 mean`, and `AUC` are better.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace):
    specs = build_specs(args.versions)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    run_count = 0
    for spec in specs:
        report = report_dir / f"{spec.name}.md"
        status = report_status(report, expected_epochs=50)
        if args.resume and status.get("status") == "done":
            results.append({"name": spec.name, "description": spec.description, "extra_args": spec.extra_args, "report": report.as_posix(), **status})
            continue
        if args.max_runs is not None and run_count >= args.max_runs:
            results.append({"name": spec.name, "description": spec.description, "extra_args": spec.extra_args, "report": report.as_posix(), **status})
            continue
        cmd = [sys.executable, "train_dynaco_ppo.py", *FIXED_BASE_ARGS, *spec.extra_args, "--run_name", spec.name, "--report_path", str(report), "--output", str(output_dir)]
        log_path = log_dir / f"{spec.name}.log"
        print(f"RUN {spec.name}: {spec.description}", flush=True)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(cmd) + "\n")
            log.flush()
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        run_count += 1
        status = report_status(report, expected_epochs=50)
        if completed.returncode != 0:
            status["status"] = "failed"
        status["seconds"] = time.time() - started
        results.append({"name": spec.name, "description": spec.description, "extra_args": spec.extra_args, "report": report.as_posix(), **status})
        write_summary(Path(args.summary), results, len(specs))
        print(f"DONE {spec.name}: {status.get('status')} best={fmt(status.get('best_metric', math.nan))} epochs={status.get('epochs', '')}", flush=True)

    # Include already completed rows after max_runs cut-off in final summary.
    if len(results) < len(specs):
        seen_names = {row["name"] for row in results}
        for spec in specs:
            if spec.name in seen_names:
                continue
            report = report_dir / f"{spec.name}.md"
            status = report_status(report, expected_epochs=50)
            results.append({"name": spec.name, "description": spec.description, "extra_args": spec.extra_args, "report": report.as_posix(), **status})
    write_summary(Path(args.summary), results, len(specs))

    csv_path = Path(args.summary).with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["name", "description", "status", "best_metric", "best_epoch", "epochs", "last_metric", "last5_mean", "auc", "max_kl", "max_clip_frac", "stopped_reason", "report", "extra_args"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            out = {key: row.get(key, "") for key in fields}
            out["extra_args"] = " ".join(row.get("extra_args", ()))
            writer.writerow(out)
    print(f"WROTE {args.summary}")
    print(f"WROTE {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=int, default=100)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-dir", default="experiments/fixed_config_sweep_runs")
    parser.add_argument("--output-dir", default="experiments/fixed_config_sweep_ckpt")
    parser.add_argument("--log-dir", default="experiments/fixed_config_sweep_logs")
    parser.add_argument("--summary", default="fixed_config_100_sweep_results.md")
    run_sweep(parser.parse_args())


if __name__ == "__main__":
    main()
