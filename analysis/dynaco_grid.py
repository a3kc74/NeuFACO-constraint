from __future__ import annotations

import argparse
import dataclasses
import math
import subprocess
import sys
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class GridConfig:
    name: str
    args: tuple[str, ...]


def default_grid() -> list[GridConfig]:
    return [
        GridConfig('old_default', ('--lr', '5e-6', '--ants', '32', '--train_H', '1', '--train_mini_H', '4', '--ppo_clip', '0.1')),
        GridConfig('lr_1e-5', ('--lr', '1e-5', '--ants', '32', '--train_H', '1', '--train_mini_H', '4', '--ppo_clip', '0.1')),
        GridConfig('lr_2e-5', ('--lr', '2e-5', '--ants', '32', '--train_H', '1', '--train_mini_H', '4', '--ppo_clip', '0.1')),
        GridConfig('miniH_8', ('--lr', '5e-6', '--ants', '32', '--train_H', '1', '--train_mini_H', '8', '--ppo_clip', '0.1')),
        GridConfig('miniH_10_lr1e-5', ('--lr', '1e-5', '--ants', '32', '--train_H', '1', '--train_mini_H', '10', '--ppo_clip', '0.1')),
        GridConfig('ants_64', ('--lr', '5e-6', '--ants', '64', '--train_H', '1', '--train_mini_H', '4', '--ppo_clip', '0.1')),
        GridConfig('ants64_miniH8_lr1e-5', ('--lr', '1e-5', '--ants', '64', '--train_H', '1', '--train_mini_H', '8', '--ppo_clip', '0.1')),
        GridConfig('clip_0.2_entropy', ('--lr', '1e-5', '--ants', '32', '--train_H', '1', '--train_mini_H', '4', '--ppo_clip', '0.2', '--entropy_coeff', '0.01')),
    ]


def base_args(smoke: bool, config_name: str, report_path: Path) -> list[str]:
    args = [
        sys.executable, 'train_dynaco_ppo.py', '100', '--threads', '8', '--batch_size', '1',
        '--val_size', '8', '--val_ants', '100', '--min_new_edges', '8', '--val_n_iter', '10',
        '--val_mini_H', '10', '--smooth_mmas', '--extend_ls', '--use_local_search',
        '--epochs', '5', '--steps', '20', '--disable_wandb', '--val_infer_mode', 'faco_test',
        '--select_metric_mode', 'faco_test', '--run_name', config_name, '--report_path', str(report_path),
    ]
    if smoke:
        replacements = {'--threads': '1', '--val_size': '1', '--val_ants': '2', '--val_n_iter': '1', '--val_mini_H': '1', '--epochs': '1', '--steps': '1'}
        for flag, value in replacements.items():
            args[args.index(flag) + 1] = value
        args.extend(['--device', 'cpu'])
    return args


def build_command(config: GridConfig, smoke: bool = False, report_dir: Path | None = None) -> list[str]:
    report_dir = report_dir or Path('experiments/dynaco_faco_test_grid')
    return base_args(smoke, config.name, report_dir / f'{config.name}.md') + list(config.args)


def parse_report(report_path: Path) -> dict[str, float | int | bool]:
    rows = []
    for line in report_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped.startswith('|') or '---' in stripped or 'epoch' in stripped:
            continue
        parts = [part.strip() for part in stripped.strip('|').split('|')]
        if len(parts) < 4:
            continue
        try:
            rows.append((int(parts[0]), float(parts[3])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f'no faco_test_best_T rows found in {report_path}')
    epoch0 = next(value for epoch, value in rows if epoch == 0)
    final_epoch, final_value = rows[-1]
    best_epoch, best_value = min(rows, key=lambda item: item[1])
    improvement = epoch0 - final_value
    return {'epoch0': epoch0, 'final_epoch': final_epoch, 'final': final_value, 'best_epoch': best_epoch, 'best': best_value, 'improvement': improvement, 'improved': improvement > 0}


def write_summary(summary_path: Path, results: list[dict], commands: dict[str, list[str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in results if row.get('ok')]
    winner = min(completed, key=lambda row: row['final']) if completed else None
    with summary_path.open('w', encoding='utf-8') as f:
        f.write('# DyNACO FACO Test Grid Summary\n\n')
        if winner is not None:
            f.write(f"- Winner by final epoch: `{winner['name']}` with `faco_test_best_T={winner['final']:.6f}`\n")
            f.write(f"- Winner improvement vs epoch 0: `{winner['improvement']:.6f}`\n")
        f.write('- Validation mode: `faco_test` only\n')
        f.write('- Objective: lowest final epoch `faco_test_best_T`\n\n')
        headers = ['rank', 'name', 'epoch0', 'final_epoch', 'final', 'improvement', 'best_epoch', 'best', 'runtime_sec', 'ok']
        f.write('## Results\n\n')
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for rank, row in enumerate(sorted(results, key=lambda item: item.get('final', math.inf)), start=1):
            values = [rank]
            for key in headers[1:]:
                value = row.get(key, '')
                values.append(f'{value:.6f}' if isinstance(value, float) else value)
            f.write('| ' + ' | '.join(str(value) for value in values) + ' |\n')
        f.write('\n## Commands\n\n')
        for name, command in commands.items():
            f.write(f'### {name}\n\n```powershell\n')
            f.write(' '.join(command).replace(sys.executable, 'uv run python'))
            f.write('\n```\n\n')


def run_grid(smoke: bool = False, only: str | None = None) -> list[dict]:
    report_dir = Path('experiments/dynaco_faco_test_grid_smoke' if smoke else 'experiments/dynaco_faco_test_grid')
    summary_path = Path('experiments/dynaco_faco_test_grid_smoke_summary.md' if smoke else 'experiments/dynaco_faco_test_grid_summary.md')
    configs = default_grid()
    if smoke:
        configs = configs[:1]
    if only:
        configs = [config for config in configs if config.name == only]
        if not configs:
            raise ValueError(f'unknown config: {only}')
    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    commands = {}
    for config in configs:
        command = build_command(config, smoke=smoke, report_dir=report_dir)
        commands[config.name] = command
        started = time.perf_counter()
        print(f'=== Running {config.name} ===')
        completed = subprocess.run(command, check=False)
        row = {'name': config.name, 'runtime_sec': time.perf_counter() - started, 'ok': completed.returncode == 0}
        report_path = report_dir / f'{config.name}.md'
        if completed.returncode == 0 and report_path.exists():
            row.update(parse_report(report_path))
        else:
            row['returncode'] = completed.returncode
        results.append(row)
        write_summary(summary_path, results, commands)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Run DyNACO PPO parameter grid with faco_test validation only.')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_grid(smoke=args.smoke, only=args.only)
