# NeuFACO Constraint

This repository contains CVRPTW/VRPTW experiments for GFACS, FACO, DyNACO-style PPO, focused PPO, DeepACO-style training, PyVRP, and MACS baselines. The source tree is organized by role, with CaR-style root entry points for training, evaluation, and data generation.

## Dependencies
- Python 3.11.5
- PyTorch 2.1.1
- PyTorch Geometric 2.4.0

For the complete list of dependencies, please refer to the `requirements.txt` file.

## Usage
Root entry points forward unknown arguments to the selected method implementation.

### Train

```raw
python train.py --method dynaco_ppo -- 100 --disable_wandb
python train.py --method ppo -- 100 --disable_wandb
python train.py --method gfacs -- 100 --disable_wandb
python train.py --method deepaco -- 100 --disable_wandb
```

### Test

```raw
python test.py --method faco -- 100 -n 100 -i 10
python test.py --method faco_ib -- 100 -n 100 -i 10
python test.py --method gfacs -- 100 -p path_to_checkpoint.pt
python test.py --method macs -- 100 --time-limit 600
python baselines/macs_baseline.py 100 --time-limit 600
python baselines/pyvrp_baseline.py 100
```

Use `--problem VRPTW` on the root wrapper to append `--vrptw` where supported.

### Generate Data

```raw
python generate_data.py --problem CVRPTW --
python generate_data.py --problem VRPTW --
```

## Layout

- `train.py`, `test.py`, `generate_data.py`: canonical root entry points.
- `solvers/`: FACO, GFACS ACO, MACS, and PyVRP local-search solver code.
- `models/`: FACO/PPO and GFACS neural network modules.
- `envs/`: shared CVRPTW/VRPTW data, graph, and instance helpers.
- `trainers/`: DyNACO PPO, focused PPO, GFACS, and DeepACO trainers.
- `evaluation/`: FACO, FACO-IB, and GFACS evaluation modules.
- `baselines/`: canonical wrappers for MACS and PyVRP baselines.
- `analysis/`: experiment runners, reports, and analysis utilities.
- `cpp/faco/`: FACO C++ backend source and build wrapper.
- `data/`, `pretrained/`: datasets and checkpoints/results.

## FACO C++ Backend

The FACO solver uses a compiled `faco_opt` extension. Build it before running FACO/DyNACO commands that require the C++ backend:

```raw
uv run python cpp/faco/setup.py build_ext --inplace
```

Generated binaries and build folders are ignored by Git; canonical C++ source lives under `cpp/faco/src/`.
