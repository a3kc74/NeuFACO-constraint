import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cvrptw-faco"))

import argparse
import torch

from net import Net
from utils_ppo import load_val_dataset
from train_ppo import infer_instance, DEVICE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate focused PPO on cached CVRPTW benchmark data')
    parser.add_argument('nodes', type=int)
    parser.add_argument('-k', '--k_sparse', type=int, default=32)
    parser.add_argument('-a', '--ants', type=int, default=64)
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-v', '--val_size', type=int, default=20)
    args = parser.parse_args()

    model = Net(value_head=True).to(DEVICE)
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    dataset = load_val_dataset(args.nodes, args.k_sparse, DEVICE, val_size=args.val_size)
    stats = [infer_instance(model, pyg_data, instance, args.ants, cand_list_size=args.k_sparse) for pyg_data, instance in dataset]
    print(torch.as_tensor(stats, dtype=torch.float32).mean(0).tolist())
