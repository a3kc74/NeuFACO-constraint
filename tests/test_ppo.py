import argparse
import numpy as np
import torch

from models.faco_net import Net
from envs.cvrptw_env import generate_cvrptw_instance, gen_pyg_data
from trainers.ppo_trainer import infer_instance, DEVICE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate focused PPO on random CVRPTW instances')
    parser.add_argument('nodes', type=int)
    parser.add_argument('-k', '--k_sparse', type=int, default=32)
    parser.add_argument('-a', '--ants', type=int, default=64)
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-n', '--instances', type=int, default=10)
    args = parser.parse_args()

    model = Net(value_head=True).to(DEVICE)
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    stats = []
    for seed in range(args.instances):
        instance = generate_cvrptw_instance(args.nodes, seed=seed)
        pyg_data = gen_pyg_data(instance, cand_list_size=args.k_sparse)
        stats.append(infer_instance(model, pyg_data, instance, args.ants, cand_list_size=args.k_sparse))
    print(np.asarray(stats).mean(0).tolist())
