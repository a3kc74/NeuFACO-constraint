import os
import random
from pathlib import Path

__test__ = False

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from models.gfacs_net import Net
from solvers.gfacs_aco import ACO
from envs.gfacs_data import load_test_dataset
from utils import load_rl4co_dataset, normalize_dataset_item

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


EPS = 1e-10
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TAM = False
ACOALG = "AS"


@torch.no_grad()
def infer_instance(

    model,
    pyg_data,
    demands,
    distances,
    positions,
    windows,
    n_ants,
    t_aco_diff,
    local_search_params=None,
    cost_evaluator=None,
):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        demands=demands.cpu(),
        distances=distances.cpu(),
        windows = windows.cpu(),
        positions=positions.cpu(),
        elitist=ACOALG == "ELITIST",
        maxmin=ACOALG == "MAXMIN",
        rank_based=ACOALG == "RANK",
        use_local_search=True,
        local_search_params=local_search_params,
        device='cpu',
    )

    results = torch.zeros(size=(len(t_aco_diff),))
    diversities = torch.zeros(size=(len(t_aco_diff),))
    elapsed_time = 0
    for i, t in enumerate(t_aco_diff):
        results[i], diversities[i], t = aco.run(t)
        if cost_evaluator is not None and aco.shortest_path is not None:
            route = aco.shortest_path.detach().cpu().numpy()
            try:
                results[i] = float(cost_evaluator([route])[0])
            except AssertionError:
                results[i] = float("inf")
        elapsed_time += t
    return results, diversities, elapsed_time


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, local_search_params):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    sum_results = torch.zeros(size=(len(t_aco_diff),))
    sum_diversities = torch.zeros(size=(len(t_aco_diff),))
    sum_times = 0
    for item in tqdm(dataset, dynamic_ncols=True):
        if isinstance(item, dict):
            demands, distances, positions, windows, cost_evaluator = normalize_dataset_item(item)
            pyg_data = None
        else:
            pyg_data, demands, distances, positions, windows = item
            cost_evaluator = None
        results, diversities, elapsed_time = infer_instance(
            model,
            pyg_data,
            demands,
            distances,
            positions,
            windows,
            n_ants,
            t_aco_diff,
            local_search_params,
            cost_evaluator,
        )
        sum_results += results
        sum_diversities += diversities
        sum_times += elapsed_time
    return sum_results / len(dataset), sum_diversities / len(dataset), sum_times / len(dataset)


def load_checkpoint_model(ckpt_path, guided_exploration=False, deepaco=False):
    if ckpt_path is None:
        return None

    if deepaco:
        net = Net(gfn=False).to(DEVICE)
    else:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return net


def main(
    ckpt_path,
    n_nodes,
    k_sparse,
    size=None,
    n_ants=100,
    n_iter=10,
    guided_exploration=False,
    seed=0,
    local_search_params=None,
    vrptw=False,
    deepaco=False,
    data_source="gfacs",
    data_dir=None,
    rl4co_root=None,
    rl4co_phase="test",
    rl4co_file=None,
    rl4co_seed=1234,
    rl4co_scale=False,
):
    if data_source == "rl4co":
        if vrptw:
            raise ValueError("RL4CO data source is available for CVRPTW only")
        if TAM:
            raise ValueError("RL4CO data source does not support TAM datasets")
        test_list = load_rl4co_dataset(
            n_nodes=n_nodes,
            size=size,
            phase=rl4co_phase,
            filename=rl4co_file,
            data_dir=data_dir,
            rl4co_root=rl4co_root,
            seed=rl4co_seed,
            scale=rl4co_scale,
        )
    else:
        test_list = load_test_dataset(
            n_nodes,
            k_sparse,
            DEVICE,
            TAM,
            vrptw=vrptw,
            data_dir=str(data_dir) if data_dir is not None else os.path.join(ROOT_DIR, "data", "cvrptw"),
        )
        test_list = test_list[:(size or len(test_list))]

    t_aco = list(range(1, n_iter + 1))
    problem_name = f"{'tam-' if TAM else ''}{'vrptw' if vrptw else 'cvrptw'}"
    print("problem:", problem_name)
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", size)
    print("data_source:", data_source)
    if data_source == "rl4co":
        print("rl4co_phase:", rl4co_phase)
        print("rl4co_file:", rl4co_file if rl4co_file is not None else "default/generated")
        print("rl4co_seed:", rl4co_seed)
        print("rl4co_scale:", rl4co_scale)
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_ants:", n_ants)
    print("seed:", seed)
    print("model:", "DeepACO" if deepaco else "GFACS" if ckpt_path is not None else "none")

    net = load_checkpoint_model(ckpt_path, guided_exploration, deepaco)
    avg_cost, avg_diversity, duration = test(test_list, net, n_ants, t_aco, local_search_params)
    print('average inference time: ', duration)
    for i, t in enumerate(t_aco):
        print(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else os.path.join(ROOT_DIR, 'pretrained', 'cvrptw', str(n_nodes), 'no_model')
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-{data_source}-{problem_name}{n_nodes}-ninst{size}-nants{n_ants}-niter{n_iter}-seed{seed}"
    result_file = os.path.join(dirname, result_filename + ".txt")
    with open(result_file, "w") as f:
        f.write(f"problem: {problem_name}\n")
        f.write(f"problem scale: {n_nodes}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"number of instances: {len(test_list)}\n")
        f.write(f"data_source: {data_source}\n")
        if data_source == "rl4co":
            f.write(f"rl4co_phase: {rl4co_phase}\n")
            f.write(f"rl4co_file: {rl4co_file if rl4co_file is not None else 'default/generated'}\n")
            f.write(f"rl4co_seed: {rl4co_seed}\n")
            f.write(f"rl4co_scale: {rl4co_scale}\n")
        f.write(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE+'+cpu'}\n")
        f.write(f"n_ants: {n_ants}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"model: {'DeepACO' if deepaco else 'GFACS' if ckpt_path is not None else 'none'}\n")
        f.write(f"average inference time: {duration}\n")
        for i, t in enumerate(t_aco):
            f.write(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    results = pd.DataFrame(columns=['T', 'avg_cost', 'avg_diversity'])
    results['T'] = t_aco
    results['avg_cost'] = avg_cost
    results['avg_diversity'] = avg_diversity
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for GFACS model w/o guided exploration.')
    parser.add_argument("--deepaco", action="store_true", help="Load checkpoint as a DeepACO model without the GFACS logZ head")
    ### Dataset
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--vrptw", action="store_true", help="Use VRPTW data with all customer demands set to zero")
    parser.add_argument("--data_source", type=str, default="gfacs", choices=["gfacs", "rl4co"], help="Dataset and metric source")
    parser.add_argument("--data_dir", type=Path, default=None, help="Dataset directory")
    parser.add_argument("--rl4co_root", type=Path, default=None, help="Path to the RL4CO repository if it is not installed")
    parser.add_argument("--rl4co_phase", type=str, default="test", choices=["train", "val", "test"], help="RL4CO dataset phase")
    parser.add_argument("--rl4co_file", type=Path, default=None, help="Optional RL4CO .npz dataset file")
    parser.add_argument("--rl4co_seed", type=int, default=1234, help="Seed used when RL4CO generates data")
    parser.add_argument("--rl4co_scale", action="store_true", help="Use RL4CO CVRPTW scaled generator")
    ### ACO
    parser.add_argument("--aco", type=str, default="AS", choices=["AS", "ELITIST", "MAXMIN", "RANK"], help="ACO algorithm")
    ### LocalSearchParams
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of cpus to use")
    parser.add_argument("--max_trials", type=int, default=10, help="Number of iterations to perform")
    parser.add_argument("--load_penalty", type=int, default=20, help="Initial load_penalty in training phase")
    parser.add_argument("--tw_penalty", type=int, default=20, help="Initial tw_penalty in training phase")
    parser.add_argument("--nb_granular", type=int, default=None, help="Granularity of neighbourhood search")
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5
    if args.nb_granular is None:
        args.nb_granular = args.nodes // 5

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    TAM = args.tam
    ACOALG = args.aco

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    local_search_params = {
        "n_cpus": args.n_cpus,
        "max_trials": args.max_trials,
        "neighbourhood_params": {"nb_granular": args.nb_granular},
        "cost_evaluator_params": {"load_penalty": args.load_penalty, "tw_penalty": args.tw_penalty},
    }

    main(
        args.path,
        args.nodes,
        args.k_sparse,
        args.size,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.seed,
        local_search_params,
        args.vrptw,
        args.deepaco,
        args.data_source,
        args.data_dir,
        args.rl4co_root,
        args.rl4co_phase,
        args.rl4co_file,
        args.rl4co_seed,
        args.rl4co_scale,
    )
