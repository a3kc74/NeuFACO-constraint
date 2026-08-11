import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_cli(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=True,
    )


def test_root_entrypoints_expose_help():
    for script in ("train.py", "test.py", "generate_data.py", "baselines/macs_baseline.py", "baselines/pyvrp_baseline.py"):
        result = run_cli(script, "--help")
        assert "usage:" in result.stdout


def test_root_train_forwards_legacy_help():
    result = run_cli("train.py", "--method", "dynaco_ppo", "--", "--help")
    assert "Train DyNACO-style PPO" in result.stdout


def test_root_test_forwards_legacy_help():
    result = run_cli("test.py", "--method", "faco", "--", "--help")
    assert "Test MFACO_CVRPTW" in result.stdout


def test_refactored_wrappers_import_backend_and_models():
    from analysis.dynaco_grid import default_grid
    from models.faco_net import Net as FacoNet
    from models.gfacs_net import Net as GfacsNet
    from solvers.faco import MFACO_CVRPTW
    from solvers.faco_backend import import_backend

    assert MFACO_CVRPTW.__name__ == "MFACO_CVRPTW"
    assert hasattr(import_backend(), "set_num_threads")
    assert FacoNet.__name__ == "Net"
    assert GfacsNet.__name__ == "Net"
    assert len(default_grid()) > 0
