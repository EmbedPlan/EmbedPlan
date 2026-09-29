"""End to end: write the toy domain with its CLI, then run the real training driver.

Both steps run as subprocesses from the repo root with EMBEDPLAN_DATA and
EMBEDPLAN_RESULTS pointed at tmp dirs, so nothing touches the repo's data/ or results/.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# `python -m experiments.train`, except that DataLoader workers are forked. The driver
# builds its loaders with 4-8 workers, and under spawn (the macOS default) every worker
# re-imports the driver on every pass over a loader: about two minutes for this 2-epoch
# run on a laptop, against about three seconds with fork.
LAUNCH_TRAIN = (
    "import multiprocessing as mp, runpy\n"
    "if 'fork' in mp.get_all_start_methods():\n"
    "    mp.set_start_method('fork', force=True)\n"
    "runpy.run_module('experiments.train', run_name='__main__', alter_sys=True)\n"
)


def run(args, env, timeout=180):
    proc = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)
    assert proc.returncode == 0, f"{args[:2]} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    return proc


@pytest.fixture(scope="module")
def toy_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    env = {**os.environ, "EMBEDPLAN_DATA": str(tmp / "data"), "EMBEDPLAN_RESULTS": str(tmp / "results"),
           "WANDB_MODE": "disabled", "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"}
    run(["-m", "tools.make_toy_domain", "--out", str(tmp / "data"), "--domain", "toy"], env)
    config = tmp / "config.yaml"
    config.write_text(yaml.safe_dump({"wandb": {"enabled": False},
                                      "training": {"eval_start_epoch": 1, "checkpoint_freq": 1, "patience": 100}}))
    return tmp, env, config


def test_toy_domain_cli_writes_the_expected_layout(toy_env):
    tmp, _, _ = toy_env
    data = tmp / "data"
    expected = ["original_df_pkls_factorized/toy-test.pkl",
                "original_df_pkls_factorized/toy-test_values.pkl",
                "full_embeddings/toy-hash-bow/original/toy.pt",
                "full_embeddings/toy-hash-bow/original/toy_prompt_to_index.pkl",
                "full_embeddings_actions/toy-hash-bow/original/toy_actions.pt",
                "full_embeddings_actions/toy-hash-bow/original/toy_actions_index.pkl"]
    assert all((data / f).is_file() for f in expected)


@pytest.mark.parametrize("split", ["random", "problem_grouped"])
def test_train_driver_writes_finite_metrics(toy_env, split):
    tmp, env, config = toy_env
    prefix = tmp / f"job_{split}"
    run(["-c", LAUNCH_TRAIN, "--domain", "toy", "--model_name", "toy-hash-bow", "--split_type", split,
         "--epochs", "2", "--no_wandb", "--use_projection", "--projection_dim", "16", "--hidden_size", "16",
         "--n_layers", "2", "--use_layer_norm", "--config", str(config), "--save_prefix", str(prefix)], env)

    result = json.loads(prefix.with_suffix(".json").read_text())
    assert (result["train_domain"], result["test_domain"]) == ("toy", "toy")
    assert result["args"]["split_type"] == split
    metrics = result["metrics"]
    # train_loop returns best_* keys once some epoch beats Hit@5 = 0, and the raw keys otherwise
    pre = "best_" if "best_hit@5" in metrics else ""
    hit1, hit5 = metrics[f"{pre}hit@1"], metrics[f"{pre}hit@5"]
    assert all(math.isfinite(v) for v in (hit1, hit5))
    assert 0.0 <= hit1 <= hit5 <= 1.0
