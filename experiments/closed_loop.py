"""Train one (domain, split) and run the three rebuttal evaluations on it.

E4  Closed-loop rollout    teacher-forced vs closed-loop vs free-running.
E1  Candidate-pool scaling Hit@k as |C| grows from 128 to the full domain pool.
E3  Open-set abstention    is max similarity a usable "the answer is absent" signal?

Usage:
    python -m experiments.closed_loop --domain ferry --split random
    python -m experiments.closed_loop --domain ferry --split problem_grouped
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from embedplan import build_model, load_domain, make_split, train_transition
from embedplan.config import REBUTTAL_DIR
from embedplan.data import build_trajectories
from embedplan.evaluation import open_set_abstention, pool_sweep
from embedplan.rollout import rollout
from embedplan.scoring import ABSOLUTE, project_pool
from embedplan.utils import resolve_device


def add_model_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Hyperparameters shared by every experiment that trains a transition model."""
    ap.add_argument("--model_name", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--action_weight", type=float, default=2.0)
    ap.add_argument("--projection_dim", type=int, default=128)
    ap.add_argument("--projection_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transition", default="mlp", choices=["mlp", "anchored"])
    return ap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="ferry")
    ap.add_argument("--split", default="random", choices=["random", "problem_grouped"])
    ap.add_argument("--max_trajs", type=int, default=300)
    ap.add_argument("--pool_sizes", type=int, nargs="+", default=[128, 512, 2048, 8192, -1])
    ap.add_argument("--scoring", default=ABSOLUTE, choices=["absolute", "delta"])
    add_model_args(ap)
    args = ap.parse_args()

    t0 = time.time()
    device = resolve_device()
    REBUTTAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.domain}_{args.split}_seed{args.seed}"
    if args.transition != "mlp":
        tag += f"_{args.transition}"

    ds, tri = load_domain(args.domain, args.model_name)
    S = torch.as_tensor(ds.state_embs, dtype=torch.float32, device=device)
    A = torch.as_tensor(ds.action_embs, dtype=torch.float32, device=device)
    train_idx, valid_idx = make_split(ds, tri, args.split, args.seed)
    print(f"[{tag}] train {len(train_idx)}  test {len(valid_idx)}  states {tuple(S.shape)}", flush=True)

    model = build_model(S.shape[1], A.shape[1], args, device)
    n_par = sum(p.numel() for p in model.parameters())
    n_trans = sum(p.numel() for p in model.transition_model.parameters())
    print(f"[{tag}] params total {n_par:,} | transition-only {n_trans:,}", flush=True)

    train_transition(model, S, A, tri, train_idx, args, device)
    torch.save({"model": model.state_dict(), "args": vars(args)}, REBUTTAL_DIR / f"ckpt_{tag}.pt")

    pool = project_pool(model, S)
    trajs = build_trajectories(tri, valid_idx, args.max_trajs, args.seed)
    print(f"[{tag}] usable test trajectories: {len(trajs)}", flush=True)

    out = {"tag": tag, "args": vars(args), "n_train": len(train_idx), "n_test": len(valid_idx),
           "n_states": int(S.shape[0]), "n_actions": int(A.shape[0]),
           "params_total": n_par, "params_transition": n_trans}

    if trajs:
        out["E4_closed_loop"] = rollout(model, S, A, trajs, F.normalize(pool, dim=-1),
                                        device, prefix_curve=True)
    out["E1_pool_sweep"] = pool_sweep(model, S, A, tri, valid_idx, pool, device,
                                      args.pool_sizes, args.seed, mode=args.scoring)
    out["E3_open_set"] = open_set_abstention(model, S, A, tri, valid_idx, pool, device,
                                             args.seed, mode=args.scoring)
    out["runtime_s"] = time.time() - t0

    path = REBUTTAL_DIR / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[{tag}] wrote {path}  ({out['runtime_s']:.0f}s)", flush=True)

    if "E4_closed_loop" in out:
        e4 = out["E4_closed_loop"]
        print(f"[{tag}] E4 step_hit@1  tf={e4['teacher_forced']['step_hit@1']:.3f}  "
              f"cl={e4['closed_loop']['step_hit@1']:.3f}  "
              f"fr={e4['free_running']['step_hit@1']:.3f}", flush=True)
    print(f"[{tag}] E3 AUROC {out['E3_open_set']['auroc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
