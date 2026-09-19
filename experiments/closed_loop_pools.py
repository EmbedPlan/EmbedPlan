"""Closed-loop rollout as a function of candidate-pool size, from a saved checkpoint.

The main harness rolls out against the full domain state pool (hardest setting).
The submission's headline numbers use a 128-candidate pool, so to compare like with
like we re-run the same trained model at several pool sizes. No retraining.

Pool construction mirrors the paper: every true successor needed by the evaluated
trajectories, plus uniformly sampled distractors up to the target size.

Usage:
    python -m experiments.closed_loop_pools --tag logistics_random_seed0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from embedplan import build_model, load_domain, make_split
from embedplan.config import REBUTTAL_DIR as OUT_DIR
from embedplan.data import build_trajectories


@torch.no_grad()
def rollout(model, S, A, trajs, pool_idx, device, topk=(1, 5)):
    """Same three regimes as the main harness, restricted to a candidate pool."""
    model.eval()
    pool_idx = torch.as_tensor(pool_idx, device=device, dtype=torch.long)
    pool = F.normalize(model.state_projection_head(S[pool_idx]), dim=-1)
    # map global state id -> position in pool
    g2l = torch.full((S.shape[0],), -1, device=device, dtype=torch.long)
    g2l[pool_idx] = torch.arange(len(pool_idx), device=device)

    L = max(len(t[0]) for t in trajs)
    N = len(trajs)
    s0 = torch.tensor([t[0][0] for t in trajs], device=device, dtype=torch.long)
    acts = torch.full((N, L), -1, device=device, dtype=torch.long)
    tgts = torch.full((N, L), -1, device=device, dtype=torch.long)
    for i, (s, a, sp) in enumerate(trajs):
        acts[i, :len(a)] = torch.as_tensor(a.copy(), device=device)
        tgts[i, :len(sp)] = torch.as_tensor(sp.copy(), device=device)

    out = {}
    for mode in ("teacher_forced", "closed_loop", "free_running"):
        cur_raw = S[s0] if mode != "free_running" else None
        cur_proj = model.state_projection_head(S[s0]) if mode == "free_running" else None
        hits = {k: torch.zeros(N, L, device=device) for k in topk}
        steps = torch.zeros(N, L, dtype=torch.bool, device=device)

        for t in range(L):
            act = acts[:, t]
            m = act >= 0
            if not m.any():
                break
            steps[:, t] = m
            a_emb = A[act.clamp(min=0)]
            if mode == "free_running":
                pred = model.transition_model(cur_proj, model.action_projection_head(a_emb))
            else:
                pred = model(cur_raw, a_emb)
            scores = F.normalize(pred, dim=-1) @ pool.T
            top = scores.topk(min(max(topk), pool.shape[0]), dim=1).indices
            tgt_local = g2l[tgts[:, t].clamp(min=0)]
            for k in topk:
                hits[k][:, t] = ((top[:, :k] == tgt_local.unsqueeze(1)).any(1) & m).float()
            if mode == "teacher_forced":
                cur_raw = torch.where(m.unsqueeze(1), S[tgts[:, t].clamp(min=0)], cur_raw)
            elif mode == "closed_loop":
                cur_raw = torch.where(m.unsqueeze(1), S[pool_idx[top[:, 0]]], cur_raw)
            else:
                cur_proj = torch.where(m.unsqueeze(1), pred, cur_proj)

        ns = steps.sum().item()
        r = {}
        for k in topk:
            r[f"step_hit@{k}"] = (hits[k].sum() / max(1, ns)).item()
            r[f"exact_hit@{k}"] = ((hits[k].sum(1) == steps.sum(1)).float().mean().item())
        out[mode] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 2048, -1])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / f"ckpt_{args.tag}.pt", map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])

    ds, tri = load_domain(a.domain, a.model_name)
    S = torch.as_tensor(ds.state_embs, dtype=torch.float32, device=device)
    A = torch.as_tensor(ds.action_embs, dtype=torch.float32, device=device)
    _, valid_idx = make_split(ds, tri, a.split, a.seed)
    trajs = build_trajectories(tri, valid_idx, a.max_trajs, a.seed)

    model = build_model(S.shape[1], A.shape[1], a, device)
    model.load_state_dict(ck["model"])

    needed = np.unique(np.concatenate([t[2] for t in trajs] + [t[0] for t in trajs]))
    rng = np.random.default_rng(a.seed)
    P = S.shape[0]
    res = {}
    for size in args.sizes:
        if size < 0 or size >= P:
            pool = np.arange(P)
            key = "full"
        else:
            size = max(size, len(needed))
            extra = rng.choice(np.setdiff1d(np.arange(P), needed),
                               max(0, size - len(needed)), replace=False)
            pool = np.concatenate([needed, extra])
            key = str(size)
        r = rollout(model, S, A, trajs, pool, device)
        r["pool_size"] = int(len(pool))
        res[key] = r
        print(f"{args.tag}  |C|={len(pool):>6}  tf={r['teacher_forced']['step_hit@1']:.3f}  "
              f"cl={r['closed_loop']['step_hit@1']:.3f}  fr={r['free_running']['step_hit@1']:.3f}",
              flush=True)

    out = OUT_DIR / f"closedloop_pools_{args.tag}.json"
    json.dump({"tag": args.tag, "n_trajectories": len(trajs),
               "n_states": int(P), "results": res}, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
