"""Multi-step rollout under three feedback regimes.

teacher_forced   the true state is re-supplied at every step, so errors cannot
                 accumulate. This is what the submission's plan-level tables report.
closed_loop      the prediction is snapped to the nearest pool state and that
                 state is fed back — closed loop with per-step re-grounding.
free_running     the raw predicted latent is fed back, never snapped. No pool is
                 consulted for feedback, so this is the pool-free regime.

The tf-vs-closed_loop gap answers "you never closed the loop"; the
closed_loop-vs-free_running gap isolates re-grounding as the mechanism.
"""

import numpy as np
import torch
import torch.nn.functional as F

REGIMES = ("teacher_forced", "closed_loop", "free_running")


def _pack(trajs, device):
    """Ragged trajectories -> padded (N, L) index tensors with -1 as the pad."""
    L = max(len(t[0]) for t in trajs)
    N = len(trajs)
    acts = torch.full((N, L), -1, device=device, dtype=torch.long)
    tgts = torch.full((N, L), -1, device=device, dtype=torch.long)
    for i, (_, a, sp) in enumerate(trajs):
        acts[i, :len(a)] = torch.as_tensor(a.copy(), device=device)
        tgts[i, :len(sp)] = torch.as_tensor(sp.copy(), device=device)
    s0 = torch.tensor([t[0][0] for t in trajs], device=device, dtype=torch.long)
    return s0, acts, tgts, N, L


@torch.no_grad()
def rollout(model, S, A, trajs, pool_norm, device, topk=(1, 5), pool_idx=None,
            prefix_curve=False):
    """Roll out every trajectory under all three regimes. Vectorized across trajectories.

    `pool_norm` is the L2-normalized projected candidate pool. `pool_idx` maps pool
    rows back to global state ids; pass it when the pool is a subset, so a snapped
    candidate can be re-embedded for the next step.
    """
    model.eval()
    s0, acts, tgts, N, L = _pack(trajs, device)

    if pool_idx is None:
        g2l = None
    else:
        pool_idx = torch.as_tensor(pool_idx, device=device, dtype=torch.long)
        g2l = torch.full((S.shape[0],), -1, device=device, dtype=torch.long)
        g2l[pool_idx] = torch.arange(len(pool_idx), device=device)

    results = {}
    for mode in REGIMES:
        cur_raw = S[s0] if mode != "free_running" else None
        cur_proj = model.state_projection_head(S[s0]) if mode == "free_running" else None
        hits = {k: torch.zeros(N, L, device=device) for k in topk}
        steps = torch.zeros(N, L, dtype=torch.bool, device=device)

        for t in range(L):
            act = acts[:, t]
            live = act >= 0
            if not live.any():
                break
            steps[:, t] = live
            a_emb = A[act.clamp(min=0)]

            if mode == "free_running":
                pred = model.transition_model(cur_proj, model.action_projection_head(a_emb))
            else:
                pred = model(cur_raw, a_emb)

            scores = F.normalize(pred, dim=-1) @ pool_norm.T
            top = scores.topk(min(max(topk), pool_norm.shape[0]), dim=1).indices

            tgt = tgts[:, t]
            tgt_local = tgt.clamp(min=0) if g2l is None else g2l[tgt.clamp(min=0)]
            for k in topk:
                hits[k][:, t] = ((top[:, :k] == tgt_local.unsqueeze(1)).any(1) & live).float()

            if mode == "teacher_forced":
                cur_raw = torch.where(live.unsqueeze(1), S[tgt.clamp(min=0)], cur_raw)
            elif mode == "closed_loop":
                snapped = top[:, 0] if g2l is None else pool_idx[top[:, 0]]
                cur_raw = torch.where(live.unsqueeze(1), S[snapped], cur_raw)
            else:
                cur_proj = torch.where(live.unsqueeze(1), pred, cur_proj)

        results[mode] = _summarize(hits, steps, topk, N, L, prefix_curve)

    results["_meta"] = {
        "n_trajectories": N,
        "max_len": int(L),
        "mean_len": float(np.mean([len(t[0]) for t in trajs])),
        "pool_size": int(pool_norm.shape[0]),
    }
    return results


def _summarize(hits, steps, topk, N, L, prefix_curve):
    n_steps = steps.sum().item()
    out = {}
    for k in topk:
        out[f"step_hit@{k}"] = (hits[k].sum() / max(1, n_steps)).item()
        out[f"exact_hit@{k}"] = (hits[k].sum(1) == steps.sum(1)).float().mean().item()

    # per-depth breakdown: accuracy at step t among trajectories that reach step t
    depth = {}
    for t in range(L):
        live = steps[:, t]
        if live.any():
            depth[t + 1] = {"n": int(live.sum().item()),
                            **{f"hit@{k}": hits[k][live, t].mean().item() for k in topk}}
    out["by_depth"] = depth

    if prefix_curve:
        curve, cum = {}, torch.ones(N, dtype=torch.bool, device=steps.device)
        for t in range(L):
            live = steps[:, t]
            cum = cum & (~live | (hits[1][:, t] > 0))
            if live.any():
                curve[t + 1] = cum[live].float().mean().item()
        out["prefix_success@1"] = curve
    return out
