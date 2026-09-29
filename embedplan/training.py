"""The shared training loop.

Embeddings live on the GPU as two lookup tables and batches are gathered by
index, so there is no DataLoader in the hot path. A domain trains in ~15 min.

Batch composition matters and is tied to the split: under Extrapolation we build
batches within a problem, so in-batch negatives are same-problem near-misses and
match the distractor distribution used at evaluation.
"""

from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch

from embedplan.losses import (compute_action_loss, compute_delta_action_loss,
                              compute_delta_infonce_loss, compute_infonce_loss)


def _make_batches(n, batch_size, rng, groups=None):
    if groups is None:
        perm = rng.permutation(n)
        return [perm[i:i + batch_size] for i in range(0, n, batch_size)]
    batches = []
    for g in groups:
        gg = g.copy()
        rng.shuffle(gg)
        batches.extend([gg[i:i + batch_size] for i in range(0, len(gg), batch_size)])
    rng.shuffle(batches)
    return batches


def train_transition(model, S, A, tri, train_idx, args, device, log_every: int = 50, verbose: bool = True,
                     callback: Optional[Callable[[int, float], bool]] = None):
    """Train `model` in place over GPU-resident embedding tables. Returns the model.

    `callback(epoch, mean_loss)` runs after every epoch; returning True stops training early.
    """
    s_i = torch.as_tensor(tri["s_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
    a_i = torch.as_tensor(tri["a_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
    p_i = torch.as_tensor(tri["sp_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)

    groups = None
    if getattr(args, "split", "random") != "random":
        by_prob = defaultdict(list)
        for pos, pid in enumerate(tri["problem_idx"].to_numpy()[train_idx]):
            by_prob[int(pid)].append(pos)
        groups = [np.array(v) for v in by_prob.values()]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    n, B = len(train_idx), args.batch_size

    for ep in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for b in _make_batches(n, B, rng, groups):
            if len(b) < 2:
                continue
            bi = torch.as_tensor(b, device=device, dtype=torch.long)
            s, a, sp = S[s_i[bi]], A[a_i[bi]], S[p_i[bi]]

            pred = model(s, a)
            sp_proj = model.state_projection_head(sp)

            # `loss_space` selects the geometry the objective lives in. 'absolute' is the
            # published objective and the default, so every existing call site is
            # unchanged. 'delta' optimizes the displacement, which is where the signal
            # actually is under Extrapolation (see compute_delta_infonce_loss).
            if getattr(args, "loss_space", "absolute") == "delta":
                s_proj = model.state_projection_head(s)
                loss = compute_delta_infonce_loss(pred, sp_proj, s_proj, args.tau)
            else:
                loss = compute_infonce_loss(pred, sp_proj, args.tau)

            if args.action_weight > 0:
                m = min(max(2, int(len(b) ** 0.5)), len(b))
                if getattr(args, "loss_space", "absolute") == "delta":
                    loss = loss + args.action_weight * compute_delta_action_loss(
                        model, s, a, sp_proj, s_proj, args.tau, m, device)
                else:
                    loss = loss + args.action_weight * compute_action_loss(
                        model, s, a, sp_proj, args.tau, m, device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            seen += len(b)

        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"  epoch {ep:4d}  loss {total / max(1, seen):.4f}", flush=True)
        if callback is not None and callback(ep, total / max(1, seen)):
            break

    return model
