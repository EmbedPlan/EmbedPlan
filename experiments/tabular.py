"""Tabular (one-hot / learned-lookup) transition baseline.

Isolates what the frozen LLM embedding contributes beyond state identity. Everything
matches the main harness except the inputs: instead of frozen Llama-3.3-70B embeddings,
states and actions are learned lookup tables trained from scratch. Same transition MLP,
same InfoNCE objective, same splits, same evaluation.

Expected contrast, and the reason the experiment is informative:
  interpolation  - held-out transitions involve states seen in training, so a lookup
                   table can learn them; the embedding may add little.
  extrapolation  - held-out problems contain states never seen, whose lookup rows stay
                   at initialization. Only a model with semantic input features can
                   generalize. This is what the frozen embedding buys.

Usage:
    python -m experiments.tabular --domain ferry --split random
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedplan import load_domain, make_split
from embedplan.config import RUNS_DIR as OUT_DIR
from embedplan.models import TransitionMLP
from embedplan.losses import compute_infonce_loss


class TabularTransition(nn.Module):
    """Learned state/action lookup tables feeding the same residual transition MLP."""

    def __init__(self, n_states, n_actions, dim, hidden, n_layers):
        super().__init__()
        self.state_emb = nn.Embedding(n_states, dim)
        self.action_emb = nn.Embedding(n_actions, dim)
        nn.init.normal_(self.state_emb.weight, std=0.02)
        nn.init.normal_(self.action_emb.weight, std=0.02)
        self.transition_model = TransitionMLP(d_state=dim, d_action=dim, hidden=hidden,
                                              n_layers=n_layers, use_layer_norm=True)

    def forward(self, s_idx, a_idx):
        return self.transition_model(self.state_emb(s_idx), self.action_emb(a_idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="ferry")
    ap.add_argument("--split", default="random", choices=["random", "problem_grouped"])
    ap.add_argument("--model_name", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_sizes", type=int, nargs="+", default=[128, 2048, -1])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    ds, tri = load_domain(args.domain, args.model_name)
    n_states = ds.state_embs.shape[0]
    n_actions = ds.action_embs.shape[0]
    train_idx, valid_idx = make_split(ds, tri, args.split, args.seed)

    s_all = torch.as_tensor(tri["s_emb_idx"].to_numpy(), device=device, dtype=torch.long)
    a_all = torch.as_tensor(tri["a_idx"].to_numpy(), device=device, dtype=torch.long)
    p_all = torch.as_tensor(tri["sp_emb_idx"].to_numpy(), device=device, dtype=torch.long)

    model = TabularTransition(n_states, n_actions, args.dim, args.hidden_size,
                              args.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    tr = np.asarray(train_idx)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        tot = seen = 0
        for i in range(0, len(tr), args.batch_size):
            b = torch.as_tensor(tr[perm[i:i + args.batch_size]], device=device, dtype=torch.long)
            if len(b) < 2:
                continue
            pred = model(s_all[b], a_all[b])
            tgt = model.state_emb(p_all[b])
            loss = compute_infonce_loss(pred, tgt, args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            seen += len(b)
        if ep % 100 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  loss {tot / max(1, seen):.4f}", flush=True)

    # evaluation: identical protocol to the frozen-embedding harness
    model.eval()
    with torch.no_grad():
        pool = F.normalize(model.state_emb.weight, dim=-1)
        q = np.asarray(valid_idx)
        if len(q) > 4000:
            q = rng.choice(q, 4000, replace=False)
        qi = torch.as_tensor(q, device=device, dtype=torch.long)
        preds = F.normalize(model(s_all[qi], a_all[qi]), dim=-1)
        tgt = p_all[qi]

        out = {}
        P = pool.shape[0]
        for size in args.pool_sizes:
            sz = P if size < 0 or size >= P else size
            if sz >= P:
                sc = preds @ pool.T
                rank = (sc > sc.gather(1, tgt.unsqueeze(1))).sum(1) + 1
            else:
                cand = torch.as_tensor(rng.integers(0, P, size=(len(q), sz - 1)), device=device)
                cand = torch.cat([tgt.unsqueeze(1), cand], dim=1)
                s = torch.einsum("qd,qcd->qc", preds, pool[cand])
                rank = (s > s[:, :1]).sum(1) + 1
            key = "full" if sz >= P else str(sz)
            out[key] = {f"hit@{k}": (rank <= k).float().mean().item() for k in (1, 5, 10)}
            out[key]["pool"] = int(sz)

        # how many held-out states were never seen in training (untrained lookup rows)
        seen_states = set(tri["s_emb_idx"].to_numpy()[train_idx]) | set(
            tri["sp_emb_idx"].to_numpy()[train_idx])
        test_states = set(tri["sp_emb_idx"].to_numpy()[q])
        unseen_frac = 1.0 - len(test_states & seen_states) / max(1, len(test_states))

    tag = f"tabular_{args.domain}_{args.split}_seed{args.seed}"
    res = {"tag": tag, "args": vars(args), "n_states": n_states, "n_actions": n_actions,
           "n_train": len(train_idx), "n_test": len(valid_idx),
           "frac_test_targets_unseen_in_train": unseen_frac, "pool_sweep": out}
    json.dump(res, open(OUT_DIR / f"{tag}.json", "w"), indent=2)
    print(f"{tag}: unseen-target fraction {unseen_frac:.3f}")
    for k, v in out.items():
        print(f"  |C|={v['pool']:>6}  hit@1={v['hit@1']:.4f}  hit@5={v['hit@5']:.4f}")


if __name__ == "__main__":
    main()
