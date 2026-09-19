"""Evaluate EmbedPlan on the exact candidate pools used by the LLM ranking experiment.

Two purposes:

1. Fair LLM comparison. llm_transition_experiment.py ranks candidates drawn by
   ProblemGroupedBatchSampler (batch 128, same problem => hard, near-miss distractors).
   To compare like with like, EmbedPlan must be scored on the same pool construction.

2. De-confounding the generalization gap. The submission's Interpolation numbers use
   uniformly-drawn batch distractors while Extrapolation uses same-problem distractors,
   so split difficulty and negative-mining difficulty are entangled. Scoring both splits
   with problem-grouped pools separates them.

Usage:
    python -m experiments.matched_pool --tag logistics_random_seed0
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from embedplan import build_model, load_domain, make_split
from embedplan.config import REBUTTAL_DIR as OUT_DIR
from embedplan.data import ProblemGroupedBatchSampler


@torch.no_grad()
def eval_pools(model, S, A, tri, idx, device, batch_size, ds, seed=42):
    """Hit@k with the candidate pool = the problem-grouped batch, as in the LLM experiment."""
    model.eval()
    sampler = ProblemGroupedBatchSampler(ds, batch_size, indices=idx,
                                         shuffle_problems=False,
                                         shuffle_within_problem=False, seed=seed)
    s_i = tri["s_emb_idx"].to_numpy()
    a_i = tri["a_idx"].to_numpy()
    p_i = tri["sp_emb_idx"].to_numpy()

    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    pools = []
    for batch in sampler:
        if len(batch) < 2:
            continue
        b = np.asarray(batch)
        s = S[torch.as_tensor(s_i[b], device=device, dtype=torch.long)]
        a = A[torch.as_tensor(a_i[b], device=device, dtype=torch.long)]
        sp = S[torch.as_tensor(p_i[b], device=device, dtype=torch.long)]
        pred = F.normalize(model(s, a), dim=-1)
        pool = F.normalize(model.state_projection_head(sp), dim=-1)
        scores = pred @ pool.T                      # candidates = this batch's next-states
        diag = scores.diag().unsqueeze(1)
        rank = (scores >= diag).sum(1)              # worst-case tie-breaking, as in eval_metrics
        for k in hits:
            hits[k] += (rank <= min(k, len(b))).sum().item()
        total += len(b)
        pools.append(len(b))

    return {**{f"hit@{k}": hits[k] / max(1, total) for k in hits},
            "n_queries": total, "mean_pool": float(np.mean(pools)) if pools else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / f"ckpt_{args.tag}.pt", map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])

    ds, tri = load_domain(a.domain, a.model_name)
    S = torch.as_tensor(ds.state_embs, dtype=torch.float32, device=device)
    A = torch.as_tensor(ds.action_embs, dtype=torch.float32, device=device)
    _, valid_idx = make_split(ds, tri, a.split, a.seed)

    model = build_model(S.shape[1], A.shape[1], a, device)
    model.load_state_dict(ck["model"])

    res = eval_pools(model, S, A, tri, valid_idx, device, args.batch_size, ds)
    out = {"tag": args.tag, "domain": a.domain, "split": a.split,
           "pool_construction": "problem_grouped_batch", "results": res}
    json.dump(out, open(OUT_DIR / f"matchedpool_{args.tag}.json", "w"), indent=2)
    print(f"{args.tag}  [{a.split}]  pool~{res['mean_pool']:.0f}  "
          f"hit@1={res['hit@1']:.4f}  hit@5={res['hit@5']:.4f}  hit@10={res['hit@10']:.4f}  "
          f"(n={res['n_queries']})")


if __name__ == "__main__":
    main()
