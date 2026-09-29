"""The evaluation protocol used for the published numbers, reproduced exactly.

Reconstructed from `eval_metrics.evaluate_hit_across_states` at commit 7fb1e10 (deleted
by the refactor in e7aa70b, which is why nothing in the current package reproduced it).
Three details differ from `embedplan.evaluation.matched_pool_eval` and all three matter:

1. **Worst-case tie-breaking.** The published code ranks with `(scores >= true).sum()`,
   so a candidate scoring exactly equal to the ground truth counts *against* it. The
   refactored evaluators default to best-case (`>`). Under Extrapolation, where every
   distractor comes from the query's own problem, near-duplicate representations are
   common and the two conventions diverge sharply.

2. **Pool construction differs by split.** Per the paper's Metrics paragraph: under
   Interpolation the 127 distractors are drawn uniformly from all domain states; under
   Extrapolation they are drawn exclusively from the query's own problem instance.
   `matched_pool_eval` uses problem-grouped batches for *both*, which makes
   Interpolation harder than the published protocol.

3. **Best-epoch checkpoint selection.** The published `best_hit@5` is the maximum over
   training epochs of the validation metric, where the validation set *is* the test
   set. This is an optimistic selection rule; `final` is reported alongside it here so
   the size of that optimism is visible rather than assumed.
"""

from typing import Dict, Optional

import numpy as np
import torch

from embedplan.data import ProblemGroupedBatchSampler
from embedplan.evaluation import _rank_on_diagonal
from embedplan.scoring import ABSOLUTE, hit_at_k, predict, rank_in_candidates

TOPK = (1, 5, 10)


@torch.no_grad()
def paper_hit(model, S, A, tri, ds, valid_idx, device, split: str, seed: int,
              pool_size: int = 128, max_q: int = 4000, mode: str = ABSOLUTE,
              action_index: bool = False, projected_pool: Optional[torch.Tensor] = None
              ) -> Dict[str, float]:
    """Hit@k under the published protocol. `split` selects the pool construction."""
    model.eval()
    if split == "problem_grouped":
        # same-problem distractors: the batch is the pool, ranked on the diagonal
        sampler = ProblemGroupedBatchSampler(ds, pool_size, indices=valid_idx,
                                             shuffle_problems=False,
                                             shuffle_within_problem=False, seed=seed)
        s_i, a_i, p_i = (tri[c].to_numpy() for c in ("s_emb_idx", "a_idx", "sp_emb_idx"))
        hits = {k: 0 for k in TOPK}
        total, pools = 0, []
        for batch in sampler:
            if len(batch) < 2:
                continue
            bs = torch.as_tensor(s_i[batch], device=device, dtype=torch.long)
            ba = torch.as_tensor(a_i[batch], device=device, dtype=torch.long)
            bp = torch.as_tensor(p_i[batch], device=device, dtype=torch.long)
            preds = model(S[bs], ba if action_index else A[ba])
            anchor = model.state_projection_head(S[bs])
            cand = model.state_projection_head(S[bp])
            ranks = _rank_on_diagonal(preds, cand, anchor, mode, best_case_ties=False)
            for k in TOPK:
                hits[k] += (ranks <= k).sum().item()
            total += len(batch)
            pools.append(len(batch))
        return {**{f"hit@{k}": hits[k] / max(1, total) for k in TOPK},
                "n_queries": total, "mean_pool": float(np.mean(pools)) if pools else 0.0}

    # Interpolation: 127 distractors drawn uniformly from all domain states
    rng = np.random.default_rng(seed)
    q = (valid_idx if len(valid_idx) <= max_q
         else rng.choice(valid_idx, max_q, replace=False).tolist())
    cols = ("s_emb_idx", "a_idx", "sp_emb_idx")
    s_i, a_i, p_i = (torch.as_tensor(tri[c].to_numpy()[q], device=device, dtype=torch.long)
                     for c in cols)
    preds = predict(model, S, A, s_i, a_i, action_index=action_index)
    anchor = model.state_projection_head(S[s_i])
    pool = projected_pool if projected_pool is not None else model.state_projection_head(S)
    P = pool.shape[0]
    # Drawn with replacement from all states, as for the published numbers, so a distractor can
    # occasionally be the true next state itself; under worst-case ties that copy then counts
    # against the truth (at most ~(pool_size - 1) / P of queries). Kept to reproduce the paper.
    distractors = torch.as_tensor(rng.integers(0, P, size=(len(q), pool_size - 1)), device=device)
    cand = torch.cat([p_i.unsqueeze(1), distractors], dim=1)
    ranks = rank_in_candidates(preds, pool[cand], anchor=anchor, mode=mode,
                               best_case_ties=False)
    return {**hit_at_k(ranks, TOPK), "n_queries": int(len(q)), "mean_pool": float(pool_size)}
