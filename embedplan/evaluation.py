"""Single-step evaluations: pool scaling, matched pools, abstention, action disambiguation.

Everything routes through embedplan.scoring, so each of these can be run under
either the ABSOLUTE or the DELTA scoring rule by passing `mode`.
"""

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from embedplan.data import ProblemGroupedBatchSampler
from embedplan.scoring import (ABSOLUTE, hit_at_k, predict, rank_in_candidates, rank_in_pool)


def _query_indices(tri, valid_idx, device, max_q, rng):
    """Subsample the evaluation queries. Takes an rng rather than a seed so a caller
    that later draws distractors advances the same stream — the pre-refactor code
    shared one generator across both draws and the saved numbers depend on it."""
    q = valid_idx if len(valid_idx) <= max_q else rng.choice(valid_idx, max_q, replace=False).tolist()
    cols = ("s_emb_idx", "a_idx", "sp_emb_idx")
    return q, *(torch.as_tensor(tri[c].to_numpy()[q], device=device, dtype=torch.long) for c in cols)


@torch.no_grad()
def pool_sweep(model, S, A, tri, valid_idx, pool, device, sizes: Sequence[int], seed: int,
               max_q: int = 4000, mode: str = ABSOLUTE, exclude_self: bool = False,
               action_index: bool = False, best_case_ties: bool = True) -> Dict[str, Dict]:
    """Hit@k as the candidate pool grows. The true target is always included;
    the remainder are sampled uniformly from the domain's states.

    exclude_self  drop the query's own state from contention. Required when
                  comparing against the identity baseline, which otherwise
                  retrieves s at rank 1 and tells us nothing about whether it
                  found s'. Off by default so published numbers reproduce.
    action_index  pass raw action indices to the model rather than embeddings
                  (the offset baseline is keyed by action id).
    best_case_ties  True reproduces the published pool-sweep numbers. False is the
                  pre-refactor convention (`(scores >= true).sum()`), which is what the
                  paper's evaluate_hit_across_states used; pass False to reproduce it.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    q, s_i, a_i, p_i = _query_indices(tri, valid_idx, device, max_q, rng)
    preds = predict(model, S, A, s_i, a_i, action_index=action_index)
    anchor = model.state_projection_head(S[s_i])
    P = pool.shape[0]

    out = {}
    for size in sizes:
        sz = P if size < 0 else min(size, P)
        if sz >= P:
            ranks = rank_in_pool(preds, pool, p_i, anchor=anchor, mode=mode,
                                 best_case_ties=best_case_ties,
                                 exclude_idx=s_i if exclude_self else None)
        else:
            distractors = torch.as_tensor(rng.integers(0, P, size=(len(q), sz - 1)), device=device)
            if exclude_self:
                # redraw any distractor that landed on the query's own state
                clash = distractors == s_i.unsqueeze(1)
                if clash.any():
                    distractors[clash] = (distractors[clash] + 1) % P
            cand = torch.cat([p_i.unsqueeze(1), distractors], dim=1)
            ranks = rank_in_candidates(preds, pool[cand], anchor=anchor, mode=mode,
                                       best_case_ties=best_case_ties)
        out["full" if size < 0 else str(size)] = {**hit_at_k(ranks), "pool": int(sz)}
    return out


@torch.no_grad()
def matched_pool_eval(model, S, A, tri, ds, idx, device, batch_size: int = 128,
                      seed: int = 42, mode: str = ABSOLUTE,
                      action_index: bool = False,
                      best_case_ties: bool = True) -> Dict[str, float]:
    """Hit@k where the candidate pool is a problem-grouped batch.

    This is the pool construction the LLM ranking experiment uses, so it is the
    only like-for-like comparison against those numbers. It also de-confounds the
    generalization gap: scoring both splits with same-problem distractors
    separates split difficulty from negative-mining difficulty.

    action_index  pass raw action ids instead of embeddings, for models keyed by
                  action identity (the offset baselines). Default off, so every
                  published call site is unchanged.
    best_case_ties  True reproduces the published numbers. False gives the pessimistic
                  bound, which is the one to quote for any collapsed representation —
                  see _rank_on_diagonal.
    """
    model.eval()
    sampler = ProblemGroupedBatchSampler(ds, batch_size, indices=idx, shuffle_problems=False,
                                         shuffle_within_problem=False, seed=seed)
    s_i, a_i, p_i = (tri[c].to_numpy() for c in ("s_emb_idx", "a_idx", "sp_emb_idx"))

    hits = {1: 0, 5: 0, 10: 0}
    total, pool_sizes = 0, []
    for batch in sampler:
        if len(batch) < 2:
            continue
        bs = torch.as_tensor(s_i[batch], device=device, dtype=torch.long)
        ba = torch.as_tensor(a_i[batch], device=device, dtype=torch.long)
        bp = torch.as_tensor(p_i[batch], device=device, dtype=torch.long)

        preds = model(S[bs], ba if action_index else A[ba])
        anchor = model.state_projection_head(S[bs])
        cand = model.state_projection_head(S[bp])  # (B, D), shared across the batch
        ranks = _rank_on_diagonal(preds, cand, anchor, mode, best_case_ties)

        for k in hits:
            hits[k] += (ranks <= k).sum().item()
        total += len(batch)
        pool_sizes.append(len(batch))

    return {**{f"hit@{k}": hits[k] / max(1, total) for k in hits},
            "n_queries": total, "mean_pool": float(np.mean(pool_sizes)) if pool_sizes else 0.0}


def _rank_on_diagonal(preds, cand, anchor, mode, best_case_ties: bool = True):
    """Rank of candidate i for query i, where every query scores the same (B, D) pool.

    Under DELTA each query subtracts its own anchor, so the candidate matrix
    becomes query-specific and the comparison is (B, B, D) rather than (B, D).

    `best_case_ties` matters here far more than the scoring.py docstring suggests. That
    note says exact ties are "vanishingly rare" with continuous cosine scores, which
    holds for a well-behaved encoder but fails badly for collapsed representations:
    delete the state text from the prompt and every candidate in a problem-grouped batch
    becomes the identical vector, so nothing is strictly greater and every query scores
    rank 1 — a measured matched Hit@5 of 1.0000 for a representation carrying zero state
    information. Bag features collapse too (ferry TF-IDF: 35% of queries have a duplicated
    true candidate, against 3.7% for BGE-M3). Roughly 12% of ties are the data's own fault
    — the same next state is reachable by two transitions in a batch — and that floor
    applies to every arm. Pass False to get the pessimistic bound.
    """
    from embedplan.scoring import _prepare
    if mode == ABSOLUTE:
        pn, cn = _prepare(preds, cand, None, mode)
        scores = pn @ cn.T
    else:
        pn, cn = _prepare(preds, cand.unsqueeze(0).expand(preds.size(0), -1, -1), anchor, mode)
        scores = torch.einsum("qd,qcd->qc", pn, cn)
    true = scores.diagonal().unsqueeze(1)
    cmp = scores > true if best_case_ties else scores >= true
    return cmp.sum(1) + (1 if best_case_ties else 0)


@torch.no_grad()
def open_set_abstention(model, S, A, tri, valid_idx, pool, device, seed: int,
                        max_q: int = 4000, mode: str = ABSOLUTE) -> Dict[str, float]:
    """Can retrieval tell that the answer is absent?

    answerable   : max similarity over a pool containing the true next state
    unanswerable : max similarity over the same pool with it removed

    AUROC over the two score populations measures the quality of an abstention
    signal. At chance, forced-argmax retrieval cannot decline to answer.
    """
    model.eval()
    q, s_i, a_i, p_i = _query_indices(tri, valid_idx, device, max_q, np.random.default_rng(seed))
    preds = predict(model, S, A, s_i, a_i)
    anchor = model.state_projection_head(S[s_i])

    from embedplan.scoring import _delta_scores_shared_pool, _prepare
    if mode == ABSOLUTE:
        pn, cn = _prepare(preds, pool, None, mode)
        scores = pn @ cn.T
    else:  # a pool shared by every query: DELTA needs each query's own anchor
        scores = _delta_scores_shared_pool(preds, pool, anchor)

    in_pool = scores.max(1).values
    masked = scores.clone().scatter_(1, p_i.unsqueeze(1), -1e9)
    out_pool = masked.max(1).values

    gap = in_pool - out_pool
    return {
        "auroc": _auroc(torch.cat([in_pool, out_pool]),
                        torch.cat([torch.ones_like(in_pool), torch.zeros_like(out_pool)])),
        "mean_score_answerable": in_pool.mean().item(),
        "mean_score_unanswerable": out_pool.mean().item(),
        "mean_margin": gap.mean().item(),
        "frac_margin_positive": (gap > 0).float().mean().item(),
        "n_queries": len(q),
        "pool_size": int(pool.shape[0]),
    }


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels[scores.argsort(descending=True)].float()
    tp, fp = torch.cumsum(y, 0), torch.cumsum(1 - y, 0)
    if tp[-1] == 0 or fp[-1] == 0:
        return float("nan")
    return torch.trapz(tp / tp[-1], fp / fp[-1]).item()


@torch.no_grad()
def action_disambiguation(model, S, A, tri, valid_idx, device, batch: int = 128,
                          seed: int = 0, max_q: int = 4000, topk=(1, 5, 10)) -> Dict[str, float]:
    """Among the actions available in a batch, does the correct one place its
    prediction closest to the true s'? Tests that action effects are separable
    rather than the model mapping states to generic neighbours."""
    model.eval()
    q, s_i, a_i, p_i = _query_indices(tri, valid_idx, device, max_q, np.random.default_rng(seed))
    hits = {k: 0 for k in topk}
    total = 0
    for i in range(0, len(q), batch):
        s, a, sp = S[s_i[i:i + batch]], A[a_i[i:i + batch]], S[p_i[i:i + batch]]
        B = s.size(0)
        if B < 2:
            continue
        preds = model(s.repeat_interleave(B, 0), a.repeat(B, 1)).view(B, B, -1)
        preds = F.normalize(preds, dim=-1)
        target = F.normalize(model.state_projection_head(sp), dim=-1)
        scores = (preds * target.unsqueeze(1)).sum(-1)
        ranks = (scores >= scores.diagonal().unsqueeze(1)).sum(1)
        for k in topk:
            hits[k] += (ranks <= k).sum().item()
        total += B
    return {f"acc_action@{k}": hits[k] / max(1, total) for k in topk}


# ---------------------------------------------------------------------------
# Loader-based evaluators used by experiments/train.py and train_multi_domain.py.
#
# These are the evaluators behind the main-table Hit@k and action Acc@k numbers,
# restored verbatim from the pre-refactor `eval_metrics.py` (only imports changed).
# Each validation batch is its own candidate pool: row i's true next state is
# column i, every other next state in the batch is a distractor, and a candidate
# scoring exactly equal to the true one counts against it (worst-case ties).
# ---------------------------------------------------------------------------

def _compute_ranks(scores: torch.Tensor, break_ties: str = "worst") -> torch.Tensor:
    scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    if break_ties == "random":
        scores = scores + 1e-6 * torch.randn_like(scores)

    diag = torch.diag(scores).unsqueeze(1)
    if break_ties == "worst":
        ranks = (scores >= diag).sum(dim=1)
    else:
        ranks = 1 + (scores > diag).sum(dim=1)
    return ranks


@torch.inference_mode()
def eval_action_disambiguation(model, loader, device, topk=(1, 5, 10)):
    """Acc@k: for each state, does its own action rank the true next state highest
    among the actions paired with the other states in the batch?"""
    model.eval()
    hit_at = {k: 0 for k in topk}
    total = 0
    for batch in loader:
        s = batch["s_emb"].to(device)
        a = batch["a_emb"].to(device)
        sp = batch["sp_emb"].to(device)
        B = s.size(0)
        s_repeated = s.repeat_interleave(B, dim=0)
        a_tiled = a.repeat(B, 1)

        preds = model(s_repeated, a_tiled)
        preds = preds.view(B, B, -1)
        sp_proj = model.state_projection_head(sp)
        sp_proj = F.normalize(sp_proj, dim=-1)
        preds = F.normalize(preds, dim=-1)
        scores = (preds @ sp_proj.unsqueeze(-1)).squeeze(-1)

        ranks = _compute_ranks(scores, break_ties="worst")
        for k in topk:
            hit_at[k] += (ranks <= k).sum().item()
        total += B

    return {f"acc_action@{k}": hit_at[k] / max(1, total) for k in topk}


@torch.inference_mode()
def evaluate_hit_across_states(model, loader, device, cfg) -> Dict[str, float]:
    """Hit@k of the true next state among the batch's next states (the published protocol)."""
    from embedplan.models import ProjectedTransitionModel

    model.eval()
    hit_at = {k: 0 for k in cfg.topk}
    total = 0
    has_projection = isinstance(model, ProjectedTransitionModel)

    for _, batch in enumerate(loader):
        s = batch["s_emb"].to(device)
        a = batch["a_emb"].to(device)
        sp = batch["sp_emb"].to(device)

        if has_projection:
            pred = model(s, a)
            sp_proj = model.state_projection_head(sp)
            pred_norm = F.normalize(pred, dim=-1)
            sp_proj_norm = F.normalize(sp_proj, dim=-1)
            scores = pred_norm @ sp_proj_norm.T
        else:
            pred = model(s, a)
            pred_norm = F.normalize(pred, p=2, dim=-1)
            sp_norm = F.normalize(sp, p=2, dim=-1)
            scores = pred_norm @ sp_norm.T

        rank = _compute_ranks(scores, break_ties="worst")
        max_k = scores.size(1)
        for k in cfg.topk:
            kk = min(k, max_k)
            hit_at[k] += (rank <= kk).float().sum().item()
        total += s.size(0)

    return {f"hit@{k}": hit_at[k] / max(1, total) for k in cfg.topk}
