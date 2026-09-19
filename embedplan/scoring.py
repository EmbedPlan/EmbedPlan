"""Retrieval scoring rules and Hit@k.

Two ways to rank a candidate next-state c against a prediction:

ABSOLUTE  cos(pred, z_c)
    What the submission uses.

DELTA     cos(pred - z_s, z_c - z_s)
    Scores the *displacement* rather than the destination. Under Extrapolation
    every distractor is drawn from the query's own problem, so all candidates
    share a large common component (same objects, same problem text, same goal
    text). That component is identical across candidates and therefore carries no
    discriminative signal, but it dominates the cosine and compresses the score
    differences that do matter. Subtracting the anchor z_s cancels it and puts
    the whole budget on the direction of change.

Both operate in the projected space: `pred` is the transition network's output,
`z_c`/`z_s` are pool states pushed through the state projection head.

Tie-breaking: the two evaluation paths in the pre-refactor code disagreed —
eval_metrics used worst-case (`>=`) and the pool sweep used best-case (`>`).
The submitted E1/E4 numbers come from the best-case path, so that is the default
here. With continuous cosine scores exact ties are vanishingly rare; the flag
exists so the choice is visible rather than accidental.
"""

from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

ABSOLUTE = "absolute"
DELTA = "delta"
MODES = (ABSOLUTE, DELTA)


def _prepare(pred: torch.Tensor, cand: torch.Tensor, anchor: Optional[torch.Tensor], mode: str):
    """Apply the scoring rule, then L2-normalize both sides.

    cand is (P, D) for a shared pool or (Q, C, D) for per-query pools; anchor is
    (Q, D) and is required for DELTA.

    DELTA takes the displacement *on the unit sphere*. The transition network ends
    in a LayerNorm, so predictions come out at a different radius than projected
    states (measured: 11.1 vs 26.7). A cosine objective never had to fix that —
    it is scale-invariant — but a displacement is not: with ||pred|| << ||z_s||,
    `pred - z_s` collapses to `-z_s` for every query and the ranking is destroyed.
    Projecting all three onto the unit sphere first removes the free scale.
    """
    if mode not in MODES:
        raise ValueError(f"unknown scoring mode {mode!r}, expected one of {MODES}")
    if mode == ABSOLUTE:
        return F.normalize(pred, dim=-1), F.normalize(cand, dim=-1)
    if anchor is None:
        raise ValueError("DELTA scoring requires the projected current state as `anchor`")
    p, c, s = F.normalize(pred, dim=-1), F.normalize(cand, dim=-1), F.normalize(anchor, dim=-1)
    s_b = s if c.dim() == 2 else s.unsqueeze(1)
    return F.normalize(p - s, dim=-1), F.normalize(c - s_b, dim=-1)


def _delta_scores_shared_pool(pred, pool, anchor, eps: float = 1e-8):
    """Unit-sphere DELTA for a pool shared across queries, in (Q, P) memory.

    Materializing (c_hat - s_hat) would need (Q, P, D) — 11B floats for a 43K-state
    pool. With everything on the unit sphere the inner products collapse:

        <p_hat - s_hat, c_hat - s_hat> = <p_hat, c_hat> - <p_hat, s_hat> - <s_hat, c_hat> + 1
        ||c_hat - s_hat||^2            = 2 - 2 <s_hat, c_hat>
        ||p_hat - s_hat||^2            = 2 - 2 <p_hat, s_hat>
    """
    p = F.normalize(pred, dim=-1)
    c = F.normalize(pool, dim=-1)
    s = F.normalize(anchor, dim=-1)

    pc = p @ c.T                                # (Q, P)
    sc = s @ c.T                                # (Q, P)
    ps = (p * s).sum(-1, keepdim=True)          # (Q, 1)

    num = pc - ps - sc + 1.0
    den_p = (2.0 - 2.0 * ps).clamp_min(0).sqrt()    # (Q, 1)
    den_c = (2.0 - 2.0 * sc).clamp_min(0).sqrt()    # (Q, P)
    return num / (den_p * den_c + eps)


def rank_in_pool(pred, pool, target_idx, anchor=None, mode=ABSOLUTE, best_case_ties=True,
                 chunk=2048, exclude_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rank of the true next state among every state in `pool`. Returns (Q,), 1-indexed.

    Chunked over queries so a 46K-state pool against 30K queries stays in memory.

    `exclude_idx` (Q,) removes one pool row per query from contention — pass the
    query's own state index when evaluating baselines that would otherwise
    trivially retrieve s itself (the identity predictor ranks s first by
    construction, which says nothing about whether it found s').
    """
    ranks = []
    for i in range(0, pred.shape[0], chunk):
        sl = slice(i, i + chunk)
        if mode == ABSOLUTE:
            pn, cn = _prepare(pred[sl], pool, None, mode)
            scores = pn @ cn.T
        else:
            if anchor is None:
                raise ValueError("DELTA scoring requires the projected current state as `anchor`")
            scores = _delta_scores_shared_pool(pred[sl], pool, anchor[sl])
        if exclude_idx is not None:
            scores = scores.scatter(1, exclude_idx[sl].unsqueeze(1), float("-inf"))
        true = scores.gather(1, target_idx[sl].unsqueeze(1))
        cmp = scores > true if best_case_ties else scores >= true
        ranks.append(cmp.sum(1) + (1 if best_case_ties else 0))
    return torch.cat(ranks)


def rank_in_candidates(pred, cand, anchor=None, mode=ABSOLUTE, best_case_ties=True) -> torch.Tensor:
    """Rank of the true next state within per-query candidate sets.

    `cand` is (Q, C, D) with the true next state at position 0. Returns (Q,), 1-indexed.
    """
    pn, cn = _prepare(pred, cand, anchor, mode)
    scores = torch.einsum("qd,qcd->qc", pn, cn)
    true = scores[:, :1]
    cmp = scores > true if best_case_ties else scores >= true
    return cmp.sum(1) + (1 if best_case_ties else 0)


def hit_at_k(ranks: torch.Tensor, topk: Iterable[int] = (1, 5, 10)) -> Dict[str, float]:
    return {f"hit@{k}": (ranks <= k).float().mean().item() for k in topk}


@torch.no_grad()
def project_pool(model, S: torch.Tensor, chunk: int = 4096, normalize: bool = False) -> torch.Tensor:
    """Push every state embedding through the state projection head.

    Defaults to *unnormalized*: the scoring functions normalize internally, and
    DELTA must subtract the anchor before normalizing or the geometry changes.
    Cosine is scale-invariant, so ABSOLUTE is unaffected by which you pass.
    Pass normalize=True only for code that consumes the pool directly (rollout).
    """
    out = [model.state_projection_head(S[i:i + chunk]) for i in range(0, len(S), chunk)]
    out = torch.cat(out)
    return F.normalize(out, dim=-1) if normalize else out


@torch.no_grad()
def predict(model, S, A, s_idx, a_idx, chunk: int = 1024, action_index: bool = False) -> torch.Tensor:
    """Batched next-state prediction for a set of (state, action) index pairs.

    `action_index` passes the action id instead of its embedding, for models keyed
    by action identity (the offset baseline's per-action displacement table).
    """
    out = []
    for i in range(0, len(s_idx), chunk):
        sl = slice(i, i + chunk)
        out.append(model(S[s_idx[sl]], a_idx[sl] if action_index else A[a_idx[sl]]))
    return torch.cat(out)
