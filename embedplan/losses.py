"""Training objectives.

L = L_state + lambda * L_action

L_state    InfoNCE between the predicted next-state embedding and the true one,
           against in-batch negatives.
L_action   the correct action must place its prediction closer to s' than the
           other actions sampled from the same batch do. This is what stops the
           model collapsing to a generic "small step from s" predictor.
"""

import torch
import torch.nn.functional as F


def compute_infonce_loss(pred: torch.Tensor, true: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    p, t = F.normalize(pred, dim=-1), F.normalize(true, dim=-1)
    logits = (p @ t.T) / tau
    return F.cross_entropy(logits, torch.arange(p.size(0), device=p.device))


def compute_action_loss(model, s, a, sp_proj, tau: float, m: int, device) -> torch.Tensor:
    """Cross-entropy over an m x m grid of (state, action) pairs.

    Row i holds state i paired with every sampled action; the correct entry is the
    diagonal. m is kept small (sqrt of batch) because the grid is quadratic.
    """
    sel = torch.randperm(s.size(0), device=device)[:m]
    ss, aa, pp = s[sel], a[sel], sp_proj[sel]
    preds = model(ss.repeat_interleave(m, 0), aa.repeat(m, 1)).view(m, m, -1)
    preds = F.normalize(preds, dim=-1)
    targets = F.normalize(pp, dim=-1)
    logits = (preds * targets.unsqueeze(1)).sum(-1) / tau
    return F.cross_entropy(logits, torch.arange(m, device=device))


def compute_delta_infonce_loss(pred: torch.Tensor, true: torch.Tensor,
                               anchor: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """InfoNCE on the displacement instead of the destination.

    Motivation is measured, not aesthetic. Under Extrapolation every in-batch negative
    is a state from the query's own problem, so with ABSOLUTE scoring the negatives sit
    at cosine 0.995-0.998 with sd 0.0009-0.0026. Divided by tau=0.07 that is a logit
    spread of 0.013-0.037 — the softmax is nearly uniform and carries almost no gradient
    about which candidate is correct. The same negatives as *displacements* sit at cosine
    ~0.00 with sd 0.29-0.38, a logit spread of 4.1-5.5, i.e. 148-327x more signal.

    The construction mirrors scoring._prepare(DELTA) exactly, including the detail that
    candidate j is measured against query i's own anchor:

        dp[i]    = normalize(normalize(pred_i)  - normalize(s_i))
        dc[i, j] = normalize(normalize(cand_j) - normalize(s_i))

    so that what the loss optimizes is literally what rank_in_pool/matched_pool_eval
    score. Training on absolute geometry and scoring on delta geometry does not work —
    results/analysis/scoring_rules.json shows eval-time-only DELTA *lowering* matched
    Hit@5 from 0.581 to 0.439 on blocksworld extrapolation.
    """
    p = F.normalize(pred, dim=-1)
    c = F.normalize(true, dim=-1)
    s = F.normalize(anchor, dim=-1)
    dp = F.normalize(p - s, dim=-1)                                  # (B, D)
    dc = F.normalize(c.unsqueeze(0) - s.unsqueeze(1), dim=-1)        # (B, B, D)
    logits = torch.einsum("id,ijd->ij", dp, dc) / tau
    return F.cross_entropy(logits, torch.arange(p.size(0), device=p.device))


def compute_delta_action_loss(model, s, a, sp_proj, anchor, tau: float, m: int,
                              device) -> torch.Tensor:
    """Action-disambiguation term in displacement space; same grid as its absolute twin."""
    sel = torch.randperm(s.size(0), device=device)[:m]
    ss, aa, pp, an = s[sel], a[sel], sp_proj[sel], anchor[sel]
    preds = model(ss.repeat_interleave(m, 0), aa.repeat(m, 1)).view(m, m, -1)
    p = F.normalize(preds, dim=-1)                       # (m, m, D), row i = state i
    sn = F.normalize(an, dim=-1)                         # (m, D)
    cn = F.normalize(pp, dim=-1)                         # (m, D)
    dp = F.normalize(p - sn.unsqueeze(1), dim=-1)        # (m, m, D)
    dc = F.normalize(cn - sn, dim=-1)                    # (m, D) true delta for state i
    logits = (dp * dc.unsqueeze(1)).sum(-1) / tau
    return F.cross_entropy(logits, torch.arange(m, device=device))
