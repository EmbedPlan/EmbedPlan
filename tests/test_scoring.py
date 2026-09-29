"""Retrieval scoring on hand-built examples: ranks, tie conventions, Hit@k, DELTA."""

import pytest
import torch
import torch.nn.functional as F

from embedplan.scoring import (
    ABSOLUTE,
    DELTA,
    _delta_scores_shared_pool,
    hit_at_k,
    rank_in_candidates,
    rank_in_pool,
)


def test_rank_in_pool_finds_exact_match_first():
    torch.manual_seed(0)
    pool = torch.randn(50, 16)
    target = torch.tensor([3, 17, 42])
    ranks = rank_in_pool(pool[target].clone(), pool, target)
    assert ranks.tolist() == [1, 1, 1]


def test_rank_in_pool_counts_better_candidates():
    pool = torch.eye(4)
    pred = torch.tensor([[1.0, 0.9, 0.0, 0.0]])        # closest to row 0, then row 1
    assert rank_in_pool(pred, pool, torch.tensor([0])).item() == 1
    assert rank_in_pool(pred, pool, torch.tensor([1])).item() == 2


def test_tie_conventions_differ_only_on_exact_ties():
    pool = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # rows 0 and 1 tie exactly
    pred = torch.tensor([[1.0, 0.0]])
    best = rank_in_pool(pred, pool, torch.tensor([0]), best_case_ties=True)
    worst = rank_in_pool(pred, pool, torch.tensor([0]), best_case_ties=False)
    assert best.item() == 1
    assert worst.item() == 2   # the published protocol counts a tie against the ground truth


def test_exclude_idx_removes_the_query_state():
    pool = torch.eye(3)
    pred = torch.tensor([[1.0, 0.5, 0.0]])             # identity-like: closest to its own state 0
    target = torch.tensor([1])
    assert rank_in_pool(pred, pool, target).item() == 2
    assert rank_in_pool(pred, pool, target, exclude_idx=torch.tensor([0])).item() == 1


def test_rank_in_candidates_true_state_at_position_zero():
    torch.manual_seed(0)
    cand = torch.randn(5, 8, 12)
    pred = cand[:, 0].clone()
    assert rank_in_candidates(pred, cand).tolist() == [1] * 5


def test_hit_at_k():
    ranks = torch.tensor([1, 2, 5, 6, 10, 11])
    h = hit_at_k(ranks, topk=(1, 5, 10))
    assert h == pytest.approx({"hit@1": 1 / 6, "hit@5": 3 / 6, "hit@10": 5 / 6})


def test_delta_shared_pool_matches_explicit_computation():
    torch.manual_seed(0)
    pred, pool, anchor = torch.randn(4, 8), torch.randn(10, 8), torch.randn(4, 8)
    fast = _delta_scores_shared_pool(pred, pool, anchor)
    p, c, s = (F.normalize(x, dim=-1) for x in (pred, pool, anchor))
    dp = F.normalize(p - s, dim=-1)                                   # (Q, D)
    dc = F.normalize(c.unsqueeze(0) - s.unsqueeze(1), dim=-1)         # (Q, P, D)
    explicit = torch.einsum("qd,qpd->qp", dp, dc)
    assert torch.allclose(fast, explicit, atol=1e-5)


def test_delta_and_absolute_agree_on_a_perfect_prediction():
    torch.manual_seed(0)
    pool, anchor = torch.randn(30, 8), torch.randn(2, 8)
    target = torch.tensor([4, 9])
    for mode in (ABSOLUTE, DELTA):
        ranks = rank_in_pool(pool[target].clone(), pool, target, anchor=anchor, mode=mode)
        assert ranks.tolist() == [1, 1], mode


def test_delta_rejects_a_shared_pool_in_prepare():
    from embedplan.scoring import _prepare
    with pytest.raises(ValueError):
        _prepare(torch.randn(3, 8), torch.randn(10, 8), torch.randn(3, 8), DELTA)


def test_delta_rank_in_pool_matches_per_query_candidates():
    torch.manual_seed(0)
    pred, pool, anchor = torch.randn(4, 8), torch.randn(10, 8), torch.randn(4, 8)
    target = torch.tensor([0, 3, 5, 9])
    ranks_pool = rank_in_pool(pred, pool, target, anchor=anchor, mode=DELTA)
    order = torch.stack([torch.cat([t.view(1), torch.tensor([j for j in range(10) if j != t])]) for t in target])
    ranks_cand = rank_in_candidates(pred, pool[order], anchor=anchor, mode=DELTA)
    assert ranks_pool.tolist() == ranks_cand.tolist()
