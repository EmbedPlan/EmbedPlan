"""Evaluators behind the paper's tables, on hand-built worlds with known answers.

Two reference models recur:
  oracle()     identity projections and pred = s + a. Giving transition k its own action
               vector A[k] = S[s'_k] - S[s_k] makes every prediction exact.
  collapsed()  maps every state and prediction to the same vector, so every candidate
               ties with the truth; worst-case tie handling must then score it as a miss.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from embedplan.evaluation import (
    _auroc,
    _compute_ranks,
    action_disambiguation,
    eval_action_disambiguation,
    evaluate_hit_across_states,
    matched_pool_eval,
    open_set_abstention,
    pool_sweep,
)
from embedplan.models import ProjectedTransitionModel, build_model
from embedplan.paper_protocol import paper_hit
from embedplan.scoring import ABSOLUTE, DELTA
from embedplan.utils import EvalConfig



class AddTransition(nn.Module):
    def forward(self, s, a):
        return s + a


class Constant(nn.Module):
    def forward(self, x, *_):
        return torch.ones(x.shape[0], 4)


def oracle():
    return ProjectedTransitionModel(nn.Identity(), nn.Identity(), AddTransition())


def collapsed():
    return ProjectedTransitionModel(Constant(), Constant(), Constant())


def exact_rows(n, dim=16, seed=0):
    """Distinct rows with four +-1 entries. Every cosine between them is a sum of +-1/4,
    exact in float32, so a tie is a tie on any BLAS."""
    rng, seen, rows = np.random.default_rng(seed), set(), []
    while len(rows) < n:
        v = np.zeros(dim, dtype=np.float32)
        v[rng.choice(dim, 4, replace=False)] = rng.choice([-1.0, 1.0], 4)
        if v.tobytes() not in seen:
            seen.add(v.tobytes())
            rows.append(v)
    return torch.from_numpy(np.stack(rows))


def world(n=40, dim=16, n_problems=4, seed=0, exact=True):
    """n transitions over 2n states: query states 0..n-1, next states n..2n-1, one action
    each. exact=False draws Gaussian rows instead, i.e. realistic float rounding."""
    if exact:
        S = exact_rows(2 * n, dim, seed)
    else:
        S = torch.randn(2 * n, dim, generator=torch.Generator().manual_seed(seed))
    s, sp = np.arange(n), np.arange(n, 2 * n)
    A = S[sp] - S[s]
    tri = pd.DataFrame({"s_emb_idx": s, "a_idx": np.arange(n), "sp_emb_idx": sp,
                        "problem_idx": s % n_problems, "plan_id_idx": s})
    return S, A, tri


# ----------------------------------------------------------------------------- loader-based evaluators

class Triplets(Dataset):
    def __init__(self, s, a, sp):
        self.s, self.a, self.sp = s, a, sp

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return {"s_emb": self.s[i], "a_emb": self.a[i], "sp_emb": self.sp[i]}


def loader(s, a, sp, batch_size):
    return DataLoader(Triplets(s, a, sp), batch_size=batch_size, shuffle=False)


def oracle_loader(n=12, dim=16, batch_size=4, seed=0):
    rows = exact_rows(2 * n, dim, seed)
    s, sp = rows[:n], rows[n:]
    return s, sp - s, sp, batch_size


def small_args(transition="mlp"):
    return SimpleNamespace(projection_dim=16, projection_layers=2, hidden_size=32, n_layers=2,
                           dropout=0.0, transition=transition)


def test_hit_across_states_perfect_predictor():
    s, a, sp, bs = oracle_loader()
    out = evaluate_hit_across_states(oracle(), loader(s, a, sp, bs), "cpu", EvalConfig())
    assert out == {"hit@1": 1.0, "hit@5": 1.0, "hit@10": 1.0}


def test_hit_across_states_anchored_model_is_perfect_on_no_op_transitions():
    """build_model(transition='anchored') is exactly the identity at init, so it predicts s' = s."""
    torch.manual_seed(0)
    model = build_model(state_dim=12, action_dim=6, args=small_args("anchored"), device="cpu")
    s = torch.randn(16, 12)
    out = evaluate_hit_across_states(model, loader(s, torch.randn(16, 6), s.clone(), 8), "cpu", EvalConfig())
    assert out["hit@1"] == 1.0


def test_hit_across_states_each_batch_is_its_own_pool():
    """Rows 0 and 2 share a next state. They only tie (and both lose Hit@1 under the
    worst-case convention) when they land in the same validation batch."""
    s, a, sp, _ = oracle_loader(n=4)
    sp[2] = sp[0]
    a = sp - s
    same_batch = evaluate_hit_across_states(oracle(), loader(s, a, sp, 4), "cpu", EvalConfig())
    split = evaluate_hit_across_states(oracle(), loader(s, a, sp, 2), "cpu", EvalConfig())
    assert same_batch["hit@1"] == 0.5
    assert split["hit@1"] == 1.0


def test_hit_across_states_ties_count_against_the_truth():
    s = torch.randn(8, 12)
    out = evaluate_hit_across_states(collapsed(), loader(s, torch.randn(8, 6), s, 8), "cpu", EvalConfig())
    assert out == {"hit@1": 0.0, "hit@5": 0.0, "hit@10": 1.0}   # every rank is 8, the batch size


def test_hit_across_states_scores_raw_space_models_against_raw_next_states():
    s, a, sp, bs = oracle_loader()
    out = evaluate_hit_across_states(AddTransition(), loader(s, a, sp, bs), "cpu", EvalConfig())
    assert out["hit@1"] == 1.0


def test_hit_across_states_random_model_is_well_formed():
    torch.manual_seed(0)
    model = build_model(state_dim=12, action_dim=6, args=small_args(), device="cpu")
    s, sp = torch.randn(20, 12), torch.randn(20, 12)
    data = loader(s, torch.randn(20, 6), sp, 4)
    out = evaluate_hit_across_states(model, data, "cpu", EvalConfig())
    assert set(out) == {"hit@1", "hit@5", "hit@10"}
    assert 0.0 <= out["hit@1"] <= out["hit@5"] <= out["hit@10"]
    assert out["hit@5"] == 1.0          # k >= batch size: the truth is always inside the pool
    acc = eval_action_disambiguation(model, data, "cpu", topk=(1, 2, 4))
    assert 0.0 <= acc["acc_action@1"] <= acc["acc_action@2"] <= acc["acc_action@4"] == 1.0


def test_action_disambiguation_loader_perfect_predictor():
    s, a, sp, bs = oracle_loader()
    out = eval_action_disambiguation(oracle(), loader(s, a, sp, bs), "cpu")
    assert out == {"acc_action@1": 1.0, "acc_action@5": 1.0, "acc_action@10": 1.0}


def test_action_disambiguation_loader_action_blind_model_scores_zero():
    """With every action vector zero the prediction ignores the action, so all B actions
    tie for every state; worst-case ties put the true action last."""
    s, a, sp, _ = oracle_loader(n=8)
    out = eval_action_disambiguation(oracle(), loader(s, torch.zeros_like(a), sp, 8), "cpu", topk=(1, 5, 8))
    assert out == {"acc_action@1": 0.0, "acc_action@5": 0.0, "acc_action@8": 1.0}


def test_compute_ranks_tie_conventions():
    scores = torch.tensor([[1.0, 1.0, 0.0],
                           [0.5, 0.9, 0.9],
                           [0.0, 0.0, 1.0]])
    assert _compute_ranks(scores, "worst").tolist() == [2, 2, 1]
    assert _compute_ranks(scores, "best").tolist() == [1, 1, 1]


def test_compute_ranks_nan_score_is_the_worst_rank():
    scores = torch.tensor([[float("nan"), 0.1, 0.2],
                           [0.2, 0.3, 0.1],
                           [0.0, 0.0, 0.5]])
    assert _compute_ranks(scores).tolist() == [3, 1, 1]


def test_compute_ranks_random_matches_worst_without_ties():
    torch.manual_seed(0)
    scores = torch.randn(6, 6)
    assert torch.equal(_compute_ranks(scores, "random"), _compute_ranks(scores, "worst"))


# ----------------------------------------------------------------------------- pool sweep

def test_pool_sweep_perfect_predictor_absolute():
    S, A, tri = world()
    out = pool_sweep(oracle(), S, A, tri, list(range(len(tri))), S, "cpu", sizes=(4, 16, -1, 10_000), seed=0)
    assert {k: v["hit@1"] for k, v in out.items()} == {"4": 1.0, "16": 1.0, "full": 1.0, "10000": 1.0}
    assert {k: v["pool"] for k, v in out.items()} == {"4": 4, "16": 16, "full": 80, "10000": 80}


def test_pool_sweep_delta_perfect_predictor_with_own_state_excluded():
    S, A, tri = world(n=500, exact=False)
    out = pool_sweep(oracle(), S, A, tri, list(range(len(tri))), S, "cpu", sizes=(8, -1), seed=0,
                     mode=DELTA, exclude_self=True)
    assert out["8"]["hit@1"] == out["full"]["hit@1"] == 1.0


def test_pool_sweep_delta_full_pool_perfect_predictor():
    S, A, tri = world(n=500, exact=False)
    out = pool_sweep(oracle(), S, A, tri, list(range(len(tri))), S, "cpu", sizes=(-1,), seed=0, mode=DELTA)
    assert out["full"]["hit@1"] == 1.0


def test_pool_sweep_tie_convention_on_a_duplicated_next_state():
    S, A, tri = world()
    S2 = torch.cat([S, S[tri["sp_emb_idx"].tolist()]])      # every true next state has an exact twin
    args = (oracle(), S2, A, tri, list(range(len(tri))), S2, "cpu")
    best = pool_sweep(*args, sizes=(-1,), seed=0, best_case_ties=True)["full"]
    worst = pool_sweep(*args, sizes=(-1,), seed=0, best_case_ties=False)["full"]
    assert (best["hit@1"], worst["hit@1"], worst["hit@5"]) == (1.0, 0.0, 1.0)


# ----------------------------------------------------------------------------- matched pools

@pytest.mark.parametrize("mode", [ABSOLUTE, DELTA])
def test_matched_pool_perfect_predictor(mode):
    S, A, tri = world(n=40, n_problems=4)
    out = matched_pool_eval(oracle(), S, A, tri, SimpleNamespace(triplets=tri), list(range(40)), "cpu",
                            batch_size=5, mode=mode)
    assert out["hit@1"] == 1.0
    assert out["n_queries"] == 40 and out["mean_pool"] == 5.0


def test_matched_pool_skips_singleton_pools_and_counts_same_problem_ties():
    S, A, tri = world(n=12, n_problems=3)          # problem = row % 3
    tri.loc[3, "sp_emb_idx"] = tri.loc[0, "sp_emb_idx"]        # rows 0 and 3 are both problem 0
    A[3] = S[tri.loc[3, "sp_emb_idx"]] - S[tri.loc[3, "s_emb_idx"]]
    tri.loc[11, "problem_idx"] = 99                           # a problem with a single row
    args = (oracle(), S, A, tri, SimpleNamespace(triplets=tri), list(range(12)), "cpu")
    best = matched_pool_eval(*args, batch_size=8, best_case_ties=True)
    worst = matched_pool_eval(*args, batch_size=8, best_case_ties=False)
    assert best["n_queries"] == worst["n_queries"] == 11
    assert best["hit@1"] == 1.0
    assert worst["hit@1"] == pytest.approx(9 / 11)


# ----------------------------------------------------------------------------- open-set abstention

@pytest.mark.parametrize("mode", [ABSOLUTE, DELTA])
def test_open_set_abstention_output_contract(mode):
    torch.manual_seed(0)
    S, A, tri = world(n=30)
    model = build_model(state_dim=16, action_dim=16, args=small_args(), device="cpu")
    with torch.no_grad():
        pool = model.state_projection_head(S)
    out = open_set_abstention(model, S, A, tri, list(range(30)), pool, "cpu", seed=0, max_q=20, mode=mode)
    assert out["n_queries"] == 20 and out["pool_size"] == 60
    assert out["mean_score_answerable"] >= out["mean_score_unanswerable"]   # a max over a superset
    assert out["mean_margin"] >= 0.0 and 0.0 <= out["frac_margin_positive"] <= 1.0
    assert 0.0 <= out["auroc"] <= 1.0


@pytest.mark.parametrize("mode", [ABSOLUTE, DELTA])
def test_open_set_abstention_perfect_predictor_separates(mode):
    S, A, tri = world(n=500, exact=False)
    out = open_set_abstention(oracle(), S, A, tri, list(range(500)), S, "cpu", seed=0, mode=mode)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["frac_margin_positive"] == 1.0


def test_open_set_abstention_identity_has_no_abstention_signal():
    """pred = s finds s itself in the pool, with or without the true next state."""
    S, A, tri = world()
    out = open_set_abstention(oracle(), S, torch.zeros_like(A), tri, list(range(40)), S, "cpu", seed=0)
    assert out["frac_margin_positive"] == 0.0 and out["mean_margin"] == 0.0


def test_auroc():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert _auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(1.0)
    assert _auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(0.0)
    assert np.isnan(_auroc(torch.tensor([0.3, 0.4]), torch.tensor([1.0, 1.0])))


# ----------------------------------------------------------------------------- action disambiguation (indexed)

def test_action_disambiguation_perfect_and_action_blind():
    S, A, tri = world()
    idx = list(range(40))
    assert action_disambiguation(oracle(), S, A, tri, idx, "cpu", batch=8) == \
        {"acc_action@1": 1.0, "acc_action@5": 1.0, "acc_action@10": 1.0}
    blind = action_disambiguation(oracle(), S, torch.zeros_like(A), tri, idx, "cpu", batch=8)
    assert blind == {"acc_action@1": 0.0, "acc_action@5": 0.0, "acc_action@10": 1.0}


# ----------------------------------------------------------------------------- the published protocol

@pytest.mark.parametrize("mode", [ABSOLUTE, DELTA])
def test_paper_hit_problem_grouped_perfect_predictor(mode):
    S, A, tri = world(n=40, n_problems=4)
    out = paper_hit(oracle(), S, A, tri, SimpleNamespace(triplets=tri), list(range(40)), "cpu",
                    split="problem_grouped", seed=0, pool_size=8, mode=mode)
    assert (out["hit@1"], out["n_queries"]) == (1.0, 40)
    assert out["mean_pool"] == pytest.approx(np.mean([8, 2] * 4))


def test_paper_hit_problem_grouped_pools_never_mix_problems():
    """Rows 0 (problem 0) and 1 (problem 1) share a next state: no tie, since they never share a pool."""
    S, A, tri = world(n=40, n_problems=4)
    tri.loc[1, "sp_emb_idx"] = tri.loc[0, "sp_emb_idx"]
    A[1] = S[tri.loc[1, "sp_emb_idx"]] - S[tri.loc[1, "s_emb_idx"]]
    out = paper_hit(oracle(), S, A, tri, SimpleNamespace(triplets=tri), list(range(40)), "cpu",
                    split="problem_grouped", seed=0, pool_size=128)
    assert out["hit@1"] == 1.0


@pytest.mark.parametrize("split", ["random", "problem_grouped"])
def test_paper_hit_counts_ties_against_the_truth(split):
    S, A, tri = world(n=48, n_problems=3)
    out = paper_hit(collapsed(), S, A, tri, SimpleNamespace(triplets=tri), list(range(48)), "cpu",
                    split=split, seed=0, pool_size=16)
    assert (out["hit@1"], out["hit@5"], out["hit@10"]) == (0.0, 0.0, 0.0)


def test_paper_hit_random_output_contract():
    torch.manual_seed(0)
    S, A, tri = world(n=60)
    model = build_model(state_dim=16, action_dim=16, args=small_args(), device="cpu")
    out = paper_hit(model, S, A, tri, None, list(range(60)), "cpu", split="random", seed=0,
                    pool_size=32, max_q=50)
    assert out["n_queries"] == 50 and out["mean_pool"] == 32.0
    assert 0.0 <= out["hit@1"] <= out["hit@5"] <= out["hit@10"] <= 1.0


@pytest.mark.xfail(strict=True, reason="Kept to reproduce the paper: paper_hit(split='random') samples distractors with replacement from "
                                       "all states, including the true next state, and the worst-case tie then "
                                       "counts that copy against the truth")
def test_paper_hit_random_perfect_predictor():
    """With 15 distractors drawn from 80 states, 5 of 40 queries draw their own next
    state and score Hit@1 = 0 although every prediction is exact."""
    S, A, tri = world(n=40)
    out = paper_hit(oracle(), S, A, tri, None, list(range(40)), "cpu", split="random", seed=0, pool_size=16)
    assert out["hit@1"] == 1.0
