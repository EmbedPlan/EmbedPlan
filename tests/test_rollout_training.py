"""The shared training loop and multi-step rollout, on tiny CPU tasks."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from embedplan.data import build_trajectories
from embedplan.losses import compute_delta_infonce_loss, compute_infonce_loss
from embedplan.models import ProjectedTransitionModel, build_model
from embedplan.rollout import REGIMES, rollout
from embedplan.scoring import project_pool
from embedplan.training import _make_batches, train_transition


def ring(n=24, n_actions=4, dim=16, seed=0):
    """Action k moves state i to state (i + k + 1) mod n: learnable, and deterministic."""
    g = torch.Generator().manual_seed(seed)
    S, A = torch.randn(n, dim, generator=g), torch.randn(n_actions, 8, generator=g)
    s, a = np.repeat(np.arange(n), n_actions), np.tile(np.arange(n_actions), n)
    tri = pd.DataFrame({"s_emb_idx": s, "a_idx": a, "sp_emb_idx": (s + a + 1) % n, "problem_idx": s % 3,
                        "plan_id_idx": s // 6, "goal_val": n - s})
    return S, A, tri


def train_args(**over):
    args = dict(projection_dim=16, projection_layers=2, hidden_size=32, n_layers=2, dropout=0.0, transition="mlp",
                split="random", lr=1e-2, seed=0, batch_size=16, epochs=2, loss_space="absolute", tau=0.1,
                action_weight=0.0)
    return SimpleNamespace(**{**args, **over})


@torch.no_grad()
def objective(model, S, A, tri, loss_space, tau=0.1):
    """The training objective over the whole table in one batch."""
    s, a, sp = (torch.tensor(tri[c].tolist()) for c in ("s_emb_idx", "a_idx", "sp_emb_idx"))
    model.eval()
    pred, sp_proj = model(S[s], A[a]), model.state_projection_head(S[sp])
    if loss_space == "delta":
        return compute_delta_infonce_loss(pred, sp_proj, model.state_projection_head(S[s]), tau).item()
    return compute_infonce_loss(pred, sp_proj, tau).item()


# ----------------------------------------------------------------------------- training

@pytest.mark.parametrize("loss_space", ["absolute", "delta"])
@pytest.mark.parametrize("split", ["random", "problem_grouped"])
def test_train_transition_two_epochs_reduces_the_loss(loss_space, split):
    torch.manual_seed(0)
    S, A, tri = ring()
    args = train_args(loss_space=loss_space, split=split, action_weight=0.5)
    model = build_model(16, 8, args, "cpu")
    before = objective(model, S, A, tri, loss_space)
    out = train_transition(model, S, A, tri, list(range(len(tri))), args, "cpu", verbose=False)
    assert out is model
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert objective(model, S, A, tri, loss_space) < before


def test_train_transition_only_sees_the_training_rows():
    """Rows outside train_idx must not influence the model: training on a subset gives
    the same weights whatever the held-out rows contain."""
    S, A, tri = ring()
    train_idx = list(range(0, len(tri), 2))
    poisoned = tri.copy()
    held_out = poisoned.index.difference(train_idx)
    poisoned.loc[held_out, "sp_emb_idx"] = 0
    weights = []
    for table in (tri, poisoned):
        torch.manual_seed(0)
        model = build_model(16, 8, train_args(), "cpu")
        train_transition(model, S, A, table, train_idx, train_args(), "cpu", verbose=False)
        weights.append(torch.cat([p.flatten() for p in model.parameters()]))
    assert torch.equal(*weights)


def test_train_transition_logs_only_when_verbose(capsys):
    S, A, tri = ring()
    model = build_model(16, 8, train_args(), "cpu")
    train_transition(model, S, A, tri, list(range(len(tri))), train_args(), "cpu", verbose=False)
    assert capsys.readouterr().out == ""
    train_transition(model, S, A, tri, list(range(len(tri))), train_args(), "cpu", verbose=True)
    assert "epoch    1" in capsys.readouterr().out


def test_make_batches_within_groups_cover_every_row_once():
    rng = np.random.default_rng(0)
    groups = [np.arange(0, 7), np.arange(7, 10), np.arange(10, 22)]
    batches = _make_batches(22, 4, rng, groups)
    assert sorted(np.concatenate(batches).tolist()) == list(range(22))
    assert all(len(b) <= 4 for b in batches)
    assert all(any(set(b) <= set(g) for g in groups) for b in batches)
    assert sorted(np.concatenate(_make_batches(22, 4, rng)).tolist()) == list(range(22))


# ----------------------------------------------------------------------------- rollout

class AddTransition(nn.Module):
    def forward(self, s, a):
        return s + a


def oracle():
    """Identity projections, pred = s + a: exact when A[a] = S[s'] - S[s]."""
    return ProjectedTransitionModel(nn.Identity(), nn.Identity(), AddTransition())


def chain_world(lengths=(4, 3, 3, 2, 1), dim=16, seed=0):
    """One fresh chain of states per trajectory and one action per step, A[k] = S[s'_k] - S[s_k]."""
    S = torch.randn(sum(lengths) + len(lengths) + 5, dim, generator=torch.Generator().manual_seed(seed))
    trajs, A, nxt = [], [], 0
    for n in lengths:
        states = np.arange(nxt, nxt + n + 1)
        nxt += n + 1
        acts = np.arange(len(A), len(A) + n)
        A += [S[states[t + 1]] - S[states[t]] for t in range(n)]
        trajs.append((states[:-1], acts, states[1:]))
    return S, torch.stack(A), trajs


def test_rollout_oracle_is_perfect_in_every_regime():
    S, A, trajs = chain_world()
    model = oracle()
    out = rollout(model, S, A, trajs, project_pool(model, S, normalize=True), "cpu", prefix_curve=True)
    assert set(out) == {*REGIMES, "_meta"}
    for mode in REGIMES:
        assert out[mode]["step_hit@1"] == out[mode]["exact_hit@1"] == 1.0, mode
        assert out[mode]["prefix_success@1"] == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        assert {t: d["n"] for t, d in out[mode]["by_depth"].items()} == {1: 5, 2: 4, 3: 3, 4: 1}
    assert out["_meta"] == {"n_trajectories": 5, "max_len": 4, "mean_len": 2.6, "pool_size": len(S)}


def test_rollout_invariants_for_a_random_model():
    torch.manual_seed(0)
    S, A, tri = ring()
    tri = tri[tri["a_idx"] == 0].reset_index(drop=True)          # plans are runs of +1 moves
    trajs = build_trajectories(tri, range(len(tri)), max_trajs=10, seed=0)
    model = build_model(16, 8, train_args(), "cpu")
    out = rollout(model, S, A, trajs, project_pool(model, S, normalize=True), "cpu", topk=(1, 5))
    for mode in REGIMES:
        r = out[mode]
        assert 0.0 <= r["step_hit@1"] <= r["step_hit@5"] <= 1.0
        assert r["exact_hit@1"] <= r["exact_hit@5"]
        depths = r["by_depth"]
        assert sorted(depths) == list(range(1, out["_meta"]["max_len"] + 1))
        assert depths[1]["n"] == len(trajs)
        assert all(depths[t]["hit@1"] <= depths[t]["hit@5"] for t in depths)
        assert "prefix_success@1" not in r


def test_rollout_subset_pool_maps_back_to_global_ids():
    S, A, trajs = chain_world()
    model = oracle()
    pool_idx = np.arange(0, len(S) - 3)          # drops three unused states
    pool = project_pool(model, S[pool_idx], normalize=True)
    out = rollout(model, S, A, trajs, pool, "cpu", pool_idx=pool_idx)
    assert all(out[mode]["step_hit@1"] == 1.0 for mode in REGIMES)
    assert out["_meta"]["pool_size"] == len(pool_idx)

    last_target = int(trajs[0][2][-1])            # only trajectory 0 reaches depth 4
    keep = pool_idx[pool_idx != last_target]
    out = rollout(model, S, A, trajs, project_pool(model, S[keep], normalize=True), "cpu", pool_idx=keep)
    for mode in REGIMES:
        assert out[mode]["by_depth"][4]["hit@1"] == 0.0
        assert out[mode]["by_depth"][3]["hit@1"] == 1.0


def test_rollout_closed_loop_feeds_back_the_snapped_state():
    """Step 1 is sent to a wrong state X on purpose, and X + a_2 is exactly a decoy state Y.
    Teacher forcing recovers at step 2; closed loop snaps to X and then lands on Y."""
    S, A, trajs = chain_world(lengths=(3,))
    s0, x = trajs[0][0][0], len(S) - 1
    A = A.clone()
    A[0] = S[x] - S[s0]
    S = S.clone()
    S[len(S) - 2] = S[x] + A[1]                  # the decoy Y
    model = oracle()
    out = rollout(model, S, A, trajs, project_pool(model, S, normalize=True), "cpu")
    tf, cl = out["teacher_forced"]["by_depth"], out["closed_loop"]["by_depth"]
    assert [tf[t]["hit@1"] for t in (1, 2, 3)] == [0.0, 1.0, 1.0]
    assert [cl[t]["hit@1"] for t in (1, 2)] == [0.0, 0.0]
