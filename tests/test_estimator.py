"""The scikit-learn style estimator: fit/predict/evaluate on text, protocols, persistence."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from embedplan import EmbedPlan, split_transitions, transitions_from_trajectories
from embedplan.datasets import load_toy_ferry
from embedplan.encoders import HashingEncoder, get_encoder

SMALL = dict(projection_dim=32, hidden_size=64, max_epochs=60, random_state=0)


@pytest.fixture(scope="module")
def data():
    return load_toy_ferry(n_problems=8, n_plans=2, random_state=0)


@pytest.fixture(scope="module")
def interp(data):
    return split_transitions(data.X, data.y, data.groups, protocol="interpolation", random_state=0)


@pytest.fixture(scope="module")
def model(interp):
    X_tr, _, y_tr, _, _, _ = interp
    return EmbedPlan(encoder=HashingEncoder(n_features=256), **SMALL).fit(X_tr, y_tr)


def test_load_toy_ferry_is_consistent(data):
    assert len(data.X) == len(data.y) == len(data.groups) == len(data.frame)
    assert all(isinstance(s, str) and isinstance(a, str) for s, a in data.X)
    f = data.frame
    # along a plan, each next state is the next step's state
    for (_, _), plan in f.groupby(["problem", "plan"]):
        assert (plan["next_state"].iloc[:-1].to_numpy() == plan["state"].iloc[1:].to_numpy()).all()


def test_split_protocols(data):
    _, _, _, _, g_tr, g_te = split_transitions(data.X, data.y, data.groups, protocol="extrapolation")
    assert set(g_tr).isdisjoint(g_te)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = split_transitions(data.X, data.y, data.groups, protocol="interpolation")
    assert len(X_tr) + len(X_te) == len(data.X) and len(y_te) == len(X_te)
    assert set(g_tr) & set(g_te)          # the same problems on both sides
    with pytest.raises(ValueError):
        split_transitions(data.X, data.y, None, protocol="extrapolation")
    with pytest.raises(ValueError):
        split_transitions(data.X, data.y, data.groups, protocol="nope")


def test_fit_predict_and_learns_above_chance(model, interp):
    _, X_te, _, y_te, _, _ = interp
    pred = model.predict(X_te[:5])
    assert len(pred) == 5 and all(p in set(model.candidate_states_) for p in pred)
    top = model.predict_topk(X_te[:5], k=3)
    assert all(len(t) == 3 for t in top) and [t[0] for t in top] == pred
    r = model.evaluate(X_te, y_te)
    assert r["hit@5"] > 2 * r["chance@5"]
    assert 0 < r["mrr"] <= 1 and r["n_queries"] == len(X_te)


def test_interpolation_pools_have_128_candidates(model, interp):
    _, X_te, _, y_te, _, _ = interp
    r = model.evaluate(X_te, y_te, ks=(1, 5))
    assert r["mean_pool_size"] == 128
    assert r["chance@5"] == pytest.approx(5 / 128)


def test_group_distractors_stay_inside_the_group(data):
    X_tr, X_te, y_tr, y_te, g_tr, g_te = split_transitions(data.X, data.y, data.groups, protocol="extrapolation")
    m = EmbedPlan(encoder=HashingEncoder(n_features=256), **dict(SMALL, max_epochs=10)).fit(X_tr, y_tr, groups=g_tr)
    r = m.evaluate(X_te, y_te, groups=g_te)
    states_per_group = pd.DataFrame({"g": g_te, "s": [s for s, _ in X_te], "n": y_te})
    largest = max(len(set(d["s"]) | set(d["n"])) for _, d in states_per_group.groupby("g"))
    assert r["mean_pool_size"] <= largest
    full = m.evaluate(X_te, y_te, groups=g_te, pool_size=None)
    assert full["mean_pool_size"] >= r["mean_pool_size"]
    with pytest.raises(ValueError):
        m.evaluate(X_te, y_te, distractors="group")


def test_ties_count_against_the_truth(interp):
    """A representation that maps every state to one vector must score 0, not 100%."""
    X_tr, X_te, y_tr, y_te, _, _ = interp
    constant = lambda texts: np.ones((len(texts), 8), dtype=np.float32)  # noqa: E731
    m = EmbedPlan(encoder=constant, **dict(SMALL, max_epochs=2)).fit(X_tr, y_tr)
    r = m.evaluate(X_te, y_te, ks=(1, 5))
    assert r["hit@1"] == 0.0 and r["hit@5"] == 0.0


def test_sklearn_clone_and_params(model):
    c = clone(model)
    same = lambda p: {k: v for k, v in p.items() if k != "encoder"}  # noqa: E731
    assert same(c.get_params()) == same(model.get_params())
    assert c.encoder.n_features == model.encoder.n_features      # clone deep-copies the encoder object
    assert not hasattr(c, "model_")
    c.set_params(max_epochs=3)
    assert c.max_epochs == 3
    assert EmbedPlan(**EmbedPlan.paper_params()).lr == 4e-5


def test_same_random_state_same_results(interp):
    X_tr, X_te, y_tr, y_te, _, _ = interp
    runs = [EmbedPlan(encoder="hashing", **dict(SMALL, max_epochs=5)).fit(X_tr, y_tr).evaluate(X_te, y_te)
            for _ in range(2)]
    assert runs[0] == runs[1]


def test_save_and_load_roundtrip(model, interp, tmp_path):
    _, X_te, _, y_te, _, _ = interp
    path = tmp_path / "model.pt"
    model.save(path)
    with pytest.raises(ValueError):
        EmbedPlan.load(path)                   # fit with a custom encoder object: it must be passed
    loaded = EmbedPlan.load(path, encoder=HashingEncoder(n_features=256))
    assert loaded.predict(X_te[:10]) == model.predict(X_te[:10])
    assert loaded.evaluate(X_te, y_te) == model.evaluate(X_te, y_te)


def test_named_encoder_roundtrip_needs_no_encoder(interp, tmp_path):
    X_tr, X_te, y_tr, _, _, _ = interp
    m = EmbedPlan(encoder="hashing", **dict(SMALL, max_epochs=2)).fit(X_tr, y_tr)
    m.save(tmp_path / "m.pt")
    assert EmbedPlan.load(tmp_path / "m.pt").predict(X_te[:3]) == m.predict(X_te[:3])


def test_rollout_and_candidates(model, data):
    frame = data.frame[(data.frame.problem == 0) & (data.frame.plan == 0)]
    out = model.rollout(frame["state"].iloc[0], frame["action"].tolist()[:4])
    assert len(out) == 4 and all(s in set(model.candidate_states_) for s in out)
    own = frame["next_state"].tolist()
    per_query = model.predict([(frame["state"].iloc[0], frame["action"].iloc[0])], candidates=[own])
    assert per_query[0] in own
    n = len(model.candidate_states_)
    model.add_states(["A brand new state that was never seen."])
    assert len(model.candidate_states_) == n + 1


def test_dataframe_input_and_validation(data):
    df = data.frame[["state", "action", "next_state"]].head(40)
    m = EmbedPlan(encoder="hashing", **dict(SMALL, max_epochs=2)).fit(df)
    assert len(m.predict(df)) == 40
    with pytest.raises(ValueError):
        EmbedPlan().fit(data.X[:5], data.y[:4])
    with pytest.raises(TypeError):
        EmbedPlan().fit([("state", 3)], ["next"])
    with pytest.raises(ValueError):
        EmbedPlan().fit(data.X[:5])                     # no next states
    with pytest.raises(ValueError):
        EmbedPlan().fit(data.X[:5], data.y[:5], groups=[0, 1])


def test_early_stopping_selects_on_validation(data):
    m = EmbedPlan(encoder="hashing", early_stopping=True, eval_every=2, patience=4,
                  **dict(SMALL, max_epochs=30)).fit(data.X, data.y, groups=data.groups)
    evals = [h for h in m.history_ if "val_mrr" in h]
    assert evals and m.best_epoch_ in [h["epoch"] for h in evals]
    assert m.best_validation_score_ == max(h["val_mrr"] for h in evals)


def test_encoder_callables_are_checked():
    enc = get_encoder(lambda t: torch.ones(len(t), 4))
    assert enc(["a", "b"]).shape == (2, 4) and enc(["a"]).dtype == np.float32
    bad = get_encoder(lambda t: np.ones((1, 4)))
    with pytest.raises(ValueError):
        bad(["a", "b"])
    with pytest.raises(TypeError):
        get_encoder(3)


def test_transitions_from_trajectories():
    trajs = [(["s0", "s1", "s2"], ["a", "b"]), (["t0", "t1"], ["c"])]
    X, y, g = transitions_from_trajectories(trajs, groups=["p1", "p2"])
    assert X == [("s0", "a"), ("s1", "b"), ("t0", "c")] and y == ["s1", "s2", "t1"]
    assert g.tolist() == ["p1", "p1", "p2"]
    with pytest.raises(ValueError):
        transitions_from_trajectories([(["s0"], ["a"])])
