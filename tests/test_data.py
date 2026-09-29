"""Splits, problem-grouped batching, plan parsing and trajectory reconstruction.

Most tests use a synthetic triplet table: plans are chains of fresh state ids, so
sp[t] == s[t+1] inside a plan by construction. Problem ids are deliberately not
0..n-1 so that positions and labels cannot be confused. The last block loads the
toy domain from tools/make_toy_domain.py through the real loader.
"""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from embedplan.data import (
    ProblemGroupedBatchSampler,
    build_trajectories,
    grouped_split_by_plan,
    grouped_split_by_problem,
    leave_one_problem_out,
    make_split,
    parse_plan,
)
from embedplan.prompts import create_original_prompt


def make_tri(n_problems=5, plans_per_problem=3, steps=5, seed=0):
    """Shuffled rows; problem ids 10, 20, ...; plan ids unique across problems."""
    rng = np.random.default_rng(seed)
    rows, next_state, plan_id = [], 0, 0
    for p in range(n_problems):
        for _ in range(plans_per_problem):
            states = range(next_state, next_state + steps + 1)
            next_state += steps + 1
            for t in range(steps):
                rows.append({"s_emb_idx": states[t], "sp_emb_idx": states[t + 1], "a_idx": int(rng.integers(0, 7)),
                             "problem_idx": 10 * (p + 1), "plan_id_idx": plan_id, "goal_val": steps - t})
            plan_id += 1
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def as_ds(tri):
    return SimpleNamespace(triplets=tri)


def labels(tri, idx, col):
    return set(tri[col].to_numpy()[list(idx)].tolist())


def assert_partition(train_idx, valid_idx, n):
    assert set(train_idx).isdisjoint(valid_idx)
    assert sorted(list(train_idx) + list(valid_idx)) == list(range(n))


# ----------------------------------------------------------------------------- splits

def test_grouped_split_by_problem_holds_out_whole_problems():
    tri = make_tri(n_problems=5)
    tr, va, (train_probs, valid_probs) = grouped_split_by_problem(as_ds(tri), train_frac=0.6, seed=1,
                                                                   return_problem_sets=True)
    assert_partition(tr, va, len(tri))
    assert labels(tri, tr, "problem_idx") == train_probs
    assert labels(tri, va, "problem_idx") == valid_probs
    assert train_probs.isdisjoint(valid_probs)
    assert (len(train_probs), len(valid_probs)) == (3, 2)


def test_grouped_split_by_problem_is_seeded():
    ds = as_ds(make_tri(n_problems=6))
    assert grouped_split_by_problem(ds, seed=3) == grouped_split_by_problem(ds, seed=3)
    splits = {tuple(grouped_split_by_problem(ds, train_frac=0.5, seed=s)[1]) for s in range(6)}
    assert len(splits) > 1


@pytest.mark.parametrize("frac, n_train", [(0.01, 1), (0.99, 3)])
def test_grouped_split_by_problem_keeps_both_sides_non_empty(frac, n_train):
    tri = make_tri(n_problems=4)
    tr, va, (train_probs, valid_probs) = grouped_split_by_problem(as_ds(tri), train_frac=frac,
                                                                   return_problem_sets=True)
    assert len(train_probs) == n_train and len(valid_probs) == 4 - n_train
    assert tr and va


@pytest.mark.xfail(strict=True, reason="Kept to reproduce the paper: train_frac=1.0 still holds one problem out (n_train clamped to "
                                       "n-1), so train.py's cross-domain mode silently drops a training problem")
def test_grouped_split_by_problem_train_frac_one_keeps_every_problem():
    tri = make_tri(n_problems=4)
    tr, va, _ = grouped_split_by_problem(as_ds(tri), train_frac=1.0)
    assert len(tr) == len(tri) and va == []


def test_grouped_split_by_plan_never_splits_a_plan():
    tri = make_tri(n_problems=4, plans_per_problem=3)
    tr, va = grouped_split_by_plan(as_ds(tri), train_frac=0.5, seed=2)
    assert_partition(tr, va, len(tri))
    assert labels(tri, tr, "plan_id_idx").isdisjoint(labels(tri, va, "plan_id_idx"))
    assert len(labels(tri, tr, "plan_id_idx")) == 6


def test_leave_one_problem_out_tests_on_exactly_that_problem():
    tri = make_tri(n_problems=5)
    tr, va, chosen = leave_one_problem_out(as_ds(tri), test_problem=30)
    assert labels(tri, va, "problem_idx") == {30}
    assert len(va) == int((tri["problem_idx"] == 30).sum())
    assert chosen == [10, 20, 40, 50]
    assert labels(tri, tr, "problem_idx") == set(chosen)
    assert_partition(tr, va, len(tri))


def test_leave_one_problem_out_training_sets_are_nested_in_n():
    """The learning curve varies only how many problems are trained on, so for one
    seed the smaller training sets must be prefixes of the larger ones."""
    ds = as_ds(make_tri(n_problems=6))
    sets = [set(leave_one_problem_out(ds, test_problem=20, n_train_problems=n, seed=4)[2]) for n in (1, 2, 3, 5)]
    assert [len(s) for s in sets] == [1, 2, 3, 5]
    assert sets[0] < sets[1] < sets[2] < sets[3]
    assert all(20 not in s for s in sets)


def test_leave_one_problem_out_rejects_an_unknown_problem():
    with pytest.raises(ValueError):
        leave_one_problem_out(as_ds(make_tri()), test_problem=7)


def test_make_split_random_is_a_seeded_permutation():
    tri = make_tri()
    tr, va = make_split(as_ds(tri), tri, "random", seed=0, train_frac=0.8)
    assert_partition(tr, va, len(tri))
    assert len(tr) == int(0.8 * len(tri))
    assert make_split(as_ds(tri), tri, "random", seed=0) == (tr, va)
    assert make_split(as_ds(tri), tri, "random", seed=1) != (tr, va)


def test_make_split_problem_grouped_is_the_extrapolation_split():
    tri = make_tri(n_problems=5)
    tr, va = make_split(as_ds(tri), tri, "problem_grouped", seed=0)
    assert_partition(tr, va, len(tri))
    assert labels(tri, tr, "problem_idx").isdisjoint(labels(tri, va, "problem_idx"))
    assert (tr, va) == grouped_split_by_problem(as_ds(tri), train_frac=0.8, seed=0)[:2]


# ----------------------------------------------------------------------------- batching

def test_problem_grouped_batches_are_single_problem_and_cover_every_row():
    tri = make_tri(n_problems=4, plans_per_problem=3, steps=5)   # 15 rows per problem
    sampler = ProblemGroupedBatchSampler(as_ds(tri), batch_size=4, shuffle_within_problem=True, seed=0)
    batches = list(sampler)
    probs = tri["problem_idx"].to_numpy()
    assert all(len({probs[i] for i in b}) == 1 for b in batches)
    assert Counter(i for b in batches for i in b) == Counter(range(len(tri)))
    assert all(len(b) <= 4 for b in batches)
    assert sorted(len(b) for b in batches if len(b) < 4) == [3] * 4    # one short tail per problem
    assert len(sampler) == len(batches)


def test_problem_grouped_drop_last_drops_only_short_tails():
    tri = make_tri(n_problems=4, plans_per_problem=3, steps=5)
    sampler = ProblemGroupedBatchSampler(as_ds(tri), batch_size=4, drop_last=True)
    batches = list(sampler)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) == len(sampler) == 4 * 3


def test_problem_grouped_sampler_respects_indices_and_order():
    """Without shuffling the batches follow the given indices, grouped by problem in
    order of first appearance; matched-pool numbers depend on this being stable."""
    tri = make_tri(n_problems=4)
    idx = list(range(0, len(tri), 2))
    sampler = ProblemGroupedBatchSampler(as_ds(tri), batch_size=5, indices=idx, shuffle_problems=False)
    flat = [i for b in sampler for i in b]
    probs = tri["problem_idx"].to_numpy()
    first_seen = list(dict.fromkeys(probs[idx].tolist()))
    assert flat == sorted(idx, key=lambda i: first_seen.index(probs[i]))


def test_problem_grouped_sampler_reshuffles_each_epoch():
    tri = make_tri(n_problems=3)
    sampler = ProblemGroupedBatchSampler(as_ds(tri), batch_size=100, shuffle_within_problem=True, seed=0)
    first, second = list(sampler), list(sampler)
    assert first != second
    assert sorted(map(sorted, first)) == sorted(map(sorted, second))


def test_problem_grouped_sampler_requires_problem_ids():
    with pytest.raises(ValueError):
        ProblemGroupedBatchSampler(as_ds(make_tri().drop(columns="problem_idx")), batch_size=4)


# ----------------------------------------------------------------------------- plans and trajectories

@pytest.mark.parametrize("raw, expected", [
    ("['(board c0 l0)', '(sail l0 l1)']", ["(board c0 l0)", "(sail l0 l1)"]),
    ("[['(board c0 l0)', '(sail l0 l1)'], ['(sail l0 l2)']]", ["(board c0 l0)", "(sail l0 l1)"]),
    ("[]", []),
    ("['(board c0 l0)', 3]", []),
    ("42", []),
    ("not a plan (", []),
])
def test_parse_plan(raw, expected):
    assert parse_plan(raw) == expected


def test_build_trajectories_reconstructs_every_plan_as_a_chain():
    tri = make_tri(n_problems=2, plans_per_problem=3, steps=5)
    trajs = build_trajectories(tri, range(len(tri)), max_trajs=100, seed=0)
    assert len(trajs) == 6
    for s, a, sp in trajs:
        assert len(s) == len(a) == len(sp) == 5
        assert (sp[:-1] == s[1:]).all()


def test_build_trajectories_keeps_the_longest_contiguous_run():
    tri = make_tri(n_problems=1, plans_per_problem=1, steps=6)
    order = tri.sort_values("goal_val", ascending=False).index.tolist()
    valid = [i for i in order if i != order[2]]             # break the chain after step 2
    [(s, _, sp)] = build_trajectories(tri, valid, max_trajs=10, seed=0)
    assert s.tolist() == tri.loc[order[3:], "s_emb_idx"].tolist()
    assert (sp[:-1] == s[1:]).all()


def test_build_trajectories_filters_short_runs_and_subsamples():
    tri = make_tri(n_problems=3, plans_per_problem=3, steps=4)
    assert build_trajectories(tri, range(len(tri)), max_trajs=100, seed=0, min_len=5) == []
    a = build_trajectories(tri, range(len(tri)), max_trajs=4, seed=7)
    b = build_trajectories(tri, range(len(tri)), max_trajs=4, seed=7)
    assert len(a) == 4
    assert all((x[0] == y[0]).all() for x, y in zip(a, b))


def test_build_trajectories_only_uses_valid_rows():
    tri = make_tri(n_problems=4)
    _, va = make_split(as_ds(tri), tri, "problem_grouped", seed=0)
    valid_states = set(tri["s_emb_idx"].to_numpy()[va].tolist())
    for s, _, _ in build_trajectories(tri, va, max_trajs=100, seed=0):
        assert set(s.tolist()) <= valid_states


# ----------------------------------------------------------------------------- the real loader on the toy domain

def test_toy_domain_loads_through_the_real_loader(toy_domain):
    ds, tri = toy_domain
    assert len(ds) == len(tri) > 100
    assert (tri["a_idx"] >= 0).all()
    assert not tri.duplicated(["s_emb_idx", "a_idx"]).any()     # a deterministic transition table
    goal_vals = ds.values["goal_distance"]
    assert (tri["goal_val"] == [int(goal_vals[i]) for i in tri["goal_distance_idx"]]).all()


def test_toy_domain_embedding_rows_match_their_prompts(toy_domain):
    """s_emb_idx must be the row cached for the prompt regenerated from the row's indices."""
    ds, _ = toy_domain
    for _, row in ds.df.sample(20, random_state=0).iterrows():
        assert ds.state_prompt_map[create_original_prompt(row, ds.values)] == row["s_emb_idx"]


def test_toy_domain_items_are_consistent(toy_domain):
    ds, tri = toy_domain
    item = ds[5]
    row = tri.iloc[5]
    assert item["a_str"] == ds.action_vocab[row["a_idx"]]
    assert torch.equal(item["s_emb"], torch.as_tensor(ds.state_embs[row["s_emb_idx"]]))
    assert torch.equal(item["sp_emb"], torch.as_tensor(ds.state_embs[row["sp_emb_idx"]]))
    assert item["s_str"] != item["sp_str"]
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert batch["s_emb"].shape == (8, ds.state_embs.shape[1])
    assert batch["a_emb"].shape == (8, ds.action_embs.shape[1])
    assert batch["s_emb"].dtype == torch.float32


def test_toy_domain_splits_and_trajectories(toy_domain):
    ds, tri = toy_domain
    tr, va = make_split(ds, tri, "problem_grouped", seed=0)
    assert labels(tri, tr, "problem_idx").isdisjoint(labels(tri, va, "problem_idx"))
    trajs = build_trajectories(tri, va, max_trajs=50, seed=0)
    assert trajs and all((sp[:-1] == s[1:]).all() for s, _, sp in trajs)
    for batch in ds.iter_problem_batches(16, indices=va):
        assert len({item["problem_id"] for item in batch}) == 1
