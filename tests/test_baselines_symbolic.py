"""Floor baselines (identity, offset, lifted offset, ridge) and lifted STRIPS induction."""

import pandas as pd
import pytest
import torch

from embedplan.baselines import IdentityTransition, LiftedOffsetTransition, OffsetTransition, RidgeTransition
from embedplan.data import make_split
from embedplan.evaluation import pool_sweep
from embedplan.symbolic import (
    LiftedActionModel,
    action_parts,
    ground_literal,
    jaccard,
    lift_literal,
    parse_literals,
    schema_index,
    symbolic_frame,
)


def displaced_world(n_base=30, n_trans=60, dim=8, n_actions=4, used=(0, 1, 2), seed=0):
    """s' = s + d[a] exactly, for actions in `used`. Next states are appended to S."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_base, dim, generator=g)
    d = torch.randn(n_actions, dim, generator=g)
    s_idx = torch.randint(0, n_base, (n_trans,), generator=g)
    a_idx = torch.as_tensor(used)[torch.randint(0, len(used), (n_trans,), generator=g)]
    S = torch.cat([base, base[s_idx] + d[a_idx]])
    sp_idx = torch.arange(n_base, n_base + n_trans)
    return S, d, s_idx, a_idx, sp_idx


# ----------------------------------------------------------------------------- embedding-space baselines

def test_identity_predicts_no_change_in_raw_space():
    model = IdentityTransition()
    s = torch.randn(3, 5)
    assert model.eval() is model
    assert torch.equal(model(s, torch.randn(3, 2)), s)
    assert torch.equal(model.state_projection_head(s), s)


def test_offset_recovers_a_constant_per_action_displacement():
    S, d, s_idx, a_idx, sp_idx = displaced_world()
    model = OffsetTransition(n_actions=4, dim=8, device="cpu").fit(S, a_idx, s_idx, sp_idx)
    assert model.seen.tolist() == [True, True, True, False]
    assert torch.allclose(model.delta[:3], d[:3], atol=1e-5)
    assert torch.allclose(model(S[s_idx], a_idx), S[sp_idx], atol=1e-5)


def test_offset_unseen_action_falls_back_to_the_global_mean():
    S, _, s_idx, a_idx, sp_idx = displaced_world()
    model = OffsetTransition(n_actions=4, dim=8, device="cpu").fit(S, a_idx, s_idx, sp_idx)
    global_mean = (S[sp_idx] - S[s_idx]).mean(0)
    assert torch.allclose(model.global_delta, global_mean, atol=1e-6)
    assert torch.allclose(model.delta[3], global_mean, atol=1e-6)


def test_offset_scores_through_pool_sweep_by_action_index():
    S, _, s_idx, a_idx, sp_idx = displaced_world()
    model = OffsetTransition(n_actions=4, dim=8, device="cpu").fit(S, a_idx, s_idx, sp_idx)
    tri = pd.DataFrame({"s_emb_idx": s_idx.numpy(), "a_idx": a_idx.numpy(), "sp_emb_idx": sp_idx.numpy()})
    out = pool_sweep(model, S, None, tri, list(range(len(tri))), S, "cpu", sizes=(-1,), seed=0, action_index=True)
    assert out["full"]["hit@1"] == 1.0


def test_lifted_offset_generalizes_across_groundings_of_a_schema():
    """Grounded actions 0,1 share schema 0 and 2,3 share schema 1; only 0 and 2 are
    trained. Grounded action 1 is unseen but its schema is not, so the lifted table
    predicts it exactly where the grounded table can only fall back to the mean."""
    S, d, s_idx, a_idx, sp_idx = displaced_world(n_actions=6, used=(0, 2))
    schema_of = [0, 0, 1, 1, 2, 2]
    lifted = LiftedOffsetTransition(schema_of, n_schemas=3, dim=8, device="cpu").fit(S, a_idx, s_idx, sp_idx)
    grounded = OffsetTransition(n_actions=6, dim=8, device="cpu").fit(S, a_idx, s_idx, sp_idx)
    s = S[:5]
    one = torch.ones(5, dtype=torch.long)
    assert lifted.seen.tolist() == [True, True, False]
    assert torch.allclose(lifted(s, one), s + d[0], atol=1e-5)
    assert torch.allclose(grounded(s, one), s + grounded.global_delta, atol=1e-6)
    assert torch.allclose(lifted(s, 4 * one), s + lifted.global_delta, atol=1e-6)   # unseen schema


def test_ridge_recovers_a_linear_map():
    g = torch.Generator().manual_seed(0)
    base, A = torch.randn(40, 6, generator=g), torch.randn(10, 4, generator=g)
    W = torch.randn(10, 6, generator=g)
    s_idx, a_idx = torch.randint(0, 40, (200,), generator=g), torch.randint(0, 10, (200,), generator=g)
    Y = torch.cat([base[s_idx], A[a_idx]], dim=1) @ W
    S, sp_idx = torch.cat([base, Y]), torch.arange(40, 240)
    model = RidgeTransition(dim_s=6, dim_a=4, device="cpu", lam=1e-6).fit(S, A, s_idx, a_idx, sp_idx)
    assert torch.allclose(model.W, W, atol=1e-3)
    assert torch.allclose(model(S[s_idx], A[a_idx]), Y, atol=1e-3)
    chunked = RidgeTransition(dim_s=6, dim_a=4, device="cpu", lam=1e-6).fit(S, A, s_idx, a_idx, sp_idx, chunk=7)
    assert torch.allclose(chunked.W, model.W, atol=1e-5)


# ----------------------------------------------------------------------------- lifted STRIPS induction

S0 = frozenset({"(at-ferry l1)", "(at c1 l1)", "(empty-ferry)", "(at c2 l0)"})
S1 = frozenset({"(at-ferry l1)", "(on c1)", "(at c2 l0)"})


def test_lifted_model_learns_board_from_one_transition():
    model = LiftedActionModel().fit([(S0, "(board c1 l1)", S1)])
    assert model.effects["board"] == {"add": frozenset({"(on ?p0)"}),
                                      "del": frozenset({"(at ?p0 ?p1)", "(empty-ferry)"})}
    new_state = frozenset({"(at-ferry l7)", "(at c9 l7)", "(empty-ferry)", "(at c3 l2)"})
    assert model.predict(new_state, "(board c9 l7)") == {"(at-ferry l7)", "(on c9)", "(at c3 l2)"}
    assert model.summary()["n_schemas"] == 1 and model.conflicts == 0


def test_lifted_model_returns_none_for_an_unseen_schema():
    model = LiftedActionModel().fit([(S0, "(board c1 l1)", S1)])
    assert model.predict(S1, "(sail l1 l0)") is None


def test_lifted_model_counts_conflicting_transitions():
    consistent = (frozenset({"(at c5 l3)", "(empty-ferry)"}), "(board c5 l3)", frozenset({"(on c5)"}))
    no_empty = (frozenset({"(at c5 l3)"}), "(board c5 l3)", frozenset({"(on c5)"}))
    model = LiftedActionModel().fit([(S0, "(board c1 l1)", S1), consistent, no_empty, (S0, "()", S1)])
    assert (model.n_fit, model.conflicts) == (3, 1)
    assert "(empty-ferry)" in model.effects["board"]["del"]      # the first observation is kept


@pytest.mark.parametrize("literal, params, lifted", [
    ("(at c1 l1)", ["c1", "l1"], "(at ?p0 ?p1)"),
    ("(at c1 l9)", ["c1", "l1"], "(at ?p0 l9)"),       # a non-argument constant stays literal
    ("(sail-to l1 l0)", ["l0", "l1"], "(sail-to ?p1 ?p0)"),
    ("(empty-ferry)", ["c1"], "(empty-ferry)"),
])
def test_lift_and_ground_round_trip(literal, params, lifted):
    assert lift_literal(literal, params) == lifted
    assert ground_literal(lifted, params) == literal


def test_ground_literal_leaves_an_out_of_range_parameter():
    assert ground_literal("(at ?p0 ?p3)", ["c1"]) == "(at c1 ?p3)"


def test_action_parts():
    assert action_parts(" (board c1 l1)\n") == ("board", ["c1", "l1"])
    assert action_parts("(empty-ferry)") == ("empty-ferry", [])
    assert action_parts("()") == ("", [])


def test_jaccard():
    a, b = frozenset({"x", "y"}), frozenset({"y", "z"})
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(a, a) == 1.0
    assert jaccard(a, frozenset({"z"})) == 0.0
    assert jaccard(a, b) == pytest.approx(1 / 3)


def test_schema_index():
    vocab = ["(board c0 l0)", "(sail l0 l1)", "(board c1 l1)", "(debark c0 l1)"]
    assert schema_index(vocab) == ([0, 1, 0, 2], ["board", "sail", "debark"])


@pytest.mark.parametrize("raw, expected", [
    ("['(at c0 l0)', '(on c2)']", {"(at c0 l0)", "(on c2)"}),
    (["(at c0 l0)"], {"(at c0 l0)"}),
    ("[]", set()),
    ("42", set()),
    ("['unterminated", set()),
])
def test_parse_literals(raw, expected):
    assert parse_literals(raw) == frozenset(expected)


# ----------------------------------------------------------------------------- the toy domain end to end

def test_lifted_model_is_exact_on_held_out_toy_problems(toy_domain):
    """symbolic_frame must key every triplet to its own literal sets; if it did not,
    the induced operators would conflict or mispredict."""
    ds, tri = toy_domain
    frame = symbolic_frame(ds, tri)
    assert frame["has_symbolic"].all()
    train, valid = make_split(ds, tri, "problem_grouped", seed=0)

    def transitions(idx):
        return list(zip(frame["s_lits"].iloc[idx], frame["action"].iloc[idx], frame["sp_lits"].iloc[idx]))

    model = LiftedActionModel().fit(transitions(train))
    assert model.conflicts == 0 and set(model.effects) == {"sail", "board", "debark"}
    assert all(model.predict(s, a) == sp for s, a, sp in transitions(valid))
    assert sorted(schema_index(ds.action_vocab)[1]) == ["board", "debark", "sail"]
