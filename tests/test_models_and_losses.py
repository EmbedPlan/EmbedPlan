"""Transition networks and training objectives on random tensors (CPU, no data files)."""

from types import SimpleNamespace

import pytest
import torch

from embedplan.losses import (
    compute_action_loss,
    compute_delta_action_loss,
    compute_delta_infonce_loss,
    compute_infonce_loss,
)
from embedplan.models import AnchoredTransitionMLP, TransitionHyper, TransitionMLP, build_model
from embedplan.scoring import project_pool


def small_args(transition="mlp"):
    return SimpleNamespace(projection_dim=16, projection_layers=2, hidden_size=32, n_layers=2,
                           dropout=0.0, transition=transition)


@pytest.mark.parametrize("net", [TransitionMLP, TransitionHyper])
def test_transition_networks_keep_the_state_dimension(net):
    model = net(d_state=24, d_action=10, hidden=32, n_layers=3)
    out = model(torch.randn(7, 24), torch.randn(7, 10))
    assert out.shape == (7, 24)
    assert torch.isfinite(out).all()


def test_anchored_mlp_starts_as_the_identity():
    model = AnchoredTransitionMLP(d_state=16, d_action=8)
    s = torch.randn(5, 16)
    assert torch.equal(model(s, torch.randn(5, 8)), s)


@pytest.mark.parametrize("transition", ["mlp", "anchored"])
def test_build_model_projects_then_transitions(transition):
    model = build_model(state_dim=40, action_dim=12, args=small_args(transition), device="cpu")
    pred, s_proj, a_proj = model(torch.randn(6, 40), torch.randn(6, 12), return_projections=True)
    assert pred.shape == s_proj.shape == a_proj.shape == (6, 16)
    assert project_pool(model, torch.randn(9, 40)).shape == (9, 16)


def test_build_model_rejects_unknown_transition():
    with pytest.raises(ValueError):
        build_model(8, 8, small_args("nope"), "cpu")


def test_infonce_prefers_the_true_pairing():
    torch.manual_seed(0)
    true = torch.randn(32, 16)
    aligned = compute_infonce_loss(true.clone(), true)
    shuffled = compute_infonce_loss(true[torch.randperm(32)], true)
    assert aligned < shuffled


def test_delta_infonce_prefers_the_true_displacement():
    torch.manual_seed(0)
    anchor, true = torch.randn(32, 16), torch.randn(32, 16)
    aligned = compute_delta_infonce_loss(true.clone(), true, anchor)
    shuffled = compute_delta_infonce_loss(true[torch.randperm(32)], true, anchor)
    assert aligned < shuffled


def test_action_losses_are_finite_and_backpropagate():
    torch.manual_seed(0)
    model = build_model(state_dim=20, action_dim=10, args=small_args(), device="cpu")
    s, a, sp = torch.randn(16, 20), torch.randn(16, 10), torch.randn(16, 20)
    for delta in (False, True):
        sp_proj = model.state_projection_head(sp)
        if delta:
            anchor = model.state_projection_head(s)
            loss = compute_delta_action_loss(model, s, a, sp_proj, anchor, tau=0.1, m=4, device="cpu")
        else:
            loss = compute_action_loss(model, s, a, sp_proj, tau=0.1, m=4, device="cpu")
        assert torch.isfinite(loss)
        model.zero_grad()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
