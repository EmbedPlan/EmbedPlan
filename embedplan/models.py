"""Transition architectures.

Attribute names here are load-bearing: checkpoints written before the refactor
key on `state_projection_head` / `action_projection_head` / `transition_model`,
and on `net` / `ln` / `res` inside TransitionMLP. Do not rename without a
migration for results/rebuttal/ckpt_*.pt.

The residual MLP outperforms the hypernetwork across encoders and protocols, so
it is the default everywhere; TransitionHyper is kept for the ablation.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TransitionMLP", "AnchoredTransitionMLP", "TransitionHyper", "ProjectionHead",
           "ProjectedTransitionModel", "model_selection", "build_model"]


class TransitionMLP(nn.Module):
    """Concatenate [s, a] -> MLP -> s'. Residual on s, LayerNorm for stability."""

    def __init__(self, d_state: int, d_action: int, hidden: int = 2048, n_layers: int = 2,
                 normalize_output: bool = True, dropout: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        dims = [d_state + d_action] + [hidden] * (n_layers - 1) + [d_state]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if use_layer_norm and i < len(dims) - 3:  # never right before the output layer
                    layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(d_state) if use_layer_norm else nn.Identity()
        self.res = nn.Linear(d_state, d_state, bias=False)
        self.normalize_output = normalize_output

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(torch.cat([s, a], dim=-1)) + self.res(s))


class AnchoredTransitionMLP(nn.Module):
    """pred = z_s + delta(z_s, z_a), with delta zero-initialized at the output.

    Two measured pathologies motivate this:

    1. On every extrapolation checkpoint the trained residual MLP lands *further*
       from the true next state than the current state does — cos(pred, s') is
       0.78 against cos(s, s') 0.96 on ferry, 0.69 vs 0.83 on logistics. InfoNCE
       only enforces relative ranking within a 128-example batch, so a model can
       minimize it while degrading absolute geometry, which is exactly what
       full-pool retrieval depends on. Anchoring makes "predict no change" the
       floor the network starts from instead of something it must rediscover:
       with the last layer zero-initialized, pred == z_s exactly at step 0.

    2. TransitionMLP's output LayerNorm pins ||pred|| near 11 while the projected
       state manifold sits at 25-42 and varies by domain, so the prediction cannot
       land on the manifold even in principle. Here the residual carries the scale
       and no LayerNorm is applied to the sum.
    """

    def __init__(self, d_state: int, d_action: int, hidden: int = 256, n_layers: int = 2,
                 dropout: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        dims = [d_state + d_action] + [hidden] * (n_layers - 1) + [d_state]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return s + self.net(torch.cat([s, a], dim=-1))


class TransitionHyper(nn.Module):
    """Action-conditioned adapters: the action emits per-layer (scale, shift)."""

    def __init__(self, d_state: int, d_action: int, hidden: int = 768, n_layers: int = 4,
                 adapter_size: int = 192, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(d_state, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers - 1)])
        self.outp = nn.Linear(hidden, d_state)
        self.res = nn.Linear(d_state, d_state, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.hyper = nn.Sequential(
            nn.Linear(d_action, 2 * adapter_size),
            nn.GELU(),
            nn.Linear(2 * adapter_size, n_layers * 2 * adapter_size),
        )
        self.A_proj = nn.Linear(adapter_size, hidden, bias=False)
        self.b_proj = nn.Linear(adapter_size, hidden, bias=False)
        self.n_layers = n_layers
        self.adapter_size = adapter_size

    def _adapt(self, h, a_emb, i):
        h = self.lns[i](h)
        chunk = a_emb[:, i * 2 * self.adapter_size:(i + 1) * 2 * self.adapter_size]
        A_vec, b_vec = chunk.chunk(2, dim=-1)
        return h * (1 + self.A_proj(A_vec)) + self.b_proj(b_vec)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_emb = self.hyper(a)
        h = F.gelu(self._adapt(self.inp(s), a_emb, 0))
        for i, lin in enumerate(self.blocks, start=1):
            h = self.dropout(F.gelu(self._adapt(lin(h), a_emb, i)))
        return self.outp(h) + self.res(s)


class ProjectionHead(nn.Module):
    """Reduces a frozen encoder's output to the transition network's working dim."""

    def __init__(self, input_dim: int, output_dim: int = 512, n_layers: int = 2):
        super().__init__()
        dims = [input_dim] + [input_dim] * (n_layers - 1) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]),
                           nn.LayerNorm(dims[i + 1]),
                           nn.GELU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectedTransitionModel(nn.Module):
    """Project state and action into a shared low-dim space, then transition there."""

    def __init__(self, state_projection_head: nn.Module, action_projection_head: nn.Module,
                 transition_model: nn.Module):
        super().__init__()
        self.state_projection_head = state_projection_head
        self.action_projection_head = action_projection_head
        self.transition_model = transition_model

    def forward(self, s: torch.Tensor, a: torch.Tensor, return_projections: bool = False):
        s_proj = self.state_projection_head(s)
        a_proj = self.action_projection_head(a)
        sp_pred = self.transition_model(s_proj, a_proj)
        return (sp_pred, s_proj, a_proj) if return_projections else sp_pred


def model_selection(action_dim, args, state_dim):
    """Bare transition network, no projection heads (used by train.py)."""
    if args.model_type == "mlp":
        return TransitionMLP(d_state=state_dim, d_action=action_dim, hidden=args.hidden_size,
                             n_layers=args.n_layers, dropout=args.dropout,
                             use_layer_norm=args.use_layer_norm)
    if args.model_type == "hyper":
        return TransitionHyper(d_state=state_dim, d_action=action_dim, hidden=args.hidden_size,
                               n_layers=args.n_layers, adapter_size=192, dropout=args.dropout)
    raise ValueError(f"Unknown model_type: {args.model_type}")


TRANSITIONS = {"mlp": TransitionMLP, "anchored": AnchoredTransitionMLP}


def build_model(state_dim: int, action_dim: int, args, device) -> ProjectedTransitionModel:
    """Projected transition model. `args.transition` selects the transition net;
    'mlp' is the published residual MLP, 'anchored' the identity-anchored variant."""
    kind = getattr(args, "transition", "mlp")
    if kind not in TRANSITIONS:
        raise ValueError(f"unknown transition {kind!r}, expected one of {list(TRANSITIONS)}")
    s_proj = ProjectionHead(state_dim, args.projection_dim, n_layers=args.projection_layers)
    a_proj = ProjectionHead(action_dim, args.projection_dim, n_layers=args.projection_layers)
    trans = TRANSITIONS[kind](d_state=args.projection_dim, d_action=args.projection_dim,
                              hidden=args.hidden_size, n_layers=args.n_layers,
                              dropout=getattr(args, "dropout", 0.0), use_layer_norm=True)
    return ProjectedTransitionModel(s_proj, a_proj, trans).to(device)
