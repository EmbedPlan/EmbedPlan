"""Floor baselines — how much of the reported accuracy needs a learned transition?

The submission's only lower bound is an untrained network with random weights
(~3.9% Hit@5). That is a weak floor. These three are stronger and all but free:

Identity    predict no change at all, s_hat' = E(s). Because state prompts share
            the entire PROBLEM and GOAL blocks and differ by one or two predicates,
            E(s) and E(s') are already very close. Under Extrapolation, where
            distractors come from the query's own problem, this is exactly the
            regime where the paper says embeddings cluster — so this baseline is
            the sharpest test of whether the learned model adds anything.

Offset      s_hat' = E(s) + mean over training of [E(s'_t) - E(s_t)] for the same
            action. Zero parameters beyond a per-action average: tests whether an
            action's effect is a constant translation in embedding space.

Ridge       s_hat' = W [E(s); E(a)], closed-form ridge regression. Tests whether
            the nonlinearity and contrastive objective are load-bearing.

Each exposes the same interface a trained model does — `__call__(s, a)` plus a
`state_projection_head` — so embedplan.scoring works on them unchanged. The
projection head is the identity, i.e. these operate in the raw encoder space.
"""

import torch
import torch.nn as nn


class _RawSpaceBaseline(nn.Module):
    """Common shell: no projection, so scoring happens in the encoder's own space."""

    def __init__(self):
        super().__init__()
        self.state_projection_head = nn.Identity()
        self.action_projection_head = nn.Identity()

    def eval(self):
        return self


class IdentityTransition(_RawSpaceBaseline):
    """s_hat' = E(s). Predicts that the action changes nothing."""

    def forward(self, s, a=None):
        return s


class OffsetTransition(_RawSpaceBaseline):
    """s_hat' = E(s) + delta[a], with delta[a] the mean training displacement of action a.

    Actions unseen in training fall back to the global mean displacement, which is
    the honest behaviour under Extrapolation (a held-out problem uses grounded
    actions over objects the training problems never mention).
    """

    def __init__(self, n_actions: int, dim: int, device):
        super().__init__()
        self.register_buffer("delta", torch.zeros(n_actions, dim, device=device))
        self.register_buffer("seen", torch.zeros(n_actions, dtype=torch.bool, device=device))
        self.register_buffer("global_delta", torch.zeros(dim, device=device))

    @torch.no_grad()
    def fit(self, S, A_idx, s_idx, sp_idx):
        """Accumulate the mean (s' - s) per action over the training triplets."""
        diffs = S[sp_idx] - S[s_idx]
        self.global_delta = diffs.mean(0)
        totals = torch.zeros_like(self.delta)
        counts = torch.zeros(self.delta.shape[0], device=self.delta.device)
        totals.index_add_(0, A_idx, diffs)
        counts.index_add_(0, A_idx, torch.ones_like(A_idx, dtype=totals.dtype))
        seen = counts > 0
        self.delta[seen] = totals[seen] / counts[seen].unsqueeze(1)
        self.delta[~seen] = self.global_delta
        self.seen = seen
        return self

    def forward(self, s, a_idx):
        return s + self.delta[a_idx]


class LiftedOffsetTransition(_RawSpaceBaseline):
    """s_hat' = E(s) + delta[schema(a)] — the offset baseline keyed by action *schema*
    instead of grounded action.

    OffsetTransition above is crippled by grounding, not by difficulty: under
    Extrapolation a held-out problem introduces new objects, so 57.6% of ferry and
    65.5% of logistics test transitions use a grounded action with no training
    displacement at all and fall back to the global mean. Zero test *schemas* are
    unseen, though, so keying on the schema (487 grounded actions -> 3 rows on ferry)
    gives full coverage.

    This is the embedding-space analogue of the lifted symbolic operator, and the gap
    between the two isolates what the embedding representation costs: the symbolic
    version of exactly this generalization is exact, so whatever is lost here is lost
    by representing an action's effect as one fixed translation of E(s).
    """

    def __init__(self, schema_of_action, n_schemas: int, dim: int, device):
        super().__init__()
        self.register_buffer("schema_of", torch.as_tensor(schema_of_action,
                                                          dtype=torch.long, device=device))
        self.register_buffer("delta", torch.zeros(n_schemas, dim, device=device))
        self.register_buffer("seen", torch.zeros(n_schemas, dtype=torch.bool, device=device))
        self.register_buffer("global_delta", torch.zeros(dim, device=device))

    @torch.no_grad()
    def fit(self, S, A_idx, s_idx, sp_idx):
        diffs = S[sp_idx] - S[s_idx]
        self.global_delta = diffs.mean(0)
        sch = self.schema_of[A_idx]
        totals = torch.zeros_like(self.delta)
        counts = torch.zeros(self.delta.shape[0], device=self.delta.device)
        totals.index_add_(0, sch, diffs)
        counts.index_add_(0, sch, torch.ones_like(sch, dtype=totals.dtype))
        seen = counts > 0
        self.delta[seen] = totals[seen] / counts[seen].unsqueeze(1)
        self.delta[~seen] = self.global_delta
        self.seen = seen
        return self

    def forward(self, s, a_idx):
        return s + self.delta[self.schema_of[a_idx]]


class RidgeTransition(_RawSpaceBaseline):
    """s_hat' = W [s; a] fitted in closed form. `lam` regularizes the normal equations."""

    def __init__(self, dim_s: int, dim_a: int, device, lam: float = 1.0):
        super().__init__()
        self.lam = lam
        self.register_buffer("W", torch.zeros(dim_s + dim_a, dim_s, device=device))

    @torch.no_grad()
    def fit(self, S, A, s_idx, a_idx, sp_idx, chunk: int = 8192):
        """Accumulate X^T X and X^T Y in chunks, then solve. X = [s; a], Y = s'."""
        d = self.W.shape[0]
        xtx = torch.zeros(d, d, device=self.W.device, dtype=torch.float64)
        xty = torch.zeros(d, self.W.shape[1], device=self.W.device, dtype=torch.float64)
        for i in range(0, len(s_idx), chunk):
            sl = slice(i, i + chunk)
            X = torch.cat([S[s_idx[sl]], A[a_idx[sl]]], dim=1).double()
            Y = S[sp_idx[sl]].double()
            xtx += X.T @ X
            xty += X.T @ Y
        xtx.diagonal().add_(self.lam)
        self.W = torch.linalg.solve(xtx, xty).to(self.W.dtype)
        return self

    def forward(self, s, a):
        return torch.cat([s, a], dim=1) @ self.W
