"""EmbedPlan: learning action-conditioned transitions in frozen LLM embedding spaces.

The package holds everything reusable; `experiments/` holds thin CLI entry points
that compose it. Import surface:

    from embedplan import Config, load_domain, make_split, build_model, train_transition
    from embedplan.scoring import hit_at_k, ABSOLUTE, DELTA
"""

from embedplan.config import Config, load_config
from embedplan.data import (
    FactorizedTripletDataset,
    ProblemGroupedBatchSampler,
    grouped_split_by_problem,
    grouped_split_by_plan,
    load_domain,
    make_split,
    build_trajectories,
)
from embedplan.models import (
    TransitionMLP,
    TransitionHyper,
    ProjectionHead,
    ProjectedTransitionModel,
    build_model,
)
from embedplan.training import train_transition

__all__ = [
    "Config",
    "load_config",
    "FactorizedTripletDataset",
    "ProblemGroupedBatchSampler",
    "grouped_split_by_problem",
    "grouped_split_by_plan",
    "load_domain",
    "make_split",
    "build_trajectories",
    "TransitionMLP",
    "TransitionHyper",
    "ProjectionHead",
    "ProjectedTransitionModel",
    "build_model",
    "train_transition",
]
