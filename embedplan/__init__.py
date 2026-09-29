"""EmbedPlan: learning action-conditioned transitions in frozen LLM embedding spaces.

On your own data, use the scikit-learn style estimator:

    from embedplan import EmbedPlan, split_transitions
    model = EmbedPlan(encoder="BAAI/bge-m3").fit(X, y, groups=problem_ids)

The building blocks behind the paper's experiments are importable too:

    from embedplan import Config, load_domain, make_split, build_model, train_transition
    from embedplan.scoring import hit_at_k, ABSOLUTE, DELTA
"""

from embedplan.config import Config, load_config
from embedplan.estimator import EmbedPlan, split_transitions, transitions_from_trajectories
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
    "EmbedPlan",
    "split_transitions",
    "transitions_from_trajectories",
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
