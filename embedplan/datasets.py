"""Small text planning datasets for trying EmbedPlan and for tests.

`load_toy_ferry` builds a ferry world in plain English: a ferry carries one car at a time between
docks, and each problem has its own cars, docks and goals. It returns transitions the way
scikit-learn's loaders return data, so the estimator API can be tried in a few seconds:

    from embedplan.datasets import load_toy_ferry
    data = load_toy_ferry()
    data.X[0]        # ("The ferry is at the north dock. ...", "sail from the north dock to the east pier")
    data.y[0]        # the next state, as text
    data.groups[0]   # the problem it comes from

It is a smoke test and a demo, not a benchmark: the paper's nine domains are far harder.
"""

import itertools
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.utils import Bunch

_COLORS = ["red", "blue", "green", "white", "black", "silver", "yellow", "orange"]
_DOCKS = ["the north dock", "the south dock", "the east pier", "the west pier", "the harbor", "the island"]


def _describe(ferry: str, cars: Dict[str, Optional[str]], on: Optional[str], goal: Dict[str, str]) -> str:
    parts = [f"The ferry is at {ferry}", f"The ferry carries the {on} car" if on else "The ferry is empty"]
    parts += [f"The {c} car is at {loc}" for c, loc in sorted(cars.items()) if loc is not None]
    goal_text = ", ".join(f"the {c} car to {loc}" for c, loc in sorted(goal.items()))
    return ". ".join(parts) + f". Goal: bring {goal_text}."


def _plan(ferry: str, cars: Dict[str, str], goal: Dict[str, str], order) -> Tuple[List[str], List[str]]:
    """Deliver the cars one at a time in `order`: the states along the plan and its actions."""
    cars, on = dict(cars), None
    states, actions = [_describe(ferry, cars, on, goal)], []
    for c in order:
        steps = [] if ferry == cars[c] else [("sail", ferry, cars[c])]
        steps += [("board", c, cars[c]), ("sail", cars[c], goal[c]), ("debark", c, goal[c])]
        for kind, a, b in steps:
            if kind == "sail":
                ferry = b
                actions.append(f"sail from {a} to {b}")
            elif kind == "board":
                cars[a], on = None, a
                actions.append(f"board the {a} car at {b}")
            else:
                cars[a], on = b, None
                actions.append(f"debark the {a} car at {b}")
            states.append(_describe(ferry, cars, on, goal))
    return states, actions


def load_toy_ferry(n_problems: int = 12, n_plans: int = 3, n_cars: int = 3, random_state: int = 0) -> Bunch:
    """Transitions from a toy ferry world, one plan-following trajectory at a time.

    Returns a Bunch with
      X        list of (state, action) pairs, as text
      y        list of next states, as text
      groups   the problem id of each transition (for splits and grouped training)
      frame    a DataFrame with columns problem, plan, step, state, action, next_state
    """
    if not 1 <= n_plans <= 6 or not 2 <= n_cars <= 4:
        raise ValueError("n_plans must be in 1..6 and n_cars in 2..4")
    rng = random.Random(random_state)
    rows = []
    for p in range(n_problems):
        colors = rng.sample(_COLORS, n_cars)
        docks = rng.sample(_DOCKS, 3 + p % 2)
        cars = {c: rng.choice(docks) for c in colors}
        goal = {c: rng.choice([d for d in docks if d != at]) for c, at in cars.items()}
        ferry = rng.choice(docks)
        orders = list(itertools.permutations(colors))
        for k, order in enumerate(rng.sample(orders, min(n_plans, len(orders)))):
            states, actions = _plan(ferry, cars, goal, order)
            for t, action in enumerate(actions):
                rows.append({"problem": p, "plan": k, "step": t, "state": states[t],
                             "action": action, "next_state": states[t + 1]})
    frame = pd.DataFrame(rows)
    return Bunch(X=list(zip(frame["state"], frame["action"])), y=frame["next_state"].tolist(),
                 groups=frame["problem"].to_numpy(), frame=frame,
                 DESCR=__doc__.split("\n\n")[1].strip())
