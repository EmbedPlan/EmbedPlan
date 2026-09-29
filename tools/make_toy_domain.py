"""Write a tiny synthetic planning domain in the exact on-disk format EmbedPlan reads.

This is for a smoke test and a 30-second demo of the pipeline, not a benchmark. The
world is ferry-like (one ferry carries one car at a time between locations): a few
problems, a few plans per problem, natural-language state descriptions, symbolic
states for the lifted baseline, and deterministic pseudo-embeddings (signed feature
hashing of words and whole sentences). Nothing is downloaded and no encoder runs.

Files written under --out, the data root that EMBEDPLAN_DATA points at:
  original_df_pkls_factorized/{domain}-test.pkl          DataFrame, one row per (plan, step), *_idx columns
  original_df_pkls_factorized/{domain}-test_values.pkl   dict: value name -> list the *_idx columns index
  full_embeddings/{encoder}/original/{domain}.pt                    float32 (n_prompts, dim) tensor
  full_embeddings/{encoder}/original/{domain}_prompt_to_index.pkl   prompt text -> row of that tensor
  full_embeddings_actions/{encoder}/original/{domain}_actions.pt    float32 (n_actions, dim) tensor
  full_embeddings_actions/{encoder}/original/{domain}_actions_index.pkl  {"actions": [...], ...}

Demo, from the repo root (a few seconds on a laptop CPU; Hit@1 ~0.8 on held-out problems):
  python -m tools.make_toy_domain --out /tmp/embedplan_toy
  EMBEDPLAN_DATA=/tmp/embedplan_toy python -m experiments.train --domain toy --model_name toy-hash-bow \\
      --split_type problem_grouped --epochs 100 --batch_size 32 --val_batch_size 32 --no_wandb \\
      --use_projection --projection_dim 32 --hidden_size 64 --n_layers 2 --use_layer_norm \\
      --num_workers 0 --save_prefix /tmp/embedplan_toy/job   # metrics land in /tmp/embedplan_toy/job.json
"""

import argparse
import hashlib
import itertools
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embedplan.data import parse_plan
from embedplan.prompts import create_original_prompt

ENCODER = "toy-hash-bow"
VALUE_KEYS = ("problem", "goal_description", "state_description", "state", "plan", "goal_distance", "plan_id")


def embed(text: str, dim: int) -> np.ndarray:
    """Unit-norm signed hash of words and whole sentences. Stable across runs and machines."""
    feats = re.findall(r"[a-z0-9-]+", text.lower()) + [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
    v = np.zeros(dim, dtype=np.float32)
    for f in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
        v[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    return v / max(float(np.linalg.norm(v)), 1e-8)


def _snapshot(ferry, cars, on):
    lits = [f"(at-ferry {ferry})", f"(on {on})" if on else "(empty-ferry)"]
    lits += [f"(at {c} {loc})" for c, loc in sorted(cars.items()) if loc is not None]
    text = [f"The ferry is at {ferry}", f"The ferry is carrying {on}" if on else "The ferry is empty"]
    text += [f"Car {c} is at {loc}" for c, loc in sorted(cars.items()) if loc is not None]
    return ". ".join(text) + ".", str(sorted(lits))


def solve(ferry, cars, goal, order):
    """Deliver cars one at a time in `order`. Returns (states along the plan, actions)."""
    cars, on = dict(cars), None
    states, actions = [_snapshot(ferry, cars, on)], []
    for c in order:
        steps = [] if ferry == cars[c] else [f"(sail {ferry} {cars[c]})"]
        steps += [f"(board {c} {cars[c]})", f"(sail {cars[c]} {goal[c]})", f"(debark {c} {goal[c]})"]
        for act in steps:
            name, *args = act.strip("()").split()
            if name == "sail":
                ferry = args[1]
            elif name == "board":
                cars[args[0]], on = None, args[0]
            else:
                cars[args[0]], on = args[1], None
            actions.append(act)
            states.append(_snapshot(ferry, cars, on))
    return states, actions


def write_toy_domain(out, domain: str = "toy", encoder: str = ENCODER, n_problems: int = 6,
                     n_plans: int = 3, dim: int = 64, seed: int = 0) -> dict:
    rng = random.Random(seed)
    values = {k: [] for k in VALUE_KEYS}
    index = {k: {} for k in VALUE_KEYS}

    def idx(key, v):
        if v not in index[key]:
            index[key][v] = len(values[key])
            values[key].append(v)
        return index[key][v]

    rows = []
    for p in range(n_problems):
        locs = [f"l{i}" for i in range(3 + p % 2)]
        cars = {f"c{i}": rng.choice(locs) for i in range(3)}
        goal = {c: rng.choice([loc for loc in locs if loc != at]) for c, at in cars.items()}
        ferry = rng.choice(locs)
        problem = (f"Problem {domain}-{p}. A ferry moves cars between locations {', '.join(locs)}, one car at a "
                   f"time. The cars are {', '.join(cars)}.")
        goal_text = " ".join(f"Car {c} should be at {loc}." for c, loc in goal.items())
        for k, order in enumerate(rng.sample(list(itertools.permutations(cars)), n_plans)):
            states, actions = solve(ferry, cars, goal, order)
            for t, (text, lits) in enumerate(states):
                rows.append({"problem_idx": idx("problem", problem), "plan_id_idx": idx("plan_id", f"{domain}-{p}-{k}"),
                             "plan_idx": idx("plan", str(actions[t:])),
                             "goal_distance_idx": idx("goal_distance", len(actions) - t),
                             "state_description_idx": idx("state_description", text), "state_idx": idx("state", lits),
                             "goal_description_idx": idx("goal_description", goal_text)})
    df = pd.DataFrame(rows).astype("int64")

    prompts = list(dict.fromkeys(create_original_prompt(r, values) for r in df.itertuples()))
    actions = list(dict.fromkeys(a for plan in values["plan"] for a in parse_plan(plan)))

    out = Path(out)
    tables = out / "original_df_pkls_factorized"
    s_dir, a_dir = (out / base / encoder / "original" for base in ("full_embeddings", "full_embeddings_actions"))
    for d in (tables, s_dir, a_dir):
        d.mkdir(parents=True, exist_ok=True)
    df.to_pickle(tables / f"{domain}-test.pkl")
    with open(tables / f"{domain}-test_values.pkl", "wb") as f:
        pickle.dump(values, f)
    torch.save(torch.from_numpy(np.stack([embed(t, dim) for t in prompts])), s_dir / f"{domain}.pt")
    with open(s_dir / f"{domain}_prompt_to_index.pkl", "wb") as f:
        pickle.dump({t: i for i, t in enumerate(prompts)}, f)
    torch.save(torch.from_numpy(np.stack([embed(a, dim // 2) for a in actions])), a_dir / f"{domain}_actions.pt")
    with open(a_dir / f"{domain}_actions_index.pkl", "wb") as f:
        pickle.dump({"actions": actions, "model_name": encoder, "domain": domain, "text_type": "original"}, f)
    return {"root": str(out), "domain": domain, "encoder": encoder, "rows": len(df),
            "states": len(prompts), "actions": len(actions), "problems": n_problems}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.environ.get("EMBEDPLAN_DATA", "data"), help="data root (EMBEDPLAN_DATA)")
    ap.add_argument("--domain", default="toy")
    ap.add_argument("--encoder", default=ENCODER, help="name used in the embedding paths (--model_name)")
    ap.add_argument("--problems", type=int, default=6)
    ap.add_argument("--plans", type=int, default=3, help="plans per problem (at most 6)")
    ap.add_argument("--dim", type=int, default=64, help="state embedding dim; actions use dim // 2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    info = write_toy_domain(a.out, a.domain, a.encoder, a.problems, a.plans, a.dim, a.seed)
    print(", ".join(f"{k}={v}" for k, v in info.items()))


if __name__ == "__main__":
    main()
