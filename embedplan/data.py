"""Transition datasets, problem-aware batching, and the evaluation splits.

A domain's data arrives factorized: a DataFrame of indices plus a `values` dict
holding the unique strings. States are matched to their cached embedding row by
regenerating the exact prompt used at encoding time and looking it up.

Splits
------
random / interpolation  : transitions shuffled; a problem may appear in both sides.
problem_grouped / extrapolation : whole problems held out.
leave_one_problem_out   : train on a chosen subset of problems, test on exactly one.
"""

import ast
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from embedplan.config import Config, DEFAULT_ENCODER
from embedplan.prompts import create_prompt


def parse_plan(plan_raw) -> List[str]:
    """Plans are stored as stringified lists, occasionally nested one level."""
    try:
        parsed = ast.literal_eval(plan_raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        if all(isinstance(x, str) for x in parsed):
            return parsed
        if all(isinstance(x, list) for x in parsed) and parsed and all(isinstance(y, str) for y in parsed[0]):
            return parsed[0]
    return []


class FactorizedTripletDataset(Dataset):
    """(s, a, s') triplets over cached frozen embeddings for one domain."""

    def __init__(self, domain: str, model_name: str = DEFAULT_ENCODER,
                 text_type: str = "original", verbose: bool = True):
        self.domain = domain
        self.model_name = model_name
        self.text_type = text_type
        self.verbose = verbose

        self._log(f"--- Initializing dataset for domain: {domain} ---")
        self.df, self.values = Config.read_factorized(domain)

        self.state_embs, self.state_prompt_map = Config.load_embeddings(
            model_name, text_type, domain, states_or_actions="states")
        self.action_embs, action_index_info = Config.load_embeddings(
            model_name, text_type, domain, states_or_actions="actions")

        self.action_vocab = action_index_info["actions"]
        self.action_to_idx = {a: i for i, a in enumerate(self.action_vocab)}

        self._map_states_to_embedding_indices()
        self._add_action_indices()
        self.triplets = self._build_triplets()
        self._drop_duplicate_states()
        self._log(f"✅ Dataset initialized with {len(self.triplets)} triplets.")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _map_states_to_embedding_indices(self):
        """Regenerate each row's prompt and resolve it to an embedding row.

        The 'original' prompt is a pure function of three index columns, so it is
        built by array lookup rather than per-row. Satellite has 1.3M rows and the
        row loop dominated dataset construction.
        """
        self._log("🗺️ Mapping states to embedding indices...")
        if self.text_type == "original":
            prompts = self._original_prompts(self.df)
        else:
            it = self.df.iterrows()
            if self.verbose:
                it = tqdm(it, total=len(self.df), desc="Mapping states")
            prompts = [create_prompt(row, self.text_type, self.values)[0] for _, row in it]

        lookup = self.state_prompt_map.get
        self.df["s_emb_idx"] = [lookup(p, -1) for p in prompts]
        self.df = self.df[self.df["s_emb_idx"] != -1].reset_index(drop=True)

    def _original_prompts(self, df) -> List[str]:
        """Vectorized equivalent of prompts.create_original_prompt. Must stay
        byte-identical to it — the cached embeddings are keyed by this string."""
        problem = self.values["problem"]
        state = self.values["state_description"]
        goal = self.values["goal_description"]
        p_i = df["problem_idx"].to_numpy()
        s_i = df["state_description_idx"].to_numpy()
        g_i = df["goal_description_idx"].to_numpy()
        return [
            "### PROBLEM DESCRIPTION ###\n"
            f"{problem[p]}\n\n"
            "### CURRENT STATE ###\n"
            f"{state[s]}\n\n"
            "### GOAL DESCRIPTION ###\n"
            f"{goal[g]}"
            for p, s, g in zip(p_i, s_i, g_i)
        ]

    def _add_action_indices(self):
        """The action leaving each state is the first action of its remaining plan.

        Plans repeat across the many states of a trajectory, so literal_eval is
        memoized per plan rather than re-run per row.
        """
        first_action: Dict[int, int] = {}
        for plan_idx in np.unique(self.df["plan_idx"].to_numpy()):
            actions = parse_plan(self.values["plan"][plan_idx])
            first_action[int(plan_idx)] = self.action_to_idx.get(actions[0], -1) if actions else -1

        goal_vals = self.values["goal_distance"]
        plan_i = self.df["plan_idx"].to_numpy()
        gd_i = self.df["goal_distance_idx"].to_numpy()
        self.df["action_idx"] = [
            -1 if int(goal_vals[g]) == 0 else first_action[int(p)]
            for p, g in zip(plan_i, gd_i)
        ]

    def _build_triplets(self) -> pd.DataFrame:
        """Within a plan, states are ordered by descending goal distance, so s' is
        the next row after sorting."""
        self._log("🛠️ Building triplets...")
        goal_vals = self.values["goal_distance"]
        self.df["goal_val"] = self.df["goal_distance_idx"].apply(lambda i: int(goal_vals[i]))

        chunks = []
        for _, group in self.df.groupby("plan_id_idx"):
            g = group.sort_values("goal_val", ascending=False).copy()
            g["sp_emb_idx"] = g["s_emb_idx"].shift(-1)
            g["sp_id"] = g["state_description_idx"].shift(-1)
            g = g.dropna(subset=["sp_emb_idx", "sp_id"])
            if not g.empty:
                chunks.append(g)

        if not chunks:
            return pd.DataFrame()

        tri = pd.concat(chunks)
        tri["sp_emb_idx"] = tri["sp_emb_idx"].astype(int)
        tri["sp_id"] = tri["sp_id"].astype(int)
        cols = {"s_emb_idx": "s_emb_idx", "sp_emb_idx": "sp_emb_idx", "action_idx": "a_idx",
                "state_description_idx": "s_id", "sp_id": "sp_id", "problem_idx": "problem_idx",
                "goal_distance_idx": "goal_distance_idx", "plan_id_idx": "plan_id_idx"}
        tri = tri[list(cols)].rename(columns=cols)
        return tri[tri["a_idx"] != -1].reset_index(drop=True)

    def _drop_duplicate_states(self):
        """Paraphrase collisions during data construction can give one (s,a) two
        outcomes; keep the first of each conflicting pair."""
        self.triplets.drop_duplicates(subset=["s_emb_idx", "sp_emb_idx"], inplace=True)
        self.triplets.drop_duplicates(subset=["s_emb_idx", "a_idx"], inplace=True)
        self.triplets.drop_duplicates(subset=["a_idx", "sp_emb_idx"], inplace=True)
        self.triplets.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        row = self.triplets.iloc[i]
        a_str = self.action_vocab[row["a_idx"]]
        return {
            "s_id": row["s_id"],
            "sp_id": row["sp_id"],
            "a_idx": row["a_idx"],
            "s_emb": torch.tensor(self.state_embs[row["s_emb_idx"]]),
            "a_emb": torch.tensor(self.action_embs[self.action_to_idx[a_str]]),
            "sp_emb": torch.tensor(self.state_embs[row["sp_emb_idx"]]),
            "sp_emb_idx": row["sp_emb_idx"],
            "a_str": a_str,
            "s_str": self.values["state_description"][row["s_id"]],
            "sp_str": self.values["state_description"][row["sp_id"]],
            "goal_distance": self.values["goal_distance"][row["goal_distance_idx"]],
            "problem_id": row["problem_idx"],
            "plan_id": row["plan_id_idx"],
        }

    def iter_problem_batches(self, *args, **kwargs):
        for idx_batch in ProblemGroupedBatchSampler(self, *args, **kwargs):
            yield [self[i] for i in idx_batch]


class ProblemGroupedBatchSampler(Sampler[List[int]]):
    """Every batch is drawn from a single problem, so in-batch negatives are
    same-problem near-misses rather than trivially separable states."""

    def __init__(self, dataset, batch_size, *, indices=None, drop_last=False,
                 shuffle_problems=True, shuffle_within_problem=False, seed=0):
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_problems = shuffle_problems
        self.shuffle_within_problem = shuffle_within_problem
        self.seed = int(seed)
        self._epoch = 0

        df = dataset.triplets
        if "problem_idx" not in df.columns:
            raise ValueError("dataset.triplets must include 'problem_idx'")
        self.indices = list(range(len(df)) if indices is None else indices)

        self._by_prob = defaultdict(list)
        probs = df.iloc[self.indices]["problem_idx"].to_numpy()
        for pos, idx in enumerate(self.indices):
            self._by_prob[int(probs[pos])].append(idx)
        self._problems = list(self._by_prob)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        problems = self._problems.copy()
        if self.shuffle_problems:
            rng.shuffle(problems)
        for pid in problems:
            idxs = self._by_prob[pid].copy()
            if self.shuffle_within_problem:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and batch):
                    yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(v) // self.batch_size for v in self._by_prob.values())
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self._by_prob.values())


# ----------------------------------------------------------------------------- splits

def grouped_split_by_problem(dataset, train_frac=0.9, seed=0, *, return_problem_sets=False):
    """Whole problems to one side or the other — the Extrapolation protocol."""
    probs = dataset.triplets["problem_idx"].to_numpy()
    unique = np.unique(probs)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n_train = int(round(len(unique) * train_frac))
    n_train = max(1, min(len(unique) - 1, n_train)) if len(unique) > 1 else 1
    train_set, valid_set = set(unique[:n_train]), set(unique[n_train:])

    train_idx = [i for i, p in enumerate(probs) if p in train_set]
    valid_idx = [i for i, p in enumerate(probs) if p in valid_set]
    assert train_set.isdisjoint(valid_set)
    return (train_idx, valid_idx, (train_set, valid_set)) if return_problem_sets else (train_idx, valid_idx, None)


def grouped_split_by_plan(dataset, train_frac=0.9, seed=0):
    """Whole plans held out — the Plan-Variant protocol."""
    plans = dataset.triplets["plan_id_idx"].to_numpy()
    unique = np.unique(plans)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(round(len(unique) * train_frac))
    n_train = max(1, min(len(unique) - 1, n_train)) if len(unique) > 1 else 1
    train_plans, valid_plans = set(unique[:n_train]), set(unique[n_train:])
    return ([i for i, p in enumerate(plans) if p in train_plans],
            [i for i, p in enumerate(plans) if p in valid_plans])


def leave_one_problem_out(dataset, test_problem: int, n_train_problems: Optional[int] = None, seed=0):
    """Train on `n_train_problems` problems (all others if None), test on exactly one.

    Used for the problem-count learning curve: holding the test problem fixed while
    varying how many problems the model trains on isolates data quantity from which
    particular problem happens to be held out.
    """
    probs = dataset.triplets["problem_idx"].to_numpy()
    unique = np.unique(probs)
    if test_problem not in unique:
        raise ValueError(f"problem {test_problem} not in domain (have {unique.tolist()})")

    pool = np.array([p for p in unique if p != test_problem])
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    chosen = set(pool.tolist() if n_train_problems is None else pool[:n_train_problems].tolist())

    train_idx = [i for i, p in enumerate(probs) if p in chosen]
    valid_idx = [i for i, p in enumerate(probs) if p == test_problem]
    return train_idx, valid_idx, sorted(chosen)


# ----------------------------------------------------------------------------- convenience

def load_domain(domain: str, model_name: str = DEFAULT_ENCODER, text_type: str = "original",
                verbose: bool = True) -> Tuple[FactorizedTripletDataset, pd.DataFrame]:
    """Dataset plus its triplet table with the integer goal distance attached."""
    ds = FactorizedTripletDataset(domain=domain, model_name=model_name,
                                  text_type=text_type, verbose=verbose)
    tri = ds.triplets.reset_index(drop=True)
    goal_vals = ds.values["goal_distance"]
    tri["goal_val"] = tri["goal_distance_idx"].apply(lambda i: int(goal_vals[i]))
    return ds, tri


def make_split(ds, tri, split: str, seed: int, train_frac: float = 0.8):
    """`random` = Interpolation, `problem_grouped` = Extrapolation."""
    if split == "random":
        perm = np.random.default_rng(seed).permutation(len(tri))
        cut = int(train_frac * len(tri))
        return perm[:cut].tolist(), perm[cut:].tolist()
    train_idx, valid_idx, _ = grouped_split_by_problem(ds, train_frac=train_frac, seed=seed)
    return train_idx, valid_idx


def build_trajectories(tri, valid_idx, max_trajs: int, seed: int, min_len: int = 2):
    """Reconstruct executable trajectories from held-out triplets.

    Within a plan, states are ordered by descending goal distance. A trajectory is
    only usable where consecutive triplets actually chain (sp_emb_idx[t] == s_emb_idx[t+1]);
    duplicate-dropping can break links, so we keep the longest contiguous run per plan.
    """
    sub = tri.iloc[sorted(set(valid_idx))]
    trajs = []
    for _, grp in sub.groupby("plan_id_idx"):
        g = grp.sort_values("goal_val", ascending=False)
        s, a, sp = (g[c].to_numpy() for c in ("s_emb_idx", "a_idx", "sp_emb_idx"))
        run, best = [0], []
        for t in range(1, len(g)):
            if sp[run[-1]] == s[t]:
                run.append(t)
            else:
                if len(run) > len(best):
                    best = run
                run = [t]
        if len(run) > len(best):
            best = run
        if len(best) >= min_len:
            trajs.append((s[best], a[best], sp[best]))

    trajs.sort(key=lambda x: -len(x[0]))
    if len(trajs) > max_trajs:
        sel = np.random.default_rng(seed).choice(len(trajs), max_trajs, replace=False)
        trajs = [trajs[i] for i in sel]
    return trajs
