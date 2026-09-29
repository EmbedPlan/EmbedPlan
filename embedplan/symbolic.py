"""Lifted STRIPS action-model induction — the classical baseline for this task.

"Learn a transition function from observed transitions" is a solved problem in the
planning literature (ARMS, LOCM, SAM learning, FAMA): induce lifted operators from
traces, then apply them. This module implements the simplest correct version so the
embedding-space results have that comparison next to them.

Why it is the sharp baseline here: under `problem_grouped` a held-out problem
introduces new *objects*, so most test transitions use a grounded action the training
split never saw — measured 57.6% on ferry, 65.5% on logistics. But the number of
action *schemas* is tiny (3-6) and **zero** test schemas are unseen. A representation
keyed on grounded actions therefore cannot generalize, while a lifted one generalizes
by construction. That distinction, not the split difficulty, is what the
extrapolation gap is measuring.

The induction: for a transition (s, a, s') with a = (schema o_1..o_n), the effects are
add = s' \\ s and del = s \\ s'. Replace each occurrence of o_i with the positional
marker ?p_i to lift, and substitute back to ground. Under STRIPS with full
observability and no conditional effects — which all of these ACPBench domains
satisfy — one transition per schema suffices, and every further transition is a
consistency check.
"""

import ast
import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import pandas as pd

_PARAM = re.compile(r"\?p(\d+)$")


def parse_literals(raw) -> FrozenSet[str]:
    """`"['(at c0 l0)', '(on c2)']"` -> frozenset of literal strings."""
    try:
        v = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return frozenset()
    return frozenset(v) if isinstance(v, (list, tuple, set, frozenset)) else frozenset()


def action_parts(action: str) -> Tuple[str, List[str]]:
    """`'(board c1 l1)'` -> `('board', ['c1', 'l1'])`."""
    toks = action.strip("() \t\n").split()
    return (toks[0], toks[1:]) if toks else ("", [])


def _rebuild(name: str, args: Sequence[str]) -> str:
    return f"({name} {' '.join(args)})" if args else f"({name})"


def lift_literal(literal: str, params: Sequence[str]) -> str:
    """Replace action arguments by position. Non-argument constants stay literal —
    that is what makes an over-general operator detectable as a conflict instead of
    silently fitting."""
    name, args = action_parts(literal)
    return _rebuild(name, [f"?p{params.index(a)}" if a in params else a for a in args])


def ground_literal(lifted: str, params: Sequence[str]) -> str:
    name, args = action_parts(lifted)
    out = []
    for a in args:
        m = _PARAM.match(a)
        out.append(params[int(m.group(1))] if m and int(m.group(1)) < len(params) else a)
    return _rebuild(name, out)


class LiftedActionModel:
    """Induced add/delete effects per action schema.

    `conflicts` counts training transitions whose lifted effects disagreed with the
    first pair seen for that schema. A nonzero count means the STRIPS assumption is
    violated (conditional effects, partial observability, or noisy states) and the
    predictions below are not trustworthy — it is reported, never silently ignored.
    """

    def __init__(self):
        self.effects: Dict[str, Dict[str, FrozenSet[str]]] = {}
        self.conflicts = 0
        self.n_fit = 0

    def fit(self, transitions: Sequence[Tuple[FrozenSet[str], str, FrozenSet[str]]]):
        for s, action, sp in transitions:
            schema, params = action_parts(action)
            if not schema:
                continue
            add = frozenset(lift_literal(l, params) for l in (sp - s))
            dele = frozenset(lift_literal(l, params) for l in (s - sp))
            self.n_fit += 1
            cur = self.effects.get(schema)
            if cur is None:
                self.effects[schema] = {"add": add, "del": dele}
            elif cur["add"] != add or cur["del"] != dele:
                self.conflicts += 1
        return self

    def predict(self, state: FrozenSet[str], action: str) -> Optional[FrozenSet[str]]:
        """None when the schema was never observed — the honest failure mode, and
        the one that would dominate in a cross-domain setting."""
        schema, params = action_parts(action)
        eff = self.effects.get(schema)
        if eff is None:
            return None
        return (state - {ground_literal(l, params) for l in eff["del"]}) \
            | {ground_literal(l, params) for l in eff["add"]}

    def summary(self) -> Dict:
        return {"n_schemas": len(self.effects), "n_fit_transitions": self.n_fit,
                "conflicts": self.conflicts,
                "effects": {k: {"add": sorted(v["add"]), "del": sorted(v["del"])}
                            for k, v in sorted(self.effects.items())}}


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def attach_symbolic_states(ds) -> Dict[Tuple[int, int, int], Tuple[FrozenSet[str], FrozenSet[str]]]:
    """Map each (plan_id_idx, s_id, sp_id) triplet key to its (s, s') literal sets.

    `ds.triplets` carries `state_description_idx` as s_id/sp_id, not the symbolic
    `state_idx`, and the description->symbolic map is many-to-many (one NL rendering
    can correspond to several literal sets, because the descriptions omit static
    predicates that differ across problems). So the symbolic side has to be rebuilt
    along the same plan ordering `_build_triplets` uses, and keyed on the triplet
    identity rather than joined on the description.
    """
    df, values = ds.df, ds.values
    gd = values["goal_distance"]
    if "goal_val" not in df.columns:
        df = df.copy()
        df["goal_val"] = df["goal_distance_idx"].apply(lambda i: int(gd[i]))

    cache: Dict[int, FrozenSet[str]] = {}

    def lits(state_idx: int) -> FrozenSet[str]:
        if state_idx not in cache:
            cache[state_idx] = parse_literals(values["state"][state_idx])
        return cache[state_idx]

    out: Dict[Tuple[int, int, int], Tuple[FrozenSet[str], FrozenSet[str]]] = {}
    for plan_id, group in df.groupby("plan_id_idx"):
        g = group.sort_values("goal_val", ascending=False)
        s_desc = g["state_description_idx"].tolist()
        s_sym = g["state_idx"].tolist()
        for t in range(len(g) - 1):
            key = (int(plan_id), int(s_desc[t]), int(s_desc[t + 1]))
            out.setdefault(key, (lits(int(s_sym[t])), lits(int(s_sym[t + 1]))))
    return out


def symbolic_frame(ds, tri: pd.DataFrame) -> pd.DataFrame:
    """`tri` plus `s_lits` / `sp_lits` / `action` columns, and a `has_symbolic` flag.

    Rows whose triplet key has no symbolic counterpart are marked rather than
    dropped, so the driver can report coverage instead of quietly evaluating on a
    subset that differs from the embedding arms'.
    """
    lookup = attach_symbolic_states(ds)
    keys = list(zip(tri["plan_id_idx"].astype(int), tri["s_id"].astype(int),
                    tri["sp_id"].astype(int)))
    pairs = [lookup.get(k) for k in keys]
    out = tri.copy()
    out["has_symbolic"] = [p is not None for p in pairs]
    out["s_lits"] = [p[0] if p else frozenset() for p in pairs]
    out["sp_lits"] = [p[1] if p else frozenset() for p in pairs]
    out["action"] = [ds.action_vocab[i] for i in tri["a_idx"].to_numpy()]
    return out


def schema_index(action_vocab: Sequence[str]) -> Tuple[List[int], List[str]]:
    """Grounded-action id -> schema id, plus the schema names.

    This is what turns the existing per-grounded-action offset baseline into a lifted
    one: ferry has 487 grounded actions but 3 schemas, so the lifted table has 3 rows
    and every test action hits a populated row.
    """
    names: List[str] = []
    by_name: Dict[str, int] = {}
    mapping: List[int] = []
    for a in action_vocab:
        schema, _ = action_parts(a)
        if schema not in by_name:
            by_name[schema] = len(names)
            names.append(schema)
        mapping.append(by_name[schema])
    return mapping, names
