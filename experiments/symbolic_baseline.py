"""Symbolic and lifted baselines — the comparisons the paper is missing.

Two arms, both nearly free and both CPU-only:

  symbolic       Induce lifted STRIPS operators from the training transitions and
                 apply them. This is the classical answer to "learn a transition
                 function from traces" (ARMS, LOCM, SAM learning, FAMA) and it
                 generalizes across problem instances by construction, because the
                 operators are lifted over their parameters.

  lifted_offset  s_hat' = E(s) + delta[schema(a)], the embedding-space analogue.
                 The published `offset` floor keys on the *grounded* action, which
                 under Extrapolation has no training displacement for 57.6% of ferry
                 and 65.5% of logistics test transitions. Zero test schemas are
                 unseen, so keying on the schema restores full coverage and separates
                 "the action was never seen" from "the embedding cannot represent
                 this action's effect as a translation".

`identity` and the grounded `offset` are carried through unchanged as reference points.

Scoring. The symbolic arm predicts a literal set, so it cannot be scored by cosine.
It gets two metrics instead:

  exact        predicted literal set == true next state. Directly comparable to the
               PRODUCE exact-match column of results/analysis/embedplan_vs_llm.md.
  matched_128  same problem-grouped batches as embedplan.evaluation.matched_pool_eval,
               with candidates ranked by Jaccard overlap against the prediction
               instead of cosine. Same queries, same pools, same tie-breaking
               (best-case, strictly-greater), so Hit@k is comparable to the
               embedding arms.

Usage:
    python -m experiments.symbolic_baseline --domains ferry logistics goldminer
    python -m experiments.symbolic_baseline --domains ferry --splits random
"""

import argparse
import json

import numpy as np
import torch

from embedplan import load_domain, make_split
from embedplan.baselines import IdentityTransition, LiftedOffsetTransition, OffsetTransition
from embedplan.config import ANALYSIS_DIR
from embedplan.data import ProblemGroupedBatchSampler
from embedplan.evaluation import matched_pool_eval, pool_sweep
from embedplan.scoring import project_pool
from embedplan.symbolic import LiftedActionModel, jaccard, schema_index, symbolic_frame
from embedplan.utils import resolve_device


def evaluate_embedding_arm(model, S, A, tri, ds, valid_idx, device, sizes, seed,
                           action_index=False, batch_size=128):
    pool = project_pool(model, S)
    out = pool_sweep(model, S, A, tri, valid_idx, pool, device, sizes, seed,
                     exclude_self=True, action_index=action_index)
    out["matched_128"] = matched_pool_eval(model, S, A, tri, ds, valid_idx, device,
                                           batch_size=batch_size, seed=seed,
                                           action_index=action_index)
    return out


def evaluate_symbolic(model, sf, ds, train_idx, valid_idx, batch_size, seed):
    """Exact-match plus Jaccard-ranked matched-pool Hit@k on the held-out triplets."""
    s_lits = sf["s_lits"].to_numpy()
    sp_lits = sf["sp_lits"].to_numpy()
    actions = sf["action"].to_numpy()

    exact = unseen_schema = 0
    preds = {}
    for i in valid_idx:
        p = model.predict(s_lits[i], actions[i])
        preds[i] = p
        if p is None:
            unseen_schema += 1
        elif p == sp_lits[i]:
            exact += 1
    n = max(1, len(valid_idx))

    # Same batches, queries and tie-breaking as matched_pool_eval, Jaccard instead of cosine.
    sampler = ProblemGroupedBatchSampler(ds, batch_size, indices=valid_idx,
                                         shuffle_problems=False,
                                         shuffle_within_problem=False, seed=seed)
    hits = {1: 0, 5: 0, 10: 0}
    total, pool_sizes = 0, []
    for batch in sampler:
        if len(batch) < 2:
            continue
        cands = [sp_lits[j] for j in batch]
        for pos, i in enumerate(batch):
            p = preds.get(i)
            # An unseen schema is a real failure, not an excluded row: it scores as a
            # miss at every k rather than being dropped from the denominator.
            if p is not None:
                scores = np.fromiter((jaccard(p, c) for c in cands), dtype=float,
                                     count=len(cands))
                rank = int((scores > scores[pos]).sum()) + 1
                for k in hits:
                    hits[k] += rank <= k
            total += 1
        pool_sizes.append(len(batch))

    return {
        "exact": exact / n,
        "n_test": int(len(valid_idx)),
        "unseen_schema_frac": unseen_schema / n,
        "matched_128": {**{f"hit@{k}": hits[k] / max(1, total) for k in hits},
                        "n_queries": total,
                        "mean_pool": float(np.mean(pool_sizes)) if pool_sizes else 0.0},
        "induction": model.summary(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=["ferry", "logistics", "goldminer"])
    ap.add_argument("--splits", nargs="+", default=["problem_grouped", "random"])
    ap.add_argument("--model_name", default="BAAI/bge-m3",
                    help="only selects which cached embedding table the embedding arms use")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, -1])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    everything = {}

    for domain in args.domains:
        ds, tri = load_domain(domain, args.model_name, verbose=False)
        sf = symbolic_frame(ds, tri)
        coverage = float(sf["has_symbolic"].mean())
        S = torch.as_tensor(ds.state_embs, dtype=torch.float32, device=device)
        A = torch.as_tensor(ds.action_embs, dtype=torch.float32, device=device)
        sch_map, sch_names = schema_index(ds.action_vocab)

        print(f"\n### {domain}: {len(tri)} triplets, {len(ds.action_vocab)} grounded "
              f"actions, {len(sch_names)} schemas {sch_names}", flush=True)
        if coverage < 1.0:
            print(f"  WARNING symbolic coverage {coverage:.4f} < 1.0 — "
                  f"{(1 - coverage) * len(sf):.0f} triplets have no literal set", flush=True)

        for split in args.splits:
            for seed in args.seeds:
                tag = f"{domain}_{split}_seed{seed}"
                train_idx, valid_idx = make_split(ds, tri, split, seed)
                a_np = tri["a_idx"].to_numpy()
                seen_grounded = set(a_np[train_idx].tolist())
                unseen_grounded = float(np.mean([x not in seen_grounded for x in a_np[valid_idx]]))

                s_tr = torch.as_tensor(tri["s_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
                a_tr = torch.as_tensor(a_np[train_idx], device=device, dtype=torch.long)
                p_tr = torch.as_tensor(tri["sp_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)

                res = {"unseen_grounded_action_frac": unseen_grounded,
                       "n_train": int(len(train_idx)), "n_test": int(len(valid_idx)),
                       "symbolic_coverage": coverage}

                print(f"\n  == {tag}  train {len(train_idx)}  test {len(valid_idx)}  "
                      f"unseen grounded actions in test {unseen_grounded:.3f} ==", flush=True)

                print("    symbolic (lifted STRIPS induction) ...", flush=True)
                sym = LiftedActionModel().fit(
                    [(sf["s_lits"].iat[i], sf["action"].iat[i], sf["sp_lits"].iat[i])
                     for i in train_idx])
                res["symbolic"] = evaluate_symbolic(sym, sf, ds, train_idx, valid_idx,
                                                    args.batch_size, seed)
                s = res["symbolic"]
                print(f"      exact={s['exact']:.4f}  matched_128 hit@1={s['matched_128']['hit@1']:.4f} "
                      f"hit@5={s['matched_128']['hit@5']:.4f}  schemas={s['induction']['n_schemas']} "
                      f"conflicts={s['induction']['conflicts']}  "
                      f"unseen_schema={s['unseen_schema_frac']:.4f}", flush=True)

                print("    identity ...", flush=True)
                res["identity"] = evaluate_embedding_arm(
                    IdentityTransition().to(device), S, A, tri, ds, valid_idx, device,
                    args.sizes, seed, batch_size=args.batch_size)

                print("    offset (grounded) ...", flush=True)
                off = OffsetTransition(A.shape[0], S.shape[1], device).fit(S, a_tr, s_tr, p_tr)
                res["offset_grounded"] = evaluate_embedding_arm(
                    off, S, A, tri, ds, valid_idx, device, args.sizes, seed,
                    action_index=True, batch_size=args.batch_size)
                res["offset_grounded"]["n_actions_seen_in_train"] = int(off.seen.sum().item())

                print("    offset (lifted, per schema) ...", flush=True)
                lof = LiftedOffsetTransition(sch_map, len(sch_names), S.shape[1], device)
                lof.fit(S, a_tr, s_tr, p_tr)
                res["offset_lifted"] = evaluate_embedding_arm(
                    lof, S, A, tri, ds, valid_idx, device, args.sizes, seed,
                    action_index=True, batch_size=args.batch_size)
                res["offset_lifted"]["n_schemas_seen_in_train"] = int(lof.seen.sum().item())

                everything[tag] = res
                _print_table(tag, res)

    out = ANALYSIS_DIR / (args.out or "symbolic_baseline.json")
    out.write_text(json.dumps(everything, indent=2, default=str))
    print(f"\nwrote {out}")


def _print_table(tag, res):
    print(f"\n  {tag}")
    print(f"  {'method':18s} {'m128 h@1':>10s} {'m128 h@5':>10s} {'full h@1':>10s} {'exact':>10s}")
    rows = [("symbolic", res["symbolic"]["matched_128"], None, res["symbolic"]["exact"]),
            ("identity", res["identity"]["matched_128"], res["identity"].get("full"), None),
            ("offset_grounded", res["offset_grounded"]["matched_128"],
             res["offset_grounded"].get("full"), None),
            ("offset_lifted", res["offset_lifted"]["matched_128"],
             res["offset_lifted"].get("full"), None)]
    for name, m, full, exact in rows:
        f1 = f"{full['hit@1']:10.4f}" if full else f"{'--':>10s}"
        ex = f"{exact:10.4f}" if exact is not None else f"{'--':>10s}"
        print(f"  {name:18s} {m['hit@1']:10.4f} {m['hit@5']:10.4f} {f1} {ex}")


if __name__ == "__main__":
    main()
