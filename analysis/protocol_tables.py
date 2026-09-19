"""Tables from the paper-protocol sweep, in text or paper-ready LaTeX.

Merges results/analysis/protocol_parts/*.json (written by experiments.baseline_sweep,
one file per domain-split-seed) and reports mean +- sd over seeds.

Every arm here shares one estimator -- the published protocol reconstructed in
embedplan.paper_protocol: worst-case tie-breaking, split-dependent pool construction
(same-problem distractors under Extrapolation, uniform under Interpolation), and
best-epoch selection. Numbers are therefore comparable across arms, and are NOT
comparable to results/analysis/baselines.md, which used best-case ties and final-epoch.

Caveat carried into the output: under `problem_grouped`, seeds 0 and 2 select the same
held-out problems, so a 3-seed sd is computed over 2 distinct splits. `--split-variance`
decomposes it.

Usage:
    python -m analysis.protocol_tables
    python -m analysis.protocol_tables --latex --metric hit@5
    python -m analysis.protocol_tables --split-variance
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from embedplan.config import ANALYSIS_DIR

PARTS = ANALYSIS_DIR / "protocol_parts"
LABEL = {
    "symbolic": "Lifted STRIPS induction", "char": "Char 3-5 grams",
    "bow": "Bag-of-words", "literals": "Bag-of-literals", "tfidf": "TF-IDF bigrams",
    "embed_BAAI/bge-m3": "EmbedPlan (BGE-M3, frozen)", "offset_lifted": "Offset, lifted",
    "offset_grounded": "Offset, grounded", "identity": "Identity",
    "random": "Random vectors [null]", "context": "Context-only [null]",
}
SPLIT_NAME = {"problem_grouped": "Extrapolation", "random": "Interpolation"}


def load():
    """(arm, domain, split) -> {seed: {hit@1, hit@5, hit@10}}"""
    out = defaultdict(dict)
    for f in sorted(glob.glob(str(PARTS / "*.json"))):
        for key, rec in json.loads(open(f).read()).items():
            domain, split, seed, arm = key.split("|")
            if arm == "symbolic":
                m = {k: rec["exact"] for k in ("hit@1", "hit@5", "hit@10")}
            else:
                m = {k: rec["best"][k] for k in ("hit@1", "hit@5", "hit@10")}
            out[(arm, domain, split)][int(seed)] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="hit@5", choices=["hit@1", "hit@5", "hit@10"])
    ap.add_argument("--domains", nargs="+", default=["ferry", "logistics", "goldminer"])
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--split-variance", action="store_true",
                    help="separate training noise (seeds 0/2, same split) from split spread")
    args = ap.parse_args()

    data = load()
    arms = sorted({a for a, _, _ in data}, key=lambda a: list(LABEL).index(a) if a in LABEL else 99)

    for split in ("problem_grouped", "random"):
        rows = []
        for arm in arms:
            per = {}
            for d in args.domains:
                cell = data.get((arm, d, split), {})
                if len(cell) == 3:
                    per[d] = np.array([cell[s][args.metric] for s in (0, 1, 2)])
            if len(per) == len(args.domains):
                rows.append((arm, per, float(np.mean([v.mean() for v in per.values()]))))
        if not rows:
            continue
        rows.sort(key=lambda r: -r[2])

        title = f"{SPLIT_NAME[split]} — {args.metric} (%), paper protocol, 3 seeds"
        if args.latex:
            print(f"\n% {title}")
            print("\\begin{tabular}{@{}l" + "c" * (len(args.domains) + 1) + "@{}}\n\\toprule")
            print("\\textbf{Method} & " + " & ".join(f"\\textbf{{{d.capitalize()}}}" for d in args.domains)
                  + " & \\textbf{Mean} \\\\\n\\midrule")
            for arm, per, mean in rows:
                cells = " & ".join(f"${per[d].mean()*100:.1f} \\pm {per[d].std(ddof=1)*100:.1f}$"
                                   for d in args.domains)
                print(f"{LABEL.get(arm, arm)} & {cells} & ${mean*100:.1f}$ \\\\")
            print("\\bottomrule\n\\end{tabular}")
        else:
            print(f"\n=== {title} ===")
            print(f"{'arm':28s}" + "".join(f"{d:>18s}" for d in args.domains) + f"{'mean':>9s}")
            for arm, per, mean in rows:
                cells = "".join(f"{per[d].mean()*100:11.1f}±{per[d].std(ddof=1)*100:<6.1f}"
                                for d in args.domains)
                print(f"{LABEL.get(arm, arm):28s}{cells}{mean*100:9.1f}")

        if args.split_variance and split == "problem_grouped":
            print("\n  variance decomposition (seeds 0 and 2 share a split):")
            print(f"  {'arm':28s}{'train noise |s0-s2|':>21s}{'split spread':>15s}")
            for arm, per, _ in rows:
                tn = np.mean([abs(v[0] - v[2]) for v in per.values()])
                ss = np.mean([abs(np.mean([v[0], v[2]]) - v[1]) for v in per.values()])
                print(f"  {LABEL.get(arm, arm):28s}{tn*100:21.2f}{ss*100:15.2f}")


if __name__ == "__main__":
    main()
