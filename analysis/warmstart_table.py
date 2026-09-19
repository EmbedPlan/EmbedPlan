"""One-stage vs two-stage encoder fine-tuning.

Both arms end identically — LoRA-adapted encoder, pool re-encoded with it, then a
head trained from scratch for `head_epochs` on the resulting space. The only
difference is the head's state when the encoder is unfrozen:

    one-stage (arm lora16)          warmup_head_epochs=0
        The joint stage optimizes the encoder against a head that is nowhere near
        converged (joint-stage loss bottoms out around 2.05), so the adapter is
        shaped to compensate for a weak transition function that is then thrown away.

    two-stage (arm lora16warm400)   warmup_head_epochs=400
        The head is first trained to convergence in the unadapted space. LoRA is a
        zero delta at init, so that space is bit-identical to the frozen encoder's
        and the warmed head transfers exactly. The encoder's gradient then comes
        from a transition model that already knows how to move.

The frozen arm (lora_rank 0) is carried through as the no-fine-tuning reference.

CAUTION on seeds. Under `problem_grouped`, seeds 0 and 2 select the *identical* held-out
problems in ferry, logistics and goldminer (see docs/experiment-status.md), because
grouped_split_by_problem picks among only ~7 partitions. So a seed-0/seed-2 pair differs
by head initialization and batch order alone, not by data, while seed 1 is a genuinely
different split. Treating seeds 0-2 as three independent samples overstates agreement;
the summary below therefore separates within-split from between-split spread instead of
reporting one pooled standard deviation.

Usage:
    python -m analysis.warmstart_table
    python -m analysis.warmstart_table --metric full_hit@1 --split random
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats

from embedplan.config import RESULTS_ROOT

FT_DIR = RESULTS_ROOT / "finetune"

ARMS = {"frozen": "frozen", "one-stage": "lora16", "two-stage": "lora16warm400"}

METRICS = {
    "matched_hit@5": lambda d: d["matched_128"]["hit@5"],
    "matched_hit@1": lambda d: d["matched_128"]["hit@1"],
    "full_hit@1": lambda d: d["pool_sweep"]["full"]["hit@1"],
    "full_hit@5": lambda d: d["pool_sweep"]["full"]["hit@5"],
    "rollout_tf_step@1": lambda d: d.get("rollout", {}).get("teacher_forced", {}).get("step_hit@1"),
    "rollout_cl_step@1": lambda d: d.get("rollout", {}).get("closed_loop", {}).get("step_hit@1"),
}


def load(domain, arm, seed, split, encoder):
    p = FT_DIR / f"{domain}_{split}_{encoder}_{arm}_seed{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def fmt(v, width=9):
    return f"{v:{width}.3f}" if isinstance(v, (int, float)) else f"{'--':>{width}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=["logistics", "ferry", "goldminer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="problem_grouped")
    ap.add_argument("--encoder", default="bge-m3")
    ap.add_argument("--metric", default="matched_hit@5", choices=list(METRICS))
    args = ap.parse_args()

    get = METRICS[args.metric]
    print(f"# One-stage vs two-stage encoder fine-tuning — {args.metric}")
    print(f"# split={args.split}  encoder={args.encoder}\n")

    hdr = f"{'domain':13s} {'seed':>4s} {'frozen':>9s} {'one-stage':>9s} {'two-stage':>9s} {'2st-1st':>9s}"
    print(hdr)
    print("-" * len(hdr))

    pairs, by_domain = [], defaultdict(list)
    for domain in args.domains:
        for seed in args.seeds:
            vals = {}
            for label, arm in ARMS.items():
                d = load(domain, arm, seed, args.split, args.encoder)
                vals[label] = get(d) if d else None
            one, two = vals["one-stage"], vals["two-stage"]
            delta = two - one if (one is not None and two is not None) else None
            if delta is not None:
                pairs.append((domain, seed, one, two))
                by_domain[domain].append(delta)
            print(f"{domain:13s} {seed:>4d} {fmt(vals['frozen'])} {fmt(one)} {fmt(two)} {fmt(delta)}")

    print()
    if not pairs:
        print("no paired cells yet — nothing to test")
        return

    print(f"{'domain':13s} {'n':>4s} {'mean 2st-1st':>13s}")
    print("-" * 32)
    for domain, deltas in by_domain.items():
        print(f"{domain:13s} {len(deltas):>4d} {np.mean(deltas):>13.3f}")

    d = np.array([two - one for _, _, one, two in pairs])
    print(f"\npaired cells n={len(d)}")
    print(f"mean delta (two-stage - one-stage) = {d.mean():+.4f}   sd = {d.std(ddof=1):.4f}")
    print(f"per-cell range = [{d.min():+.3f}, {d.max():+.3f}]   sign: "
          f"{(d > 0).sum()} up / {(d < 0).sum()} down")
    if len(d) > 1:
        t, p = stats.ttest_rel([two for _, _, _, two in pairs], [one for _, _, one, _ in pairs])
        cohen = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
        print(f"paired t-test: t = {t:.3f}, p = {p:.3f}, Cohen's dz = {cohen:.3f}")
        verdict = "no detectable difference" if p >= 0.05 else (
            "two-stage better" if d.mean() > 0 else "one-stage better")
        print(f"verdict at alpha=0.05: {verdict}")

    # Noise inside a single arm bounds what any paired comparison can resolve. Split it
    # into the part that is pure training noise (seeds 0 and 2 share a problem_grouped
    # split) and the part that is split choice, because they are wildly different sizes
    # and only the first is what a "seed average" is usually assumed to smooth over.
    same_split = (0, 2) if args.split == "problem_grouped" else None
    print("\nwithin-arm spread (same arm, same domain):")
    for label, arm in ARMS.items():
        train_noise, split_spread = [], []
        for domain in args.domains:
            vals = {s: get(x) for s in args.seeds
                    for x in [load(domain, arm, s, args.split, args.encoder)] if x}
            vals = {s: v for s, v in vals.items() if v is not None}
            if same_split and all(s in vals for s in same_split):
                train_noise.append(abs(vals[same_split[0]] - vals[same_split[1]]))
            if len(vals) > 1:
                split_spread.append(max(vals.values()) - min(vals.values()))
        parts = []
        if train_noise:
            parts.append(f"training noise (seeds {same_split[0]}/{same_split[1]}, "
                         f"same split) = {np.mean(train_noise):.3f}")
        if split_spread:
            parts.append(f"total max-min across seeds = {np.mean(split_spread):.3f}")
        if parts:
            print(f"  {label:10s} " + "; ".join(parts))


if __name__ == "__main__":
    main()
