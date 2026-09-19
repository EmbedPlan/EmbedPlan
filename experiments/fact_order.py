"""Is the encoder more sensitive to fact ORDER than to the action itself?

State descriptions are a rendered clause list — "Car c0 is at location l0, Car c1 is at
location l1, and The ferry is at l1". The underlying state is a *set* of literals, so
that ordering is arbitrary: a semantically null nuisance variable. A representation fit
for transition modelling should be invariant to it.

The comparison that matters is against how far a real action moves the embedding, which
is very little — cos(E(s), E(s')) is 0.9966 on ferry and 0.9994 on logistics. So define

    d_order  = 1 - cos(E(s), E(shuffle(s)))     movement from reordering, meaning nothing
    d_action = 1 - cos(E(s), E(s'))             movement from an actual transition
    ratio    = d_order / d_action

ratio > 1 means the encoder responds more strongly to shuffling the sentence order than
to the state actually changing, which would make the nuisance variable the dominant
signal and explain why order-invariant representations (bag-of-literals, TF-IDF) beat
the encoder on extrapolation.

Usage:
    python -m experiments.fact_order --domains ferry logistics goldminer
    python -m experiments.fact_order --domains ferry --n_states 64 --n_shuffles 4
"""

import argparse
import json
import random
import re

import numpy as np
import torch

from embedplan import load_domain, make_split
from embedplan.config import ANALYSIS_DIR
from embedplan.finetune import TextBank, TunableEncoder
from embedplan.utils import resolve_device

STATE_HDR = "### CURRENT STATE ###"


def split_clauses(state_text: str):
    """Clause list from a rendered state description. Returns None if the shape is not
    the expected comma list, so callers can skip rather than silently mangle the text."""
    parts = [p.strip() for p in state_text.split(",")]
    parts = [re.sub(r"^and\s+", "", p, flags=re.I).strip() for p in parts if p.strip()]
    return parts if len(parts) >= 3 else None


def reorder_prompt(prompt: str, rng) -> str:
    """Shuffle only the clauses inside the CURRENT STATE block; leave problem and goal
    text untouched so the intervention is exactly the nuisance variable and nothing else."""
    if STATE_HDR not in prompt:
        return None
    head, rest = prompt.split(STATE_HDR, 1)
    lines = rest.split("\n\n", 1)
    state_block, tail = (lines[0], "\n\n" + lines[1] if len(lines) > 1 else "")
    clauses = split_clauses(state_block)
    if clauses is None:
        return None
    shuffled = clauses[:]
    for _ in range(8):                      # avoid returning the identity permutation
        rng.shuffle(shuffled)
        if shuffled != clauses:
            break
    body = ", ".join(shuffled[:-1]) + ", and " + shuffled[-1] if len(shuffled) > 1 else shuffled[0]
    return f"{head}{STATE_HDR}\n{body}{tail}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=["ferry", "logistics", "goldminer"])
    ap.add_argument("--split", default="problem_grouped")
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--n_states", type=int, default=256, help="transitions sampled per domain")
    ap.add_argument("--n_shuffles", type=int, default=4, help="reorderings per state")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fact_order.json")
    args = ap.parse_args()

    device = resolve_device()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    enc = TunableEncoder(args.encoder, lora_rank=0, max_length=args.max_length,
                         device=device, gradient_checkpointing=False)
    out = {}

    for domain in args.domains:
        rng = random.Random(args.seed)
        ds, tri = load_domain(domain, args.encoder, verbose=False)
        bank = TextBank(ds)
        _, valid_idx = make_split(ds, tri, args.split, args.seed)
        sel = np.random.default_rng(args.seed).choice(
            valid_idx, min(args.n_states, len(valid_idx)), replace=False)

        s_rows = tri["s_emb_idx"].to_numpy()
        sp_rows = tri["sp_emb_idx"].to_numpy()

        originals, nexts, shuffles, owner = [], [], [], []
        for i in sel:
            p = bank.state_texts[int(s_rows[i])]
            variants = [reorder_prompt(p, rng) for _ in range(args.n_shuffles)]
            variants = [v for v in variants if v]
            if not variants:
                continue
            owner.append(len(originals))
            originals.append(p)
            nexts.append(bank.state_texts[int(sp_rows[i])])
            shuffles.append(variants)

        if not originals:
            print(f"{domain}: no parseable state blocks, skipping", flush=True)
            continue

        flat = [v for vs in shuffles for v in vs]
        E_s = enc.encode_all(originals, args.batch_size, desc=f"{domain} s")
        E_sp = enc.encode_all(nexts, args.batch_size, desc=f"{domain} s'")
        E_sh = enc.encode_all(flat, args.batch_size, desc=f"{domain} shuffled")

        n = torch.nn.functional.normalize
        E_s, E_sp, E_sh = n(E_s, dim=-1), n(E_sp, dim=-1), n(E_sh, dim=-1)

        d_action = (1 - (E_s * E_sp).sum(-1)).cpu().numpy()
        d_order, k = [], 0
        for idx, vs in enumerate(shuffles):
            block = E_sh[k:k + len(vs)]; k += len(vs)
            d_order.extend((1 - (block @ E_s[idx])).cpu().numpy().tolist())
        d_order = np.array(d_order)

        rec = {"n_states": len(originals), "n_shuffles_total": len(flat),
               "d_action_mean": float(d_action.mean()), "d_action_sd": float(d_action.std()),
               "d_order_mean": float(d_order.mean()), "d_order_sd": float(d_order.std()),
               "ratio": float(d_order.mean() / max(d_action.mean(), 1e-12)),
               "cos_action": float(1 - d_action.mean()), "cos_order": float(1 - d_order.mean())}
        out[domain] = rec
        print(f"{domain:11s} n={rec['n_states']:4d}  "
              f"cos(s,s')={rec['cos_action']:.4f} (d={rec['d_action_mean']:.5f})  "
              f"cos(s,shuffled s)={rec['cos_order']:.4f} (d={rec['d_order_mean']:.5f})  "
              f"ratio={rec['ratio']:.2f}x", flush=True)

    path = ANALYSIS_DIR / args.out
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    if out:
        r = np.mean([v["ratio"] for v in out.values()])
        print(f"mean ratio {r:.2f}x — "
              + ("reordering moves the embedding MORE than the action does"
                 if r > 1 else "the action dominates reordering"))


if __name__ == "__main__":
    main()
