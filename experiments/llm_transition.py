"""LLM baselines for single-step transition prediction.

Protocol (unchanged from the submission so the numbers stay comparable): the model
is shown a state and an action in natural language and generates the successor
state as text. That generation is then ranked against the same problem-grouped
candidate pool EmbedPlan retrieves from, by normalized character distance.

Three things this fixes relative to experiments/llm_ranking.py:

1. Split. The old harness built pools from the *entire* dataset (no `indices`
   passed to the sampler), while EmbedPlan is scored on its held-out test split.
   The two were never evaluated on the same transitions. Here the split is
   explicit and the LLM sees exactly the queries EmbedPlan is tested on.
2. Coverage. Any domain, and both Interpolation and Extrapolation. The submission
   only ever ran two domains on one split, which is the "only two domains"
   criticism.
3. A floor. `identity` ranks the pool by distance to the *current* state — the
   text-space version of "predict nothing changed." Without it there is no way to
   tell whether an LLM's Hit@5 reflects transition reasoning or the fact that
   successor states are nearly identical to their predecessors.

`exact_match` is also recorded: does the normalized generation equal the true
successor outright? That one needs no pool at all.

The API key is read from OPENROUTER_API_KEY and never written to disk.

Usage:
    export OPENROUTER_API_KEY=...
    python -m experiments.llm_transition --domains ferry logistics \\
        --splits random problem_grouped --models openai/gpt-5.4 --max_samples 100
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from embedplan import load_domain, make_split
from embedplan.config import RESULTS_ROOT
from embedplan.data import ProblemGroupedBatchSampler

API_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT_DIR = RESULTS_ROOT / "llm_transition"
POOL_SIZE = 128


# ----------------------------------------------------------------------------- text

def normalize_state(s: Optional[str]) -> str:
    """Canonicalize a state description: order-invariant, punctuation-insensitive.

    States are conjunctions of facts whose order carries no meaning, so both the
    generation and the pool are reduced to a sorted fact list before comparison.

    Tolerates None: some providers return `content: null` on a refusal or a
    reasoning-only turn, and that must score as an empty generation rather than
    crash a sweep that is hours into its API budget.
    """
    if not s:
        return ""
    s = s.strip().lower().replace(" and ", ", ")
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s*,\s*", ", ", s).strip(", ")
    return ", ".join(sorted(f.strip() for f in s.split(", ") if f.strip()))


def char_distance(a: str, b: str) -> int:
    n = max(len(a), len(b))
    return sum(x != y for x, y in zip(a.ljust(n), b.ljust(n)))


def rank_of_truth(pred: str, pool: List[str], correct_idx: int) -> int:
    """1-indexed rank of the true successor when the pool is sorted by distance."""
    order = sorted(range(len(pool)), key=lambda i: char_distance(pred, pool[i]))
    return order.index(correct_idx) + 1


PROMPT = """Predict the state after executing an action.

### PROBLEM ###
{problem}

### CURRENT STATE ###
{state}

### ACTION ###
{action}

Output only the resulting state, in the same format and style as the current \
state. Do not explain, do not add commentary.

### NEXT STATE ###"""


# ----------------------------------------------------------------------------- data

def build_queries(domain: str, split: str, seed: int, max_samples: int,
                  encoder: str) -> Tuple[List[Dict], Dict]:
    """Problem-grouped candidate pools over the held-out split, as EmbedPlan sees them."""
    ds, tri = load_domain(domain, encoder, verbose=False)
    _, valid_idx = make_split(ds, tri, split, seed)
    sampler = ProblemGroupedBatchSampler(ds, POOL_SIZE, indices=valid_idx,
                                         shuffle_problems=False,
                                         shuffle_within_problem=False, seed=seed)

    batches = [b for b in sampler if len(b) >= 2]
    flat = [(bi, pos) for bi, b in enumerate(batches) for pos in range(len(b))]
    rng = random.Random(seed)
    if len(flat) > max_samples:
        flat = rng.sample(flat, max_samples)

    queries = []
    for bi, pos in flat:
        batch = batches[bi]
        pool = [normalize_state(ds.values["state_description"][tri["sp_id"].to_numpy()[i]])
                for i in batch]
        row = batch[pos]
        queries.append({
            "triplet_idx": int(row),
            "problem": ds.values["problem"][tri["problem_idx"].to_numpy()[row]],
            "state": normalize_state(ds.values["state_description"][tri["s_id"].to_numpy()[row]]),
            "next_state": pool[pos],
            "action": ds.action_vocab[tri["a_idx"].to_numpy()[row]],
            "pool": pool,
            "correct_idx": pos,
        })
    meta = {"domain": domain, "split": split, "n_valid": len(valid_idx),
            "n_batches": len(batches), "n_queries": len(queries),
            "mean_pool": float(np.mean([len(q["pool"]) for q in queries])) if queries else 0.0}
    return queries, meta


# ----------------------------------------------------------------------------- api

def call_model(prompt: str, model: str, api_key: str, timeout: int = 180,
               retries: int = 3) -> Tuple[str, Dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1200, "temperature": 0.0}
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {})
            return (data["choices"][0]["message"].get("content") or ""), {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)}
        except Exception as e:  # network flake, rate limit, provider hiccup
            last = str(e)
            time.sleep(2 * (attempt + 1))
    return "", {"error": last, "prompt_tokens": 0, "completion_tokens": 0}


# ----------------------------------------------------------------------------- eval

def identity_scores(queries: List[Dict]) -> Dict:
    """Rank each pool by distance to the CURRENT state. The floor an LLM must beat."""
    ranks = [rank_of_truth(q["state"], q["pool"], q["correct_idx"]) for q in queries]
    return _summarize(ranks, [q["state"] == q["next_state"] for q in queries])


def _summarize(ranks: List[int], exact: List[bool]) -> Dict:
    r = np.array(ranks, dtype=float)
    return {f"hit@{k}": float((r <= k).mean()) for k in (1, 5, 10)} | {
        "mean_rank": float(r.mean()), "exact_match": float(np.mean(exact)), "n": len(ranks)}


def run_model(model: str, queries: List[Dict], api_key: str, workers: int,
              cache: Dict[str, str], desc: str) -> Tuple[Dict, Dict, Dict]:
    """Generate successors, rank them, and report. Cached by triplet index."""
    todo = [q for q in queries if str(q["triplet_idx"]) not in cache]
    tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(call_model,
                                 PROMPT.format(problem=q["problem"], state=q["state"],
                                               action=q["action"]),
                                 model, api_key): q for q in todo}
            done = 0
            for fut in as_completed(futures):
                q = futures[fut]
                text, usage = fut.result()
                cache[str(q["triplet_idx"])] = text
                tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
                tokens["completion_tokens"] += usage.get("completion_tokens", 0)
                done += 1
                if done % 25 == 0:
                    print(f"    {desc}: {done}/{len(todo)}", flush=True)

    ranks, exact, empty = [], [], 0
    for q in queries:
        gen = normalize_state(cache.get(str(q["triplet_idx"]), ""))
        if not gen:
            empty += 1
        ranks.append(rank_of_truth(gen, q["pool"], q["correct_idx"]))
        exact.append(gen == q["next_state"])

    scores = _summarize(ranks, exact)
    scores["empty_generations"] = empty
    tokens["elapsed_s"] = round(time.time() - t0, 1)
    tokens["n_called"] = len(todo)
    return scores, tokens, cache


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="+", default=["ferry", "logistics"])
    ap.add_argument("--splits", nargs="+", default=["random", "problem_grouped"])
    ap.add_argument("--models", nargs="+", default=["openai/gpt-5.4"])
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--encoder", default="BAAI/bge-m3",
                    help="only used to resolve the triplet table; no embeddings are scored here")
    ap.add_argument("--dry_run", action="store_true", help="build queries and score the floor only")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("set OPENROUTER_API_KEY (never pass the key as an argument)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Merge into any existing results rather than replacing them: separate
    # invocations cover different model sets and would otherwise clobber each
    # other's scores, leaving only the last set written.
    results_path = OUT_DIR / f"results_seed{args.seed}_n{args.max_samples}.json"
    everything = json.loads(results_path.read_text()) if results_path.exists() else {}

    for domain in args.domains:
        for split in args.splits:
            queries, meta = build_queries(domain, split, args.seed, args.max_samples, args.encoder)
            if not queries:
                print(f"{domain}/{split}: no queries, skipping", flush=True)
                continue
            key = f"{domain}/{split}"
            floor = identity_scores(queries)
            everything.setdefault(key, {"models": {}})
            everything[key]["meta"] = meta
            everything[key]["identity"] = floor
            print(f"\n=== {key}  n={meta['n_queries']}  pool~{meta['mean_pool']:.0f} ===", flush=True)
            print(f"  {'identity':32s} h@1={floor['hit@1']:.3f} h@5={floor['hit@5']:.3f} "
                  f"exact={floor['exact_match']:.3f}", flush=True)
            if args.dry_run:
                continue

            for model in args.models:
                safe = model.replace("/", "_")
                cache_path = OUT_DIR / f"cache_{safe}_{domain}_{split}_s{args.seed}_n{args.max_samples}.json"
                cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
                scores, tokens, cache = run_model(model, queries, api_key, args.workers,
                                                  cache, f"{model.split('/')[-1]} {key}")
                cache_path.write_text(json.dumps(cache))
                everything[key]["models"][model] = {"scores": scores, "usage": tokens}
                print(f"  {model:32s} h@1={scores['hit@1']:.3f} h@5={scores['hit@5']:.3f} "
                      f"exact={scores['exact_match']:.3f} "
                      f"({tokens['n_called']} calls, {tokens['elapsed_s']:.0f}s, "
                      f"{tokens['prompt_tokens'] + tokens['completion_tokens']} tok)", flush=True)

                results_path.write_text(json.dumps(everything, indent=2))

    results_path.write_text(json.dumps(everything, indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
