"""
LLM-based Transition Function Experiment

Usage:
    python -m experiments.llm_ranking --dry_run                     # Test without API
    python -m experiments.llm_ranking --all_models                  # Run all models
    python -m experiments.llm_ranking --log_file results/log.txt    # Save output to file
    python -m experiments.llm_ranking --all_models --use_cache      # Run and skip already completed models
"""

import argparse, json, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import requests
import pandas as pd
import wandb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from embedplan.data import FactorizedTripletDataset, ProblemGroupedBatchSampler

MODELS = [
    "meta-llama/llama-3.1-70b-instruct",     # Standard baseline
    "meta-llama/llama-3.1-8b-instruct",      # Small baseline
    "qwen/qwen3-30b-a3b-instruct-2507",      # Replaces Qwen 2.5 32B
    "openai/gpt-oss-120b",                   # The new heavy open-weight champ
    # "openai/gpt-5.4-mini",                   # Replaces 4o-mini
    "openai/gpt-5.4"                         # Top tier closed model
]
DEFAULT_DOMAINS = ['ferry', 'logistics']
ALL_DOMAINS = ['ferry', 'rovers', 'blocksworld', 'depot', 'floortile', 'goldminer', 'grid', 'logistics', 'satellite']
API_URL = "https://openrouter.ai/api/v1/chat/completions"
CANDIDATE_POOL_SIZE = 128


class Logger:
    """Dual output to console and file."""
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8') if log_file else None

    def write(self, msg):
        self.terminal.write(msg)
        if self.log:
            self.log.write(msg)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()


def load_domain_batches(domain: str, embedding_model: str, batch_size: int = CANDIDATE_POOL_SIZE,
                        seed: int = 42) -> List[List[Dict]]:
    """Load domain data using FactorizedTripletDataset + ProblemGroupedBatchSampler.
    Returns list of batches, each batch is a list of sample dicts with text fields."""
    ds = FactorizedTripletDataset(domain=domain, model_name=embedding_model)
    sampler = ProblemGroupedBatchSampler(ds, batch_size, shuffle_problems=False, seed=seed)

    batches = []
    for idx_batch in sampler:
        batch = []
        for i in idx_batch:
            sample = ds[i]
            sample["problem_description"] = ds.values["problem"][sample["problem_id"]]
            batch.append(sample)
        batches.append(batch)

    total = sum(len(b) for b in batches)
    print(f"{domain}: {total} triplets in {len(batches)} problem-grouped batches (pool_size={batch_size})")
    return batches


def query_llm(prompt: str, model: str, api_key: str, dry_run: bool = False) -> Tuple[str, Dict]:
    """Query OpenRouter API or return mock response for dry run."""
    est_tokens = len(prompt) // 4

    if dry_run:
        mock = "[DRY RUN] Mock prediction simulating LLM output."
        return mock, {"model": f"{model} (dry-run)", "prompt_tokens": est_tokens,
                      "completion_tokens": len(mock)//4, "total_tokens": est_tokens + len(mock)//4}

    try:
        resp = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                  "max_tokens": 2000, "temperature": 0.0}, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        usage = result.get("usage", {})
        return result["choices"][0]["message"]["content"], {
            "model": result.get("model", model), "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)}
    except Exception as e:
        print(f"API error: {e}")
        return "", {"error": str(e), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def normalize_state(s: str) -> str:
    """Canonicalize a state string: lowercase, clean punctuation noise, split into facts, sort, rejoin."""
    import re
    s = s.strip().lower()
    s = s.replace(" and ", ", ")
    # Collapse repeated commas/spaces (e.g. ",," -> ",")
    s = re.sub(r',\s*,', ',', s)
    # Normalize whitespace around commas
    s = re.sub(r'\s*,\s*', ', ', s)
    # Strip trailing/leading commas
    s = s.strip(', ')

    facts = [f.strip() for f in s.split(", ") if f.strip()]
    return ", ".join(sorted(facts))


def hamming_distance(s1: str, s2: str) -> int:
    """Character-level Hamming distance between two strings (padded to equal length)."""
    n = max(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1.ljust(n), s2.ljust(n)))


def embed_texts(texts: List[str], embedder: SentenceTransformer) -> np.ndarray:
    """Embed a list of texts, returns (N, D) array."""
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def rank_and_eval(pred: str, states: List[str], correct_idx: int,
                  pred_embedding: np.ndarray, state_embeddings: np.ndarray) -> Dict:
    """Rank states by Hamming distance and embedding cosine similarity. Compute Hit@k."""
    results = {}

    # Hamming distance ranking (character-level)
    hamming_ranked = sorted(enumerate(states), key=lambda x: hamming_distance(pred, x[1]))
    hamming_rank = next((i for i, (idx, _) in enumerate(hamming_ranked) if idx == correct_idx), -1)
    results["hamming"] = {
        "rank": hamming_rank + 1 if hamming_rank >= 0 else -1,
        "hit@1": int(correct_idx in [idx for idx, _ in hamming_ranked[:1]]),
        "hit@5": int(correct_idx in [idx for idx, _ in hamming_ranked[:5]]),
        "hit@10": int(correct_idx in [idx for idx, _ in hamming_ranked[:10]]),
    }

    # Embedding cosine ranking (using pre-computed embeddings)
    norms = np.linalg.norm(state_embeddings, axis=1) * np.linalg.norm(pred_embedding) + 1e-10
    cos_sims = (state_embeddings @ pred_embedding) / norms
    emb_order = np.argsort(-cos_sims)  # highest similarity first
    emb_rank = np.where(emb_order == correct_idx)[0]
    emb_rank = int(emb_rank[0]) if len(emb_rank) > 0 else -2
    results["embedding"] = {
        "rank": emb_rank + 1 if emb_rank >= 0 else -1,
        "hit@1": int(correct_idx in emb_order[:1]),
        "hit@5": int(correct_idx in emb_order[:5]),
        "hit@10": int(correct_idx in emb_order[:10]),
    }

    return results


def run_single_model(model: str, domains: List[str], api_key: str, embedder: SentenceTransformer,
                     domain_batches: Dict[str, List[List[Dict]]], max_samples: int = 100,
                     dry_run: bool = False, seed: int = 42, use_wandb: bool = False,
                     output_dir: str = "results/llm_experiment") -> Dict:
    """Run experiment for a single model across domains using problem-grouped batches."""
    print(f"\n{'='*60}\nMODEL: {model}\n{'='*60}")

    rng = random.Random(seed)
    all_results, domain_metrics = [], {}
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start_time = time.time()
    safe_model = model.replace('/', '_')

    for domain in domains:
        batches = domain_batches[domain]
        if not batches:
            continue

        # Flatten to get total sample count, then decide how many to run
        flat_samples = [(b, i) for b in batches for i in range(len(b))]
        if len(flat_samples) > max_samples:
            flat_samples = rng.sample(flat_samples, max_samples)
        # Re-group by batch to preserve problem-grouped candidate pools
        batch_to_items: Dict[int, List[int]] = {}
        for b, i in flat_samples:
            bid = id(b)
            if bid not in batch_to_items:
                batch_to_items[bid] = (b, [])
            batch_to_items[bid][1].append(i)

        hits = {"hamming": {f"hit@{k}": 0 for k in [1,5,10]}, "embedding": {f"hit@{k}": 0 for k in [1,5,10]}}
        domain_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        domain_results = []
        n_total = len(flat_samples)

        for batch, item_indices in tqdm(batch_to_items.values(), desc=f"{model.split('/')[-1][:20]} - {domain}"):
            # The candidate pool = all sp_str in this batch (problem-grouped, like trained model eval)
            # Embed with problem description prefix so embeddings are context-aware
            pool_texts = [f"{s['problem_description']}\n{normalize_state(s['sp_str'])}" for s in batch]
            pool_states = [normalize_state(s["sp_str"]) for s in batch]
            pool_embeddings = embed_texts(pool_texts, embedder)

            for idx in item_indices:
                sample = batch[idx]
                problem_desc = sample["problem_description"]
                norm_state = normalize_state(sample["s_str"])
                norm_next = normalize_state(sample["sp_str"])
                action = sample["a_str"]
                correct_idx = idx  # sample's own sp_str is at position idx in the batch pool

                prompt = f"""Predict the state after executing an action.

### PROBLEM ###
{problem_desc}

### CURRENT STATE ###
{norm_state}

### ACTION ###
{action}

Describe the resulting state. Output ONLY the new state description."""

                pred, meta = query_llm(prompt, model, api_key, dry_run)
                norm_pred = normalize_state(pred) if pred else ""

                if norm_pred:
                    pred_text = f"{problem_desc}\n{norm_pred}"
                    pred_embedding = embed_texts([pred_text], embedder)[0]
                    metrics = rank_and_eval(norm_pred, pool_states, correct_idx, pred_embedding, pool_embeddings)
                else:
                    metrics = {m: {"rank": -1, "hit@1": 0, "hit@5": 0, "hit@10": 0} for m in ["hamming", "embedding"]}

                for m in ["hamming", "embedding"]:
                    for k in [1, 5, 10]:
                        hits[m][f"hit@{k}"] += metrics[m][f"hit@{k}"]

                for k in domain_tokens:
                    domain_tokens[k] += meta.get(k, 0)
                    total_tokens[k] += meta.get(k, 0)

                sample_record = {
                    "domain": domain, "model": model,
                    "action": action,
                    "input_state": norm_state,
                    "prediction": norm_pred,
                    "ground_truth": norm_next,
                    "prompt": prompt,
                    "pool_size": len(batch),
                    "metrics": metrics, "token_usage": meta,
                    "is_hit1_hamming": bool(metrics["hamming"]["hit@1"]),
                    "is_hit1_embedding": bool(metrics["embedding"]["hit@1"]),
                }
                all_results.append(sample_record)
                domain_results.append(sample_record)

                # Print one example per model+domain so we can eyeball quality
                if len(domain_results) == 1:
                    print(f"\n--- Example output for {model.split('/')[-1]} / {domain} (pool={len(batch)}) ---")
                    print(f"ACTION:     {action}")
                    print(f"PREDICTION: {norm_pred[:500]}")
                    print(f"TARGET:     {norm_next[:500]}")
                    print(f"---\n")

                if not dry_run:
                    time.sleep(0.5)

        # Save I/O logs organized by model/domain
        io_dir = Path(output_dir) / "io_logs" / safe_model / domain
        io_dir.mkdir(parents=True, exist_ok=True)

        with open(io_dir / "all_samples.json", "w") as f:
            json.dump(domain_results, f, indent=2)

        successes = [r for r in domain_results if r["is_hit1_embedding"]]
        failures = [r for r in domain_results if not r["is_hit1_embedding"]]
        with open(io_dir / "successes.json", "w") as f:
            json.dump(successes, f, indent=2)
        with open(io_dir / "failures.json", "w") as f:
            json.dump(failures, f, indent=2)

        print(f"Saved {len(domain_results)} samples ({len(successes)} successes, {len(failures)} failures) to {io_dir}")

        n = n_total
        avg_completion = domain_tokens["completion_tokens"] / n if n > 0 else 0
        domain_tokens["avg_completion_tokens"] = avg_completion

        domain_metrics[domain] = {
            "n_samples": n, "num_candidates": CANDIDATE_POOL_SIZE, "token_usage": domain_tokens,
            "hamming": {k: hits["hamming"][k]/n for k in hits["hamming"]},
            "embedding": {k: hits["embedding"][k]/n for k in hits["embedding"]},
        }

        # WANDB LOGGING
        if use_wandb:
            wandb.log({
                f"{model}/{domain}/hamming/hit@1": domain_metrics[domain]['hamming']['hit@1'],
                f"{model}/{domain}/hamming/hit@5": domain_metrics[domain]['hamming']['hit@5'],
                f"{model}/{domain}/hamming/hit@10": domain_metrics[domain]['hamming']['hit@10'],
                f"{model}/{domain}/embedding/hit@1": domain_metrics[domain]['embedding']['hit@1'],
                f"{model}/{domain}/embedding/hit@5": domain_metrics[domain]['embedding']['hit@5'],
                f"{model}/{domain}/embedding/hit@10": domain_metrics[domain]['embedding']['hit@10'],
                f"{model}/{domain}/tokens/prompt": domain_tokens['prompt_tokens'],
                f"{model}/{domain}/tokens/completion": domain_tokens['completion_tokens'],
                f"{model}/{domain}/tokens/total": domain_tokens['total_tokens'],
                f"{model}/{domain}/tokens/avg_completion": domain_tokens['avg_completion_tokens'],
            })

        print(f"\n{domain}: Hamming hit@1={domain_metrics[domain]['hamming']['hit@1']:.3f}, "
              f"hit@5={domain_metrics[domain]['hamming']['hit@5']:.3f}, hit@10={domain_metrics[domain]['hamming']['hit@10']:.3f}")
        print(f"{domain}: Embedding hit@1={domain_metrics[domain]['embedding']['hit@1']:.3f}, "
              f"hit@5={domain_metrics[domain]['embedding']['hit@5']:.3f}, hit@10={domain_metrics[domain]['embedding']['hit@10']:.3f}")
        print(f"{domain}: Tokens - prompt={domain_tokens['prompt_tokens']}, completion={domain_tokens['completion_tokens']} (avg {avg_completion:.1f}), total={domain_tokens['total_tokens']}")

    elapsed = time.time() - start_time
    print(f"\nModel {model} completed in {elapsed:.1f}s")
    print(f"Total tokens: prompt={total_tokens['prompt_tokens']}, completion={total_tokens['completion_tokens']}, total={total_tokens['total_tokens']}")

    return {
        "model": model, "domains": domains, "domain_metrics": domain_metrics,
        "total_tokens": total_tokens, "elapsed_seconds": elapsed, "results": all_results
    }


def run_experiment(models: List[str], domains: List[str], api_key: str, embedding_model: str,
                   max_samples: int = 50, output_dir: str = "results/llm_experiment",
                   dry_run: bool = False, seed: int = 42, use_wandb: bool = False,
                   wandb_project: str = "llm-transition-experiment", use_cache: bool = False):
    """Run full experiment across all models and domains."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# LLM TRANSITION FUNCTION EXPERIMENT")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"# Models: {models}")
    print(f"# Domains: {domains}")
    print(f"# Max samples per domain: {max_samples}")
    print(f"# Dry run: {dry_run}")
    print(f"# Seed: {seed}")
    print(f"# Use Cache: {use_cache}")
    print(f"{'#'*60}\n")

    if dry_run:
        print("!"*50 + "\n! DRY RUN MODE - No API calls\n" + "!"*50)

    # Load embedding model once, shared across all LLM models and domains
    print("Loading sentence-transformers embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model ready.")

    # Load problem-grouped batches once per domain (shared across all LLM models)
    domain_batches = {}
    for domain in domains:
        domain_batches[domain] = load_domain_batches(domain, embedding_model, CANDIDATE_POOL_SIZE, seed)

    if use_wandb:
        wandb.init(project=wandb_project,
                   name=f"eval_{timestamp}",
                   config={"models": models, "domains": domains, "max_samples": max_samples, "dry_run": dry_run, "seed": seed})

    all_model_results = {}
    grand_total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    experiment_start = time.time()

    for model in models:
        safe_model_name = model.replace('/', '_')
        cache_file = output_path / f"cache_{safe_model_name}_samples{max_samples}_seed{seed}.json"

        # CACHE LOGIC
        if use_cache and cache_file.exists():
            print(f"\n>>> FOUND CACHE for {model}. Loading from {cache_file}...")
            with open(cache_file, "r") as f:
                result = json.load(f)

            # Log cached results to WandB to maintain dashboard integrity
            if use_wandb:
                for domain, dm in result["domain_metrics"].items():
                    wandb.log({
                        f"{model}/{domain}/hamming/hit@1": dm['hamming']['hit@1'],
                        f"{model}/{domain}/hamming/hit@5": dm['hamming']['hit@5'],
                        f"{model}/{domain}/hamming/hit@10": dm['hamming']['hit@10'],
                        f"{model}/{domain}/embedding/hit@1": dm['embedding']['hit@1'],
                        f"{model}/{domain}/embedding/hit@5": dm['embedding']['hit@5'],
                        f"{model}/{domain}/embedding/hit@10": dm['embedding']['hit@10'],
                        f"{model}/{domain}/tokens/prompt": dm['token_usage']['prompt_tokens'],
                        f"{model}/{domain}/tokens/completion": dm['token_usage']['completion_tokens'],
                        f"{model}/{domain}/tokens/total": dm['token_usage']['total_tokens'],
                        f"{model}/{domain}/tokens/avg_completion": dm['token_usage']['avg_completion_tokens'],
                    })
        else:
            # Run model if no cache exists
            result = run_single_model(model, domains, api_key, embedder, domain_batches, max_samples, dry_run, seed, use_wandb, output_dir)

            # Save deterministic cache file immediately
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved cache: {cache_file}")

        all_model_results[model] = result

        for k in grand_total_tokens:
            grand_total_tokens[k] += result["total_tokens"][k]

        # Save standard timestamped per-model result
        model_file = output_path / f"results_{safe_model_name}_{timestamp}.json"
        with open(model_file, "w") as f:
            json.dump(result, f, indent=2)

    # Final summary
    total_elapsed = time.time() - experiment_start
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT COMPLETE")
    print(f"# Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"# Grand total tokens: {grand_total_tokens}")
    print(f"{'#'*60}")

    # Summary table
    print(f"\n{'='*95}")
    print(f"{'MODEL':<35} {'DOMAIN':<12} {'H@1':<8} {'H@5':<8} {'H@10':<8} {'TOKENS':<10} {'AVG_CPL':<8}")
    print(f"{'='*95}")
    for model, res in all_model_results.items():
        model_short = model.split('/')[-1][:33]
        for domain, dm in res["domain_metrics"].items():
            print(f"{model_short:<35} {domain:<12} {dm['hamming']['hit@1']:<8.3f} {dm['hamming']['hit@5']:<8.3f} {dm['hamming']['hit@10']:<8.3f} {dm['token_usage']['total_tokens']:<10} {dm['token_usage']['avg_completion_tokens']:<8.1f}")
    print(f"{'='*95}")

    # Save combined results
    combined = {
        "config": {
            "models": models, "domains": domains, "max_samples": max_samples,
            "dry_run": dry_run, "seed": seed, "timestamp": timestamp
        },
        "grand_total_tokens": grand_total_tokens,
        "total_elapsed_seconds": total_elapsed,
        "model_results": {m: {k: v for k, v in r.items() if k != "results"} for m, r in all_model_results.items()},
        "all_results": [r for res in all_model_results.values() for r in res["results"]]
    }

    combined_file = output_path / f"combined_results_{timestamp}.json"
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results: {combined_file}")

    # Save summary CSV
    summary_rows = []
    for model, res in all_model_results.items():
        for domain, dm in res["domain_metrics"].items():
            summary_rows.append({
                "model": model, "domain": domain, "n_samples": dm["n_samples"],
                "hamming_hit@1": dm["hamming"]["hit@1"], "hamming_hit@5": dm["hamming"]["hit@5"], "hamming_hit@10": dm["hamming"]["hit@10"],
                "embedding_hit@1": dm["embedding"]["hit@1"], "embedding_hit@5": dm["embedding"]["hit@5"], "embedding_hit@10": dm["embedding"]["hit@10"],
                "prompt_tokens": dm["token_usage"]["prompt_tokens"], "completion_tokens": dm["token_usage"]["completion_tokens"],
                "avg_completion_tokens": dm["token_usage"]["avg_completion_tokens"],
                "total_tokens": dm["token_usage"]["total_tokens"]
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_path / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV: {summary_csv}")

    if use_wandb:
        wandb.finish()

    return all_model_results


def main():
    parser = argparse.ArgumentParser(description="LLM Transition Function Experiment")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS, choices=ALL_DOMAINS)
    parser.add_argument("--model", default=None, help="Single model to run")
    parser.add_argument("--all_models", action="store_true", help="Run all models")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", default="results/llm_experiment_updated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_file", default=None, help="Save all output to file")
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llm-transition-experiment", help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--use_cache", action="store_true", help="Load cached model results if they exist")
    parser.add_argument("--embedding_model", default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model name for FactorizedTripletDataset embeddings (must match encoded data)")
    args = parser.parse_args()

    # Setup logging
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Logger(args.log_file)
        print(f"Logging to: {args.log_file}")

    if args.list_models:
        print("Models:", MODELS)
        return

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: Set OPENROUTER_API_KEY or use --dry_run")
        return

    # Determine models to run
    if args.all_models:
        models = MODELS
    elif args.model:
        models = [args.model]
    else:
        models = [MODELS[2]]  # Default: llama-8b

    use_wandb = not args.no_wandb

    run_experiment(models, args.domains, api_key or "", args.embedding_model, args.max_samples,
                   args.output_dir, args.dry_run, args.seed, use_wandb, args.wandb_project, args.use_cache)


if __name__ == "__main__":
    main()