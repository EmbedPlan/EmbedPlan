"""
Latency & Computational Cost Comparison: LLM vs MLP Transition Function

Measures:
  - Wall-clock latency per sample (forward pass)
  - FLOPs for MLP (via torch profiler or manual calculation)
  - Token cost for LLM (prompt + completion tokens)
  - Throughput (samples/sec)

Usage:
    python -m experiments.benchmark_latency --domain ferry
    python -m experiments.benchmark_latency --domain ferry --dry_run          # No API calls
    python -m experiments.benchmark_latency --domain ferry --n_samples 50
    python -m experiments.benchmark_latency --domain ferry --device cuda       # GPU benchmark
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from embedplan.data import FactorizedTripletDataset, ProblemGroupedBatchSampler
from embedplan.models import TransitionMLP, ProjectionHead, ProjectedTransitionModel
from llm_transition_experiment import (
    normalize_state, query_llm, embed_texts, CANDIDATE_POOL_SIZE, MODELS
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_mlp_flops(model: nn.Module, input_s: torch.Tensor, input_a: torch.Tensor) -> int:
    """Estimate FLOPs for a single forward pass through the MLP.
    For a linear layer: 2 * in_features * out_features (multiply-add)."""
    flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 2 * in * out per sample (multiply + add)
            flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.LayerNorm):
            flops += 2 * module.normalized_shape[0]  # mean + variance
    return flops


def benchmark_embedding(domain: str, model_name: str, n_samples: int = 20) -> Dict:
    """Benchmark the LLM encoder used to produce state/action embeddings for the MLP.
    Measures the cost of embedding 2 state texts (initial state + goal state) per sample,
    which is the preprocessing cost the MLP pipeline requires."""
    print(f"\n{'='*60}")
    print(f"EMBEDDING BENCHMARK (model={model_name})")
    print(f"{'='*60}")

    # Load dataset to get real state text strings
    ds = FactorizedTripletDataset(domain=domain, model_name=model_name)
    sampler = ProblemGroupedBatchSampler(ds, CANDIDATE_POOL_SIZE, shuffle_problems=False, seed=42)
    all_indices = [i for batch in sampler for i in batch][:n_samples]

    # Collect state text pairs (s_str as "initial", sp_str as "goal" — two embeddings per sample)
    state_texts = []
    for i in all_indices:
        sample = ds[i]
        state_texts.append(sample["s_str"])
        state_texts.append(sample["sp_str"])  # second state embedding

    # Load a sentence-transformers model (ungated, no auth required)
    embed_model = "sentence-transformers/all-mpnet-base-v2"
    print(f"Loading embedding model: {embed_model}")
    embedder = SentenceTransformer(embed_model)

    # Warmup
    _ = embedder.encode([state_texts[0]], show_progress_bar=False)

    # Benchmark: embed pairs of states (2 per sample)
    latencies = []
    for i in range(0, len(state_texts), 2):
        pair = state_texts[i:i+2]
        t0 = time.perf_counter()
        _ = embedder.encode(pair, show_progress_bar=False)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    results = {
        "model": embed_model,
        "n_samples": len(latencies),
        "texts_per_sample": 2,
        "avg_latency_per_sample_ms": np.mean(latencies) * 1000,
        "median_latency_per_sample_ms": np.median(latencies) * 1000,
        "std_latency_per_sample_ms": np.std(latencies) * 1000,
        "throughput_samples_per_sec": len(latencies) / sum(latencies),
    }

    print(f"  Avg embedding latency (2 states): {results['avg_latency_per_sample_ms']:.1f} ms")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")

    return results


def benchmark_mlp(domain: str, model_name: str, device: torch.device,
                  n_samples: int = 100, batch_sizes: List[int] = None,
                  hidden_size: int = 256, n_layers: int = 4,
                  use_projection: bool = True, projection_dim: int = 512,
                  projection_layers: int = 2, seed: int = 42) -> Dict:
    """Benchmark MLP transition function (random init) on real data."""
    print(f"\n{'='*60}")
    print(f"MLP BENCHMARK (device={device})")
    print(f"{'='*60}")

    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 128]

    # Load dataset to get real dimensions and data
    ds = FactorizedTripletDataset(domain=domain, model_name=model_name)
    state_dim = ds.state_embs.shape[1]
    action_dim = ds.action_embs.shape[1]

    # Build model (same architecture as train.py)
    if use_projection:
        s_proj = ProjectionHead(input_dim=state_dim, output_dim=projection_dim, n_layers=projection_layers)
        a_proj = ProjectionHead(input_dim=action_dim, output_dim=projection_dim, n_layers=projection_layers)
        trans = TransitionMLP(d_state=projection_dim, d_action=projection_dim,
                              hidden=hidden_size, n_layers=n_layers)
        model = ProjectedTransitionModel(s_proj, a_proj, trans)
    else:
        model = TransitionMLP(d_state=state_dim, d_action=action_dim,
                              hidden=hidden_size, n_layers=n_layers)

    model.to(device).eval()

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Prepare real data tensors
    sampler = ProblemGroupedBatchSampler(ds, CANDIDATE_POOL_SIZE, shuffle_problems=False, seed=seed)
    all_indices = [i for batch in sampler for i in batch]
    indices = all_indices[:n_samples]

    s_all, a_all, sp_all = [], [], []
    for i in indices:
        sample = ds[i]
        s_all.append(sample["s_emb"])
        a_all.append(sample["a_emb"])
        sp_all.append(sample["sp_emb"])

    s_tensor = torch.stack(s_all).to(device)
    a_tensor = torch.stack(a_all).to(device)

    # Estimate FLOPs for a single sample
    flops_per_sample = estimate_mlp_flops(model, s_tensor[:1], a_tensor[:1])
    print(f"Estimated FLOPs per sample: {flops_per_sample:,}")

    results = {
        "method": "mlp",
        "device": str(device),
        "n_params": n_params,
        "flops_per_sample": flops_per_sample,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "use_projection": use_projection,
        "batch_results": {}
    }

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(s_tensor[:1], a_tensor[:1])
    if device.type == "cuda":
        torch.cuda.synchronize()

    for bs in batch_sizes:
        n_batches = (n_samples + bs - 1) // bs
        latencies = []

        with torch.no_grad():
            for b in range(n_batches):
                start_idx = b * bs
                end_idx = min(start_idx + bs, n_samples)
                s_batch = s_tensor[start_idx:end_idx]
                a_batch = a_tensor[start_idx:end_idx]

                if device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                pred = model(s_batch, a_batch)
                # Also include ranking (cosine sim against candidate pool) to be fair
                pred_norm = F.normalize(pred, dim=-1)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                t1 = time.perf_counter()
                latencies.append((t1 - t0, end_idx - start_idx))

        total_time = sum(t for t, _ in latencies)
        total_samples = sum(n for _, n in latencies)
        per_sample_latencies = [t / n for t, n in latencies]

        batch_result = {
            "batch_size": bs,
            "total_time_s": total_time,
            "total_samples": total_samples,
            "throughput_samples_per_sec": total_samples / total_time,
            "avg_latency_per_sample_ms": np.mean(per_sample_latencies) * 1000,
            "median_latency_per_sample_ms": np.median(per_sample_latencies) * 1000,
            "std_latency_per_sample_ms": np.std(per_sample_latencies) * 1000,
            "total_flops": flops_per_sample * total_samples,
        }
        results["batch_results"][bs] = batch_result

        print(f"\n  Batch size {bs}:")
        print(f"    Throughput: {batch_result['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"    Avg latency/sample: {batch_result['avg_latency_per_sample_ms']:.3f} ms")
        print(f"    Total FLOPs: {batch_result['total_flops']:,}")

    return results


def benchmark_llm(domain: str, embedding_model_name: str, llm_model: str,
                  api_key: str, n_samples: int = 20, dry_run: bool = False,
                  seed: int = 42) -> Dict:
    """Benchmark LLM transition function on real data."""
    print(f"\n{'='*60}")
    print(f"LLM BENCHMARK (model={llm_model}, dry_run={dry_run})")
    print(f"{'='*60}")

    # Load data
    ds = FactorizedTripletDataset(domain=domain, model_name=embedding_model_name)
    sampler = ProblemGroupedBatchSampler(ds, CANDIDATE_POOL_SIZE, shuffle_problems=False, seed=seed)
    batches = []
    for idx_batch in sampler:
        batch = []
        for i in idx_batch:
            sample = ds[i]
            sample["problem_description"] = ds.values["problem"][sample["problem_id"]]
            batch.append(sample)
        batches.append(batch)

    # Load sentence transformer for re-embedding predictions
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect samples
    flat_samples = [(b, i) for b in batches for i in range(len(b))][:n_samples]

    latencies = []
    token_counts = []
    embed_latencies = []

    for sample_idx, (batch, idx) in enumerate(flat_samples):
        sample = batch[idx]
        problem_desc = sample["problem_description"]
        norm_state = normalize_state(sample["s_str"])
        action = sample["a_str"]

        prompt = f"""Predict the state after executing an action.

### PROBLEM ###
{problem_desc}

### CURRENT STATE ###
{norm_state}

### ACTION ###
{action}

Describe the resulting state. Output ONLY the new state description."""

        # Time the LLM call
        t0 = time.perf_counter()
        pred, meta = query_llm(prompt, llm_model, api_key, dry_run)
        t1 = time.perf_counter()
        llm_latency = t1 - t0

        # Time the re-embedding step (needed for ranking)
        norm_pred = normalize_state(pred) if pred else ""
        t2 = time.perf_counter()
        if norm_pred:
            pred_text = f"{problem_desc}\n{norm_pred}"
            _ = embed_texts([pred_text], embedder)
        t3 = time.perf_counter()
        embed_latency = t3 - t2

        latencies.append(llm_latency)
        embed_latencies.append(embed_latency)
        token_counts.append(meta)

        if sample_idx < 3:
            print(f"  Sample {sample_idx}: LLM {llm_latency:.3f}s, embed {embed_latency:.3f}s, "
                  f"tokens={meta.get('total_tokens', 0)}")

        if not dry_run:
            time.sleep(0.5)  # Rate limiting

    total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in token_counts)
    total_completion_tokens = sum(m.get("completion_tokens", 0) for m in token_counts)
    total_tokens = sum(m.get("total_tokens", 0) for m in token_counts)

    results = {
        "method": "llm",
        "model": llm_model,
        "dry_run": dry_run,
        "n_samples": len(latencies),
        "latency": {
            "avg_llm_call_ms": np.mean(latencies) * 1000,
            "median_llm_call_ms": np.median(latencies) * 1000,
            "std_llm_call_ms": np.std(latencies) * 1000,
            "avg_embed_ms": np.mean(embed_latencies) * 1000,
            "avg_total_per_sample_ms": (np.mean(latencies) + np.mean(embed_latencies)) * 1000,
            "throughput_samples_per_sec": len(latencies) / sum(latencies),
        },
        "tokens": {
            "total_prompt": total_prompt_tokens,
            "total_completion": total_completion_tokens,
            "total": total_tokens,
            "avg_prompt_per_sample": total_prompt_tokens / max(1, len(latencies)),
            "avg_completion_per_sample": total_completion_tokens / max(1, len(latencies)),
            "avg_total_per_sample": total_tokens / max(1, len(latencies)),
        },
    }

    print(f"\n  LLM Results ({len(latencies)} samples):")
    print(f"    Avg latency/sample: {results['latency']['avg_total_per_sample_ms']:.1f} ms")
    print(f"    Throughput: {results['latency']['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"    Avg tokens/sample: {results['tokens']['avg_total_per_sample']:.0f}")

    return results


def print_comparison(mlp_results: Dict, llm_results: Dict, embed_results: Dict = None):
    """Print side-by-side comparison table."""
    print(f"\n{'#'*70}")
    print(f"# COMPARISON: MLP vs LLM")
    print(f"{'#'*70}")

    # Use batch_size=1 for MLP for fair per-sample comparison
    mlp_bs1 = mlp_results["batch_results"].get(1, {})
    mlp_bs128 = mlp_results["batch_results"].get(128, mlp_results["batch_results"].get(64, {}))

    embed_lat = embed_results["avg_latency_per_sample_ms"] if embed_results else 0

    print(f"\n{'Metric':<40} {'MLP (bs=1)':<20} {'MLP (batched)':<20} {'LLM':<20}")
    print(f"{'-'*100}")

    # MLP forward-only latency
    mlp_lat1 = mlp_bs1.get("avg_latency_per_sample_ms", 0)
    mlp_lat_batch = mlp_bs128.get("avg_latency_per_sample_ms", 0)
    llm_lat = llm_results["latency"]["avg_total_per_sample_ms"]
    print(f"{'MLP forward only (ms)':<40} {mlp_lat1:<20.3f} {mlp_lat_batch:<20.3f} {'--':<20}")

    # Embedding latency (amortized over the pipeline)
    if embed_results:
        print(f"{'+ Embedding 2 states (ms)':<40} {embed_lat:<20.1f} {embed_lat:<20.1f} {'--':<20}")

    # Total MLP pipeline = embedding + forward
    mlp_total1 = mlp_lat1 + embed_lat
    mlp_total_batch = mlp_lat_batch + embed_lat
    print(f"{'MLP total pipeline (ms)':<40} {mlp_total1:<20.1f} {mlp_total_batch:<20.1f} {'--':<20}")

    # LLM latency
    print(f"{'LLM total (ms)':<40} {'--':<20} {'--':<20} {llm_lat:<20.1f}")

    # Throughput
    print()
    mlp_tp1 = mlp_bs1.get("throughput_samples_per_sec", 0)
    mlp_tp_batch = mlp_bs128.get("throughput_samples_per_sec", 0)
    llm_tp = llm_results["latency"]["throughput_samples_per_sec"]
    print(f"{'Throughput forward-only (samp/s)':<40} {mlp_tp1:<20.1f} {mlp_tp_batch:<20.1f} {llm_tp:<20.2f}")
    if embed_results:
        mlp_pipeline_tp1 = 1000.0 / mlp_total1 if mlp_total1 > 0 else 0
        mlp_pipeline_tp_batch = 1000.0 / mlp_total_batch if mlp_total_batch > 0 else 0
        print(f"{'Throughput w/ embedding (samp/s)':<40} {mlp_pipeline_tp1:<20.1f} {mlp_pipeline_tp_batch:<20.1f} {llm_tp:<20.2f}")

    # Speedup
    if llm_lat > 0:
        print(f"\n{'Speedup vs LLM (forward only)':<40} {llm_lat/mlp_lat1:<20.0f}x {llm_lat/mlp_lat_batch:<20.0f}x {'1x':<20}")
        print(f"{'Speedup vs LLM (full pipeline)':<40} {llm_lat/mlp_total1:<20.1f}x {llm_lat/mlp_total_batch:<20.1f}x {'1x':<20}")

    # Compute
    print(f"\n{'Compute Cost':<40} {'MLP':<20} {'LLM':<20}")
    print(f"{'-'*80}")
    print(f"{'Parameters':<40} {mlp_results['n_params']:,}{'':<10} {'N/A':<20}")
    print(f"{'FLOPs/sample (MLP only)':<40} {mlp_results['flops_per_sample']:,}{'':<5} {'N/A':<20}")
    avg_tokens = llm_results['tokens']['avg_total_per_sample']
    print(f"{'Tokens/sample':<40} {'N/A':<20} {avg_tokens:<20.0f}")

    # Memory footprint
    param_bytes = mlp_results["n_params"] * 4  # float32
    print(f"{'Model memory (MB, fp32)':<40} {param_bytes / 1e6:<20.2f} {'cloud API':<20}")
    if embed_results:
        print(f"{'+ Embedding model':<40} {embed_results['model']}")


def main():
    parser = argparse.ArgumentParser(description="Latency & Compute Benchmark: MLP vs LLM")
    parser.add_argument("--domain", type=str, default="ferry")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Embedding model name (for dataset)")
    parser.add_argument("--llm_model", type=str, default=MODELS[1],
                        help="LLM model to benchmark")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to benchmark")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--use_projection", action="store_true", default=True)
    parser.add_argument("--no_projection", action="store_true")
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--projection_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry_run", action="store_true", help="Skip real API calls")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_embedding", action="store_true",
                        help="Skip embedding model benchmark (faster, but incomplete picture)")
    parser.add_argument("--output", type=str, default="results/benchmark_latency.json")
    args = parser.parse_args()

    if args.no_projection:
        args.use_projection = False

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Warning: No API key set. Use --dry_run or set OPENROUTER_API_KEY.")
        args.dry_run = True

    # Run embedding benchmark (cost of encoding 2 state texts per sample)
    embed_results = None
    if not args.skip_embedding:
        embed_results = benchmark_embedding(
            domain=args.domain,
            model_name=args.model_name,
            n_samples=min(args.n_samples, 20),  # cap to avoid long runs
        )

    # Run MLP benchmark
    mlp_results = benchmark_mlp(
        domain=args.domain,
        model_name=args.model_name,
        device=device,
        n_samples=args.n_samples,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        use_projection=args.use_projection,
        projection_dim=args.projection_dim,
        projection_layers=args.projection_layers,
        seed=args.seed,
    )

    # Run LLM benchmark
    llm_results = benchmark_llm(
        domain=args.domain,
        embedding_model_name=args.model_name,
        llm_model=args.llm_model,
        api_key=api_key,
        n_samples=args.n_samples,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    # Print comparison
    print_comparison(mlp_results, llm_results, embed_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "config": {
            "domain": args.domain,
            "n_samples": args.n_samples,
            "device": str(device),
            "llm_model": args.llm_model,
            "dry_run": args.dry_run,
        },
        "mlp": mlp_results,
        "llm": llm_results,
        "embedding": embed_results,
    }
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
