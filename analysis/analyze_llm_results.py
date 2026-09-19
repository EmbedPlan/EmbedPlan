"""
Analyze LLM experiment results and print tables.

Usage:
    python analyze_llm_results.py                           # Auto-find from io_logs
    python analyze_llm_results.py --results_dir results/llm_experiment
    python analyze_llm_results.py --combined path/to/combined_results.json
"""

import argparse, json, glob
from pathlib import Path
import pandas as pd


def load_from_io_logs(results_dir: str = "results/llm_experiment") -> dict:
    """Load results from the io_logs directory structure."""
    io_dir = Path(results_dir) / "io_logs"
    if not io_dir.exists():
        return None

    data = {}  # model -> domain -> list of sample records
    for model_dir in sorted(io_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        data[model_name] = {}
        for domain_dir in sorted(model_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            samples_file = domain_dir / "all_samples.json"
            if samples_file.exists():
                with open(samples_file) as f:
                    data[model_name][domain_dir.name] = json.load(f)

    return data if data else None


def load_from_combined(path: str = None) -> dict:
    """Load from combined_results JSON (legacy format)."""
    if path:
        files = [path]
    else:
        files = sorted(glob.glob("results/llm_experiment/combined_results_*.json"))

    if not files:
        return None

    with open(files[-1]) as f:
        return json.load(f)


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(df.to_string())
    print()


def analyze_io_logs(data: dict):
    """Analyze from io_logs structure: model -> domain -> [samples]."""
    models = list(data.keys())
    domains = sorted({d for m in data for d in data[m]})

    if not models:
        print("No results found.")
        return

    # Hit@k tables
    for dist_type in ["hamming", "embedding"]:
        for k in [1, 5, 10]:
            metric = f"hit@{k}"
            rows = {}
            for domain in domains:
                rows[domain] = {}
                for model in models:
                    samples = data.get(model, {}).get(domain, [])
                    if not samples:
                        rows[domain][model] = "-"
                        continue
                    hits = sum(1 for s in samples if s["metrics"][dist_type][metric])
                    rows[domain][model] = f"{hits / len(samples):.3f}"

            df = pd.DataFrame(rows).T
            df.index.name = "Domain"
            print_table(df, f"{dist_type.upper()} - {metric.upper()}")

    # Success/failure summary
    print(f"\n{'='*70}")
    print(f" SUCCESS / FAILURE COUNTS (embedding hit@1)")
    print(f"{'='*70}")
    rows = []
    for model in models:
        for domain in domains:
            samples = data.get(model, {}).get(domain, [])
            if not samples:
                continue
            successes = sum(1 for s in samples if s["is_hit1_embedding"])
            failures = len(samples) - successes
            pool = samples[0].get("pool_size", "?")
            rows.append({
                "Model": model, "Domain": domain,
                "Total": len(samples), "Successes": successes,
                "Failures": failures, "Rate": f"{successes/len(samples):.3f}",
                "Pool": pool,
            })
    print(pd.DataFrame(rows).to_string(index=False))

    # Example predictions (1 per model+domain)
    print(f"\n{'='*70}")
    print(f" EXAMPLE PREDICTIONS (first sample per model/domain)")
    print(f"{'='*70}")
    for model in models:
        for domain in domains:
            samples = data.get(model, {}).get(domain, [])
            if not samples:
                continue
            s = samples[0]
            hit_e = "HIT" if s["is_hit1_embedding"] else "MISS"
            hit_h = "HIT" if s["is_hit1_hamming"] else "MISS"
            print(f"\n--- {model} / {domain} [emb:{hit_e} ham:{hit_h}] ---")
            print(f"  ACTION:     {s['action']}")
            print(f"  PREDICTION: {s['prediction'][:300]}")
            print(f"  TARGET:     {s['ground_truth'][:300]}")


def analyze_combined(data: dict):
    """Analyze from combined_results JSON (legacy)."""
    model_results = data.get("model_results", {})
    if not model_results:
        print("No model results found.")
        return

    models = list(model_results.keys())
    domains = sorted({d for m in model_results for d in model_results[m].get("domain_metrics", {})})

    for dist_type in ["hamming", "embedding"]:
        for k in [1, 5, 10]:
            metric = f"hit@{k}"
            rows = {}
            for domain in domains:
                rows[domain] = {}
                for model in models:
                    short = model.split("/")[-1].replace("-instruct", "")
                    dm = model_results[model]["domain_metrics"].get(domain, {})
                    val = dm.get(dist_type, {}).get(metric, 0)
                    rows[domain][short] = f"{val:.3f}"
            df = pd.DataFrame(rows).T
            df.index.name = "Domain"
            print_table(df, f"{dist_type.upper()} - {metric.upper()}")

    # Token usage
    print(f"\n{'='*70}")
    print(f" TOKEN USAGE")
    print(f"{'='*70}")
    token_rows = []
    for model in models:
        res = model_results[model]
        short = model.split("/")[-1].replace("-instruct", "")
        token_rows.append({
            "Model": short,
            "Prompt": res["total_tokens"]["prompt_tokens"],
            "Completion": res["total_tokens"]["completion_tokens"],
            "Total": res["total_tokens"]["total_tokens"],
            "Time (s)": f"{res.get('elapsed_seconds', 0):.1f}"
        })
    print(pd.DataFrame(token_rows).set_index("Model").to_string())
    print(f"\nGrand Total Tokens: {data.get('grand_total_tokens', {})}")
    print(f"Total Time: {data.get('total_elapsed_seconds', 0):.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM experiment results")
    parser.add_argument("--results_dir", default="results/llm_experiment",
                        help="Directory containing io_logs/")
    parser.add_argument("--combined", default=None,
                        help="Path to combined_results JSON (legacy format)")
    args = parser.parse_args()

    if args.combined:
        data = load_from_combined(args.combined)
        if data:
            print(f"Loaded combined results: {args.combined}\n")
            analyze_combined(data)
        else:
            print("No combined results found.")
        return

    # Prefer io_logs, fall back to combined
    data = load_from_io_logs(args.results_dir)
    if data:
        print(f"Loaded from io_logs in {args.results_dir}\n")
        analyze_io_logs(data)
    else:
        combined = load_from_combined()

        if combined:
            print("Loaded from combined results (legacy)\n")
            analyze_combined(combined)
        else:
            print("No results found. Run the experiment first.")


if __name__ == "__main__":
    main()
