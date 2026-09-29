import os
import ast
import argparse
import pickle
from typing import Dict, List, Iterable

import torch
import pandas as pd
from tqdm import trange

# Reuse the project utilities and encoders
from embedplan.config import Config
from embedplan.encoders import Llama3ModelEncoder, SentenceTransformerEncoder

domain_names = Config.domain_names
DATA_DIR = Config.data_states_path


def get_encoder(name: str):
    """Factory function to get the appropriate encoder."""
    name_lower = name.lower()
    if 'llama' in name_lower or 'llama3' in name_lower or 'qwen' in name_lower:
        # Accepts model names like "llama3", "llama-3", "meta-llama/Llama-3.1-8B-Instruct", etc.
        return Llama3ModelEncoder(model_name=name)
    elif 'sentence-transformers' in name_lower or 'all-mpnet' in name_lower or 'all-minilm' in name_lower or 'bge' in name_lower:
        # Handle sentence-transformers models
        return SentenceTransformerEncoder(model_name=name)
    else:
        raise ValueError(f"Unknown encoder: {name}")


def parse_plan_item(item) -> List[str]:
    """
    Each values['plan'][i] is a string that needs ast.literal_eval.
    It can be:
      - List[str]                       -> actions
      - List[List[str]]                 -> take the first sublist
    """
    try:
        parsed = ast.literal_eval(item) if isinstance(item, str) else item
    except Exception:
        return []

    if isinstance(parsed, list):
        if not parsed:
            return []
        # If it's a list of lists, take the first list; if list of str, return as-is.
        if all(isinstance(x, str) for x in parsed):
            return parsed
        if all(isinstance(x, list) for x in parsed) and parsed and all(isinstance(y, str) for y in parsed[0]):
            return parsed[0]
    return []


def dedupe_ordered(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def encode_in_batches(encoder, texts: List[str], batch_size: int) -> torch.Tensor:
    """Encode texts into a single tensor (N, D)."""
    if not texts:
        return torch.empty(0)
    embeddings: List[List[float]] = []
    for i in trange(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            embs = encoder(batch, batch_size=batch_size)
        except TypeError:
            embs = encoder(batch)
        # filter None (if any)
        for e in embs:
            if e is not None:
                embeddings.append(e)
    return torch.stack([torch.tensor(emb) for emb in embeddings])


def process_domain(domain_name: str, model_name: str, batch_size: int, text_type: str) -> None:
    """
    For a given domain:
      - Load {domain}-test_values.pkl
      - Extract unique actions from the *first* plan list per item
      - Encode actions
      - Save embeddings and index map per domain & model
    """
    values_path = DATA_DIR / f"{domain_name}-test_values.pkl"
    if not values_path.exists():
        print(f"❌ Missing values for {domain_name}, skipping.")
        return

    values: Dict = pd.read_pickle(values_path)

    if "plan" not in values:
        print(f"❌ 'plan' key not in values for {domain_name}, skipping.")
        return

    # Collect actions from the first list of each plan entry
    all_actions: List[str] = []
    for item in values["plan"]:
        actions = parse_plan_item(item)
        if actions:
            all_actions.extend(actions)

    # Dedupe while preserving order
    unique_actions: List[str] = dedupe_ordered(all_actions)

    if not unique_actions:
        print(f"ℹ️ No actions found for {domain_name}.")
        return

    # Encode
    encoder = get_encoder(model_name)
    embs = encode_in_batches(encoder, unique_actions, batch_size=batch_size)
    if embs.ndim != 2 or embs.shape[0] != len(unique_actions):
        print(f"⚠️ Mismatch after encoding for {domain_name}. Got tensor of shape {tuple(embs.shape)}.")
        return

    # Save per domain & model & text_type (text_type is fixed to 'original')
    if Config.use_sample:
        base_dir = DATA_DIR.parent / "sample_embeddings_actions" / model_name / text_type
    else:
        base_dir = DATA_DIR.parent / "full_embeddings_actions" / model_name / text_type
    base_dir.mkdir(parents=True, exist_ok=True)

    emb_path = base_dir / f"{domain_name}_actions.pt"
    map_path = base_dir / f"{domain_name}_actions_index.pkl"

    tmp_path = str(emb_path) + ".tmp"
    torch.save(embs, tmp_path)
    os.replace(tmp_path, emb_path)

    with open(map_path, "wb") as f:
        pickle.dump(
            {
                "actions": unique_actions,  # index -> action string
                "model_name": model_name,
                "domain": domain_name,
                "text_type": text_type,
            },
            f,
        )

    print(f"✅ Saved {len(unique_actions)} action embeddings for {domain_name} at {emb_path}")


def main():
    parser = argparse.ArgumentParser("Encode unique actions from plan values")
    parser.add_argument("--embeddings_model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Embedding model to use (the same one that encoded the states)")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Specific domains to process. If not provided, processes all domains."
    )
    # Text type fixed to 'original' as requested, but keep arg for consistency/overrides
    parser.add_argument("--text_type", type=str, default="original", choices=["original"])
    args = parser.parse_args()

    domains_to_process = args.domains or domain_names
    for domain in domains_to_process:
        print(f"🔍 Domain: {domain}")
        process_domain(domain, args.embeddings_model_name, args.batch_size, args.text_type)


if __name__ == "__main__":
    main()
