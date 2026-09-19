import os
import ast
import random
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator, Union

import pandas as pd
import torch
from tqdm import tqdm, trange

from embedplan.encoders import *
from embedplan.prompts import create_prompt
from embedplan.config import Config

domain_names = Config.domain_names

DATA_DIR = Config.data_states_path


def get_encoder(name: str):
    """Factory function to get the appropriate encoder."""
    name_lower = name.lower()
    if name_lower == 'gemini':
        return GeminiEncoder()
    elif 'vertex' in name_lower:
        return VertexEncoder(model_name=name)
    elif 'llama' in name_lower or 'llama3' in name_lower or 'qwen' in name_lower:
        # Accepts model names like "llama3", "llama-3", "meta-llama/Llama-3.1-8B-Instruct", etc.
        return Llama3ModelEncoder(model_name=name)
    elif 'text-embedding' in name_lower or name_lower == 'openai':
        # Handle OpenAI embedding models
        return OpenAIEncoder()
    elif 'voyage' in name_lower:
        # Handle Voyage AI embedding models
        from text_embeddings import VoyageEncoder
        return VoyageEncoder()
    elif 'sentence-transformers' in name_lower or 'all-mpnet' in name_lower or 'all-minilm' in name_lower or 'bge' in name_lower:
        # Handle sentence-transformers models
        return SentenceTransformerEncoder(model_name=name)
    else:
        raise ValueError(f"Unknown encoder: {name}")


def check_for_checkpoints(domain_name: str, base_dirs: Dict[str, Path]) -> Tuple[torch.Tensor, Dict[str, int], int]:
    """Check if checkpoints exist and load them if available."""
    embedding_path = base_dirs['embeddings'] / f"{domain_name}.pt"
    prompt_to_index_path = base_dirs['embeddings'] / f"{domain_name}_prompt_to_index.pkl"

    embeddings = torch.tensor([])
    prompt_to_index = {}
    current_size = 0

    if embedding_path.exists() and prompt_to_index_path.exists():
        print(f"🔁 Resuming from checkpoint for domain: {domain_name}")
        try:
            embeddings = torch.load(embedding_path)
            with open(prompt_to_index_path, "rb") as f:
                prompt_to_index = pickle.load(f)
            # Set current_size based on number of encoded prompts
            current_size = len(prompt_to_index)
        except EOFError:
            print(f"⚠️ Warning: Checkpoint files for {domain_name} are empty or corrupted. Starting fresh.")
            embeddings = torch.tensor([])
            prompt_to_index = {}
            current_size = 0

    return embeddings, prompt_to_index, current_size


def save_checkpoints(domain_name: str, base_dirs: Dict[str, Path],
                     embeddings: torch.Tensor, prompt_to_index: Dict[str, int],
                     values: Dict = None, values_path: Path = None,
                     values_modified: bool = False) -> None:
    """Save checkpoints to disk. Always save the full tensor."""
    embedding_path = base_dirs['embeddings'] / f"{domain_name}.pt"
    prompt_to_index_path = base_dirs['embeddings'] / f"{domain_name}_prompt_to_index.pkl"

    # Save the full tensor without slicing
    tmp_path = str(embedding_path) + ".tmp"
    torch.save(embeddings, tmp_path)
    os.replace(tmp_path, embedding_path)

    with open(prompt_to_index_path, "wb") as f:
        pickle.dump(prompt_to_index, f)

    # Save values if modified
    if values_modified and values and values_path:
        with open(values_path, "wb") as f:
            pickle.dump(values, f)
        print(
            f"📝 Updated values dictionary saved with {sum(1 for x in values.get('alt_state_description', []) if x)} alternative state descriptions.")

    # print(f"💾 Checkpoint saved for domain {domain_name}: {len(prompt_to_index)} prompts encoded.")


def get_embedding_dim(encoder, sample_text="This is a sample text to determine embedding dimension"):
    """Determine the embedding dimension from the encoder."""
    try:
        # Try to get a sample embedding
        sample_embedding = encoder([sample_text])
        if isinstance(sample_embedding, list) and len(sample_embedding) > 0:
            return len(sample_embedding[0])
        return None
    except:
        return None

def process_domain(domain_name: str, args: argparse.Namespace,
                   encoder, base_dirs: Dict[str, Path]) -> None:
    """Process a single domain, handling all the encoding and checkpointing."""
    # Setup paths
    values_path = Config.data_states_path / f"{domain_name}-test_values.pkl"
    df_factorized, values = Config.read_factorized(domain_name)

    # Check for existing checkpoints
    embeddings, prompt_to_index, current_size = check_for_checkpoints(domain_name, base_dirs)

    # Generate all prompts and find unique ones
    all_prompts_set = set()
    values_modified_during_prompt_creation = False
    print(f"Generating prompts for {domain_name}...")
    for idx in tqdm(df_factorized.index, desc="Generating prompts"):
        row = df_factorized.loc[idx]
        prompt, modified = create_prompt(row, args.text_type, values)
        if modified:
            values_modified_during_prompt_creation = True
        all_prompts_set.add(prompt)

    unique_prompts = list(all_prompts_set)
    prompts_to_encode = [p for p in unique_prompts if p not in prompt_to_index]

    if not prompts_to_encode:
        print(f"✅ All {len(unique_prompts)} unique prompts already encoded for domain {domain_name}")
        if values_modified_during_prompt_creation:
             save_checkpoints(domain_name, base_dirs, embeddings, prompt_to_index, values, values_path, True)
        return

    print(f"Found {len(unique_prompts)} unique prompts, {len(prompts_to_encode)} new prompts to encode.")

    # Pre-allocate or resize tensor for all unique prompts
    embedding_dim = None
    if embeddings.numel() > 0:
        embedding_dim = embeddings.shape[1]
    else:
        embedding_dim = get_embedding_dim(encoder)

    total_prompts = len(unique_prompts)
    if embedding_dim:
        if embeddings.shape[0] < total_prompts:
            new_embeddings = torch.zeros(total_prompts, embedding_dim)
            if current_size > 0:
                new_embeddings[:current_size] = embeddings[:current_size]
            embeddings = new_embeddings
    tensor_allocated = embeddings.numel() > 0

    # Process the new prompts in batches
    values_modified_in_batch = False

    num_batches = (len(prompts_to_encode) + args.batch_size - 1) // args.batch_size
    batch_iterator = trange(num_batches, desc=f"Encoding prompts for {domain_name}")

    for batch_idx in batch_iterator:
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + args.batch_size
        batch_prompts = prompts_to_encode[start_idx:end_idx]

        # Encode this batch
        if hasattr(encoder, "__call__"):
            try:
                batch_embeddings = encoder(batch_prompts, batch_size=len(batch_prompts))
            except TypeError:
                batch_embeddings = encoder(batch_prompts)
        else:
            batch_embeddings = encoder(batch_prompts)

        # Filter out None values from embeddings and corresponding prompts
        valid_items = [(p, emb) for p, emb in zip(batch_prompts, batch_embeddings) if emb is not None]

        if len(valid_items) < len(batch_prompts):
            print(f"Skipped {len(batch_prompts) - len(valid_items)} prompts (e.g., too long).")

        if valid_items:
            valid_prompts, valid_embeddings = zip(*valid_items)

            if not tensor_allocated:
                embedding_dim = len(valid_embeddings[0])
                embeddings = torch.zeros(total_prompts, embedding_dim)
                tensor_allocated = True

            try:
                batch_tensor = torch.tensor(valid_embeddings)
            except:
                batch_tensor = torch.stack([torch.tensor(emb) for emb in valid_embeddings])
            batch_size = batch_tensor.shape[0]

            if current_size + batch_size > embeddings.shape[0]:
                new_size = max(embeddings.shape[0] * 2, current_size + batch_size, total_prompts)
                new_embeddings = torch.zeros(new_size, embeddings.shape[1])
                new_embeddings[:current_size] = embeddings[:current_size]
                embeddings = new_embeddings

            embeddings[current_size:current_size + batch_size] = batch_tensor

            for idx, prompt in enumerate(valid_prompts):
                prompt_to_index[prompt] = current_size + idx

            current_size += batch_size

        # Save checkpoint every 300 iterations or if values were modified
        if (batch_idx + 1) % 300 == 0 or values_modified_during_prompt_creation or values_modified_in_batch:
            save_checkpoints(domain_name, base_dirs, embeddings[:current_size], prompt_to_index, values, values_path, values_modified_during_prompt_creation or values_modified_in_batch)
            values_modified_during_prompt_creation = False # Reset after saving
            values_modified_in_batch = False

    # Final save
    final_embeddings = embeddings[:current_size]
    save_checkpoints(domain_name, base_dirs, final_embeddings, prompt_to_index, values, values_path, values_modified_during_prompt_creation or values_modified_in_batch)
    print(f"✅ Finished domain {domain_name}: {len(prompt_to_index)} unique prompts encoded.")


def main():
    parser = argparse.ArgumentParser("Batch encoding of domain texts")
    parser.add_argument("--embeddings_model_name", type=str, default="text-embedding-3-large")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--text_type", type=str, default='natural', choices=["pddl", "natural", "rephrased"])
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--domains", type=str, nargs='+', default=None, help="Specific domains to process. If not provided, processes all domains.")
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Custom embeddings directory (for Llama3-70B states).")
    args = parser.parse_args()

    # Map text type to internal representation
    column_map = {
        "pddl": "pddl",
        "natural": "original",
        "rephrased": "alt",
    }
    args.text_type = column_map[args.text_type]

    # Initialize encoder only if not in generation-only mode
    encoder = get_encoder(args.embeddings_model_name)

    # Setup directories
    if Config.use_sample:
        embeddings_base_dir = DATA_DIR.parent / "sample_embeddings" / args.embeddings_model_name / args.text_type
        if args.text_type == "pddl":
            prompts_base_dir = DATA_DIR.parent / "sample_embeddings/prompts" / "meta-llama/Llama-3.1-8B-Instruct" / args.text_type
        else:
            prompts_base_dir = DATA_DIR.parent / "sample_embeddings/prompts" / "gemini" / args.text_type
    else:
        embeddings_base_dir = DATA_DIR.parent / "full_embeddings" / args.embeddings_model_name / args.text_type
        if args.text_type == "pddl":
            prompts_base_dir = DATA_DIR.parent / "full_embeddings/prompts" / "meta-llama/Llama-3.1-8B-Instruct" / args.text_type
        else:
            prompts_base_dir = DATA_DIR.parent / "full_embeddings/prompts" / "gemini" / args.text_type

    base_dirs = {
        'embeddings': embeddings_base_dir,
        'prompts': prompts_base_dir
    }

    # Create directories if they don't exist
    for directory in base_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    # Process each domain
    domains_to_process = args.domains or domain_names
    for domain_name in domains_to_process:
        print(f"🔍 Processing domain: {domain_name}")
        process_domain(domain_name, args, encoder, base_dirs)


if __name__ == '__main__':
    main()
