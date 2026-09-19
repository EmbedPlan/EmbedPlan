"""Paths, domain list, and the factorized-data / embedding loaders.

The data root defaults to ./data next to the repo but can be overridden with
EMBEDPLAN_DATA (useful on the cluster, where the same code runs against a
scratch mount).
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = Path(os.environ.get("EMBEDPLAN_DATA", REPO_ROOT / "data"))

RESULTS_ROOT = Path(os.environ.get("EMBEDPLAN_RESULTS", REPO_ROOT / "results"))
REBUTTAL_DIR = RESULTS_ROOT / "rebuttal"      # experiments run during the NeurIPS discussion
ANALYSIS_DIR = RESULTS_ROOT / "analysis"      # floors, variance audit, scoring-rule comparisons
CURVE_DIR = RESULTS_ROOT / "problem_curve"    # problem-count learning curve

DOMAINS = [
    "blocksworld", "depot", "ferry", "floortile", "goldminer",
    "grid", "logistics", "rovers", "satellite",
]

DEFAULT_ENCODER = "meta-llama/Llama-3.3-70B-Instruct"


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


class Config:
    """Data locations. Kept as a class of classproperties for backwards
    compatibility with the scripts and checkpoints written before the refactor."""

    domain_names = DOMAINS
    use_sample = False
    data_path = DEFAULT_DATA_PATH

    @classproperty
    def data_states_path(cls) -> Path:
        return cls.data_path / ("sample_df_pkls_factorized" if cls.use_sample else "original_df_pkls_factorized")

    @classproperty
    def base_state_embeddings_path(cls) -> Path:
        return cls.data_path / ("sample_embeddings" if cls.use_sample else "full_embeddings")

    @classproperty
    def base_action_embeddings_path(cls) -> Path:
        return cls.data_path / ("sample_embeddings_actions" if cls.use_sample else "full_embeddings_actions")

    @classmethod
    def read_factorized(cls, domain: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        df_path = cls.data_states_path / f"{domain}-test.pkl"
        values_path = cls.data_states_path / f"{domain}-test_values.pkl"
        if not df_path.exists() or not values_path.exists():
            raise FileNotFoundError(f"factorized data missing for '{domain}' under {cls.data_states_path}")
        return pd.read_pickle(df_path), pd.read_pickle(values_path)

    @classmethod
    def load_embeddings(cls, model: str, text_type: str, domain: str,
                        states_or_actions: str = "states"):
        """Returns (embeddings ndarray, index map). States are keyed by prompt text;
        actions by the action-string vocabulary."""
        if states_or_actions == "states":
            base = cls.base_state_embeddings_path / model / text_type
            emb_path, idx_path = base / f"{domain}.pt", base / f"{domain}_prompt_to_index.pkl"
        else:
            base = cls.base_action_embeddings_path / model / text_type
            emb_path, idx_path = base / f"{domain}_actions.pt", base / f"{domain}_actions_index.pkl"

        if not (emb_path.exists() and idx_path.exists()):
            raise FileNotFoundError(f"{states_or_actions} embeddings missing for '{domain}' at {emb_path}")

        embeddings = torch.load(str(emb_path), map_location="cpu")
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        with open(idx_path, "rb") as f:
            indices = pickle.load(f)
        return embeddings, indices

    @classmethod
    def load_state_embeddings(cls, model: str, text_type: str, domain: str):
        return cls.load_embeddings(model, text_type, domain, states_or_actions="states")


def load_config(path: str = "configs/config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)
