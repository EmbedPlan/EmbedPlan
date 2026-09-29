"""Shared fixtures.

The repo root goes on sys.path so `tools.make_toy_domain` imports under a bare `pytest`
as well as `python -m pytest`. The toy domain is written once per session into a tmp
directory; `toy_domain` points Config at it for the duration of one test.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def toy_root(tmp_path_factory):
    from tools.make_toy_domain import write_toy_domain

    root = tmp_path_factory.mktemp("embedplan_data")
    write_toy_domain(root)
    return root


@pytest.fixture
def toy_domain(toy_root, monkeypatch):
    """(dataset, triplet table) for the toy domain, loaded through the real loader."""
    from embedplan.config import Config
    from embedplan.data import load_domain
    from tools.make_toy_domain import ENCODER

    monkeypatch.setattr(Config, "data_path", toy_root)
    return load_domain("toy", model_name=ENCODER, verbose=False)
