"""Small shared helpers with no better home."""

import random

import numpy as np
import torch


class EvalConfig:
    def __init__(self, topk: tuple = (1, 5, 10)):
        self.topk = topk


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def fix_seeds(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(prefer: str = "cuda") -> torch.device:
    return torch.device(prefer if torch.cuda.is_available() and prefer == "cuda" else "cpu")
