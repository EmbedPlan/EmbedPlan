"""Score a saved frozen-encoder checkpoint with the canonical evaluators.

`experiments/closed_loop.py` saves a checkpoint and reports the pool sweep, but not
matched_pool_eval — so the published Llama-3.3-70B runs in results/rebuttal/ cannot be
placed in the same table as the baselines without re-scoring them here.

This uses embedplan.evaluation.matched_pool_eval and pool_sweep directly, the same
functions experiments/symbolic_baseline.py and tfidf_baseline.py call, so every arm in
the comparison shares one estimator — same problem-grouped batches, same queries, same
best-case tie-breaking. (experiments/matched_pool.py has its own local implementation;
it is not reused here precisely to avoid two estimators in one table.)

Usage:
    python -m experiments.eval_checkpoint --tags ferry_problem_grouped_seed0 ...
    python -m experiments.eval_checkpoint --all-frozen
"""

import argparse
import json

import torch

from embedplan import build_model, load_domain, make_split
from embedplan.config import ANALYSIS_DIR, REBUTTAL_DIR
from embedplan.evaluation import matched_pool_eval, pool_sweep
from embedplan.scoring import project_pool
from embedplan.utils import resolve_device

DOMAINS = ["ferry", "logistics", "goldminer"]
SPLITS = ["problem_grouped", "random"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--all-frozen", action="store_true",
                    help=f"every {{domain}}_{{split}}_seed0 over {DOMAINS} x {SPLITS}")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, -1])
    ap.add_argument("--out", default="llama_frozen_eval.json")
    args = ap.parse_args()

    tags = args.tags or []
    if args.all_frozen or not tags:
        tags = [f"{d}_{s}_seed0" for d in DOMAINS for s in SPLITS]

    device = resolve_device()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / args.out
    out = json.loads(path.read_text()) if path.exists() else {}

    for tag in tags:
        ck_path = REBUTTAL_DIR / f"ckpt_{tag}.pt"
        if not ck_path.exists():
            print(f"  {tag:42s} NO CHECKPOINT — skipped", flush=True)
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        ck_args = argparse.Namespace(**ck["args"])
        ds, tri = load_domain(ck_args.domain, ck_args.model_name, verbose=False)
        S = torch.as_tensor(ds.state_embs, dtype=torch.float32, device=device)
        A = torch.as_tensor(ds.action_embs, dtype=torch.float32, device=device)
        _, valid_idx = make_split(ds, tri, ck_args.split, ck_args.seed)

        model = build_model(S.shape[1], A.shape[1], ck_args, device)
        model.load_state_dict(ck["model"])
        model.eval()

        pool = project_pool(model, S)
        rec = {
            "tag": tag, "encoder": ck_args.model_name, "domain": ck_args.domain,
            "split": ck_args.split, "seed": ck_args.seed, "encoder_dim": int(S.shape[1]),
            "n_test": int(len(valid_idx)), "n_states": int(S.shape[0]),
            "matched_128": matched_pool_eval(model, S, A, tri, ds, valid_idx, device,
                                             batch_size=args.batch_size, seed=ck_args.seed),
            "pool_sweep": pool_sweep(model, S, A, tri, valid_idx, pool, device,
                                     args.sizes, ck_args.seed),
        }
        out[tag] = rec
        m, f = rec["matched_128"], rec["pool_sweep"]["full"]
        print(f"  {tag:42s} matched h@1 {m['hit@1']:.4f} h@5 {m['hit@5']:.4f} | "
              f"full h@1 {f['hit@1']:.4f} h@5 {f['hit@5']:.4f}", flush=True)
        del S, A, pool, model
        torch.cuda.empty_cache()

    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}  ({len(out)} tags)")


if __name__ == "__main__":
    main()
