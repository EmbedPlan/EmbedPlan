"""Is the *LLM* part of the embedding load-bearing?

Same head, same split, same evaluation — only the encoder changes, from a 568M-param
multilingual transformer to a TF-IDF vectorizer with an SVD projection. If a bag of
word n-grams reaches the frozen BGE-M3 numbers, "LLM embedding space" is not what is
doing the work, and the paper's framing has to narrow accordingly.

This closes a hole the current setup leaves open. `experiments/finetune_encoder.py`
excludes all-mpnet-base-v2 because its 384-token window truncates ~500-token state
prompts, so short-context encoders were never compared. TF-IDF has no context window,
so that exclusion does not apply to it and the comparison is available for free.

Two featurizations:

  tfidf     word 1-2 grams over the state prompt / action string, TF-IDF weighted,
            then TruncatedSVD to `--dim` (default 1024, matching BGE-M3's width so the
            projection head has an identical parameter budget).
  literals  binary bag over the symbolic literal vocabulary, same SVD. Not a text
            baseline at all — it asks what a *structured* but non-lifted
            representation achieves, sitting between TF-IDF and the lifted operators
            of experiments/symbolic_baseline.py.
  bow       raw token counts, no IDF weighting. Is the weighting load-bearing?
  char      character 3-5 grams. Is word tokenization needed at all?
  context   TF-IDF over the prompt with the CURRENT STATE block DELETED. The state is
            the only part that varies within a problem, so anything this scores is
            achieved without seeing the state at all. Directly calibrates how much of a
            full-prompt embedding is shared boilerplate.
  random    a fixed random gaussian vector per state row. Carries zero information about
            the state; the true floor of the ladder.

Fitting is on training states only. The vectorizer and SVD never see a held-out
problem's states, which mirrors the frozen encoder's position (it never saw this
dataset) and avoids crediting the baseline with transductive access to the test pool.
A consequence worth reporting rather than hiding: object names introduced only by
held-out problems fall out of vocabulary, which is the lexical form of the same
grounding problem that sinks the per-grounded-action offset baseline.

Usage:
    python -m experiments.tfidf_baseline --domain ferry --split problem_grouped
    python -m experiments.tfidf_baseline --domain ferry --features literals
"""

import argparse
import json
import math
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from embedplan import load_domain, make_split, train_transition
from embedplan.config import RESULTS_ROOT
from embedplan.data import build_trajectories
from embedplan.evaluation import matched_pool_eval, pool_sweep
from embedplan.finetune import TextBank
from embedplan.models import ProjectionHead, ProjectedTransitionModel, TransitionMLP
from embedplan.rollout import rollout
from embedplan.scoring import project_pool
from embedplan.symbolic import symbolic_frame
from embedplan.utils import resolve_device

OUT_DIR = RESULTS_ROOT / "tfidf"


# TF-IDF is implemented here rather than pulled from sklearn on purpose: sklearn is
# absent from the sl_llada env that produced every BGE-M3 number, and running this arm
# under a different env would put a different torch build in the head-training path —
# which is the one thing this comparison has to hold fixed.

_WORD = re.compile(r"[a-z0-9_\-]+")
_STATE_HDR = "### CURRENT STATE ###"


def _strip_state(prompt):
    """Prompt with the CURRENT STATE block removed, problem and goal kept."""
    if _STATE_HDR not in prompt:
        return prompt
    head, rest = prompt.split(_STATE_HDR, 1)
    parts = rest.split("\n\n", 1)
    return head + (parts[1] if len(parts) > 1 else "")


def _char_ngrams(text, lo, hi):
    t = text.lower()
    return [t[i:i + n] for n in range(lo, hi + 1) for i in range(len(t) - n + 1)]


def _tokenize(text, ngram_max):
    toks = _WORD.findall(text.lower())
    grams = list(toks)
    for n in range(2, ngram_max + 1):
        grams.extend("_".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return grams


def _atoms(doc):
    """Pre-tokenized documents (the literal bags) split on whitespace only."""
    return doc.split()


def _fit_vocab(docs, tokenize, min_df):
    df = Counter()
    for d in docs:
        df.update(set(tokenize(d)))
    terms = sorted(t for t, c in df.items() if c >= min_df)
    n = max(1, len(docs))
    vocab = {t: i for i, t in enumerate(terms)}
    # sklearn's smooth_idf formulation, so the weighting is a known quantity
    idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms], dtype=np.float32)
    return vocab, idf


def _transform(docs, tokenize, vocab, idf, sublinear, use_idf):
    X = np.zeros((len(docs), max(1, len(vocab))), dtype=np.float32)
    for r, d in enumerate(docs):
        counts = Counter(tokenize(d))
        for t, c in counts.items():
            j = vocab.get(t)
            if j is None:
                continue
            X[r, j] = (1.0 + math.log(c)) if sublinear else float(c)
    if use_idf and len(vocab):
        X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    np.divide(X, norms, out=X, where=norms > 0)
    return X


def _maybe_svd(X, dim, seed):
    """Reduce to `dim` only when the vocabulary is wider than it. These prompts are
    templated, so the vocabulary is usually a few hundred terms and this is a no-op —
    the honest outcome is a narrower representation, not one padded to fake width."""
    if X.shape[1] <= dim:
        return X
    torch.manual_seed(seed)
    T = torch.as_tensor(X)
    U, S, _ = torch.svd_lowrank(T, q=min(dim, min(T.shape) - 1), niter=4)
    return (U * S).numpy().astype(np.float32)


def featurize(kind, bank, sf, ds, train_idx, dim, seed, min_df=2, ngram_max=2):
    """Returns (state_table, action_table, info). Vocabulary fitted on training rows only."""
    if kind in ("tfidf", "bow", "char", "context"):
        state_docs = list(bank.state_texts)
        if kind == "context":
            # drop the CURRENT STATE block; keep problem + goal. Everything that varies
            # within a problem lives in that block, so this measures the shared component.
            state_docs = [_strip_state(t) for t in state_docs]
        action_docs = list(bank.action_texts)
        if kind == "char":
            tokenize = lambda d: _char_ngrams(d, 3, 5)
            sublinear, use_idf = True, True
        else:
            tokenize = lambda d: _tokenize(d, ngram_max)
            sublinear, use_idf = (kind != "bow"), (kind != "bow")
    elif kind == "random":
        rs = np.random.default_rng(seed)
        S_dense = rs.standard_normal((len(bank.state_texts), min(dim, 512))).astype(np.float32)
        A_dense = rs.standard_normal((len(bank.action_texts), min(dim, 512))).astype(np.float32)
        S_dense /= np.maximum(np.linalg.norm(S_dense, axis=1, keepdims=True), 1e-12)
        A_dense /= np.maximum(np.linalg.norm(A_dense, axis=1, keepdims=True), 1e-12)
        return S_dense, A_dense, {"features": "random", "state_vocab": 0, "action_vocab": 0,
                                  "state_dim": int(S_dense.shape[1]),
                                  "action_dim": int(A_dense.shape[1]),
                                  "state_rows_all_zero": 0, "action_rows_all_zero": 0,
                                  "n_train_state_rows": 0, "n_train_action_rows": 0,
                                  "min_df": min_df, "ngram_max": ngram_max}
    elif kind == "literals":
        lit_of_row = {}
        for i in range(len(sf)):
            lit_of_row.setdefault(int(sf["s_emb_idx"].iat[i]), sf["s_lits"].iat[i])
            lit_of_row.setdefault(int(sf["sp_emb_idx"].iat[i]), sf["sp_lits"].iat[i])
        state_docs = [" ".join(sorted(l.replace(" ", "_") for l in lit_of_row.get(r, frozenset())))
                      for r in range(len(bank.state_texts))]
        action_docs = [a.strip("() ").replace(" ", "_") for a in bank.action_texts]
        tokenize, sublinear, use_idf = _atoms, False, False
    else:
        raise SystemExit(f"unknown --features {kind!r}")

    train_state_rows = sorted({int(sf["s_emb_idx"].iat[i]) for i in train_idx} |
                              {int(sf["sp_emb_idx"].iat[i]) for i in train_idx})
    train_action_rows = sorted({int(sf["a_idx"].iat[i]) for i in train_idx})

    s_vocab, s_idf = _fit_vocab([state_docs[r] for r in train_state_rows], tokenize, min_df)
    a_vocab, a_idf = _fit_vocab([action_docs[r] for r in train_action_rows], tokenize, 1)

    S_dense = _transform(state_docs, tokenize, s_vocab, s_idf, sublinear, use_idf)
    A_dense = _transform(action_docs, tokenize, a_vocab, a_idf, sublinear, use_idf)

    # Rows that came out all-zero are unrepresentable under a train-fitted vocabulary —
    # the lexical form of the grounding problem, so it gets reported, not hidden.
    zero_states = int((np.abs(S_dense).sum(1) == 0).sum())
    zero_actions = int((np.abs(A_dense).sum(1) == 0).sum())

    S_dense = _maybe_svd(S_dense, dim, seed)
    A_dense = _maybe_svd(A_dense, dim, seed)

    info = {"features": kind, "min_df": min_df, "ngram_max": ngram_max,
            "state_vocab": len(s_vocab), "action_vocab": len(a_vocab),
            "state_dim": int(S_dense.shape[1]), "action_dim": int(A_dense.shape[1]),
            "state_rows_all_zero": zero_states, "action_rows_all_zero": zero_actions,
            "n_train_state_rows": len(train_state_rows),
            "n_train_action_rows": len(train_action_rows)}
    return S_dense, A_dense, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="ferry")
    ap.add_argument("--split", default="problem_grouped", choices=["random", "problem_grouped"])
    ap.add_argument("--features", default="tfidf",
                    choices=["tfidf", "literals", "bow", "char", "context", "random"])
    ap.add_argument("--dim", type=int, default=1024, help="SVD width; 1024 matches BGE-M3")
    ap.add_argument("--reference_encoder", default="BAAI/bge-m3",
                    help="only used to load the cached table that defines row order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, -1])
    ap.add_argument("--max_trajs", type=int, default=300)
    # head budget identical to experiments/finetune_encoder.py
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--action_weight", type=float, default=2.0)
    ap.add_argument("--projection_dim", type=int, default=128)
    ap.add_argument("--projection_layers", type=int, default=2)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.domain}_{args.split}_{args.features}{args.dim}_seed{args.seed}"
    print(f"=== {tag} on {device} ===", flush=True)

    t0 = time.time()
    ds, tri = load_domain(args.domain, args.reference_encoder, verbose=False)
    bank = TextBank(ds)
    sf = symbolic_frame(ds, tri) if args.features == "literals" else tri.copy()
    if args.features != "literals":
        sf["s_lits"] = None
        sf["sp_lits"] = None
    train_idx, valid_idx = make_split(ds, tri, args.split, args.seed)
    print(f"train {len(train_idx)}  test {len(valid_idx)}", flush=True)

    S_np, A_np, info = featurize(args.features, bank, sf, ds, train_idx, args.dim, args.seed)
    print(f"featurized: {info}", flush=True)

    S = torch.as_tensor(S_np, dtype=torch.float32, device=device)
    A = torch.as_tensor(A_np, dtype=torch.float32, device=device)

    model = ProjectedTransitionModel(
        ProjectionHead(S.shape[1], args.projection_dim, n_layers=args.projection_layers).to(device),
        ProjectionHead(A.shape[1], args.projection_dim, n_layers=args.projection_layers).to(device),
        TransitionMLP(args.projection_dim, args.projection_dim, hidden=args.hidden_size,
                      n_layers=args.n_layers, use_layer_norm=True).to(device),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"head params {n_params/1e6:.2f}M; training {args.epochs} epochs ...", flush=True)

    train_transition(model, S, A, tri, train_idx, args, device)
    model.eval()

    pool = project_pool(model, S)
    sweep = pool_sweep(model, S, A, tri, valid_idx, pool, device, args.sizes, args.seed)
    matched = matched_pool_eval(model, S, A, tri, ds, valid_idx, device, seed=args.seed)

    out = {"tag": tag, "args": vars(args), "featurizer": info,
           "n_train": int(len(train_idx)), "n_test": int(len(valid_idx)),
           "head_params": int(n_params), "pool_sweep": sweep, "matched_128": matched}

    trajs = build_trajectories(tri, valid_idx, args.max_trajs, args.seed)
    if trajs:
        out["rollout"] = rollout(model, S, A, trajs, F.normalize(pool, dim=-1), device,
                                 prefix_curve=True)
        r = out["rollout"]
        print(f"  rollout step_hit@1  tf={r['teacher_forced']['step_hit@1']:.3f}  "
              f"cl={r['closed_loop']['step_hit@1']:.3f}", flush=True)

    out["total_seconds"] = round(time.time() - t0, 1)
    path = OUT_DIR / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\n  {'pool':>12s} {'hit@1':>9s} {'hit@5':>9s}")
    for k, v in sweep.items():
        print(f"  {k:>12s} {v['hit@1']:9.4f} {v['hit@5']:9.4f}")
    print(f"  {'matched_128':>12s} {matched['hit@1']:9.4f} {matched['hit@5']:9.4f}")
    print(f"\nwrote {path}  ({out['total_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
