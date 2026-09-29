"""All arms, one protocol: the published evaluation, applied uniformly.

Every arm here is trained and scored identically -- same split, same head budget, same
pool construction, same worst-case tie-breaking, same best-epoch selection rule (see
embedplan.paper_protocol). That uniformity is the point: the earlier baseline tables
mixed two estimators and the resulting rankings were not comparable.

Arms:
  embed_<encoder>  frozen encoder + projection head + transition MLP (EmbedPlan)
  literals         binary bag over the symbolic literal vocabulary
  tfidf/bow/char   lexical featurisations of the state prompt
  context          prompt with the CURRENT STATE block deleted   [null control]
  random           fixed Gaussian vector per state               [null control]
  symbolic         lifted STRIPS induction (no head; scored separately)
  identity/offset_grounded/offset_lifted   non-learned floors

Best-epoch selection follows the published rule, which maximises the metric over epochs
on the *test* split. `final` is recorded alongside so the optimism is measurable.

Usage:
    python -m experiments.baseline_sweep --domains ferry logistics goldminer --seeds 0 1 2
    python -m experiments.baseline_sweep --arms literals tfidf --domains ferry --seeds 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embedplan import load_domain, make_split
from embedplan.baselines import (IdentityTransition, LiftedOffsetTransition,
                                 OffsetTransition)
from embedplan.config import ANALYSIS_DIR
from embedplan.finetune import TextBank
from embedplan.losses import compute_action_loss, compute_infonce_loss
from embedplan.models import ProjectionHead, ProjectedTransitionModel, TransitionMLP
from embedplan.paper_protocol import paper_hit
from embedplan.symbolic import (LiftedActionModel, schema_index, symbolic_frame)
from embedplan.utils import resolve_device
from experiments.tfidf_baseline import featurize

FEATURE_ARMS = ["literals", "tfidf", "bow", "char", "context", "random"]
FLOOR_ARMS = ["identity", "offset_grounded", "offset_lifted"]
OUT = ANALYSIS_DIR / "paper_protocol_sweep.json"


def train_with_best(model, S, A, tri, train_idx, valid_idx, ds, args, device, split, seed):
    """Train the head, evaluating every `eval_every` epochs and keeping the best Hit@5.

    Mirrors the published loop: the selection metric is computed on the test split.
    """
    s_i = torch.as_tensor(tri["s_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
    a_i = torch.as_tensor(tri["a_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
    p_i = torch.as_tensor(tri["sp_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
    groups = None
    if split != "random":
        by = {}
        for pos, pid in enumerate(tri["problem_idx"].to_numpy()[train_idx]):
            by.setdefault(int(pid), []).append(pos)
        groups = [np.array(v) for v in by.values()]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    n, B = len(train_idx), args.batch_size
    best, best_ep, hist = None, -1, []

    for ep in range(1, args.epochs + 1):
        model.train()
        if groups is None:
            perm = rng.permutation(n)
            batches = [perm[i:i + B] for i in range(0, n, B)]
        else:
            batches = []
            for g in groups:
                gg = g.copy(); rng.shuffle(gg)
                batches.extend([gg[i:i + B] for i in range(0, len(gg), B)])
            rng.shuffle(batches)
        for b in batches:
            if len(b) < 2:
                continue
            bi = torch.as_tensor(b, device=device, dtype=torch.long)
            s, a, sp = S[s_i[bi]], A[a_i[bi]], S[p_i[bi]]
            pred = model(s, a)
            sp_proj = model.state_projection_head(sp)
            loss = compute_infonce_loss(pred, sp_proj, args.tau)
            if args.action_weight > 0:
                m = min(max(2, int(len(b) ** 0.5)), len(b))
                loss = loss + args.action_weight * compute_action_loss(
                    model, s, a, sp_proj, args.tau, m, device)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            h = paper_hit(model, S, A, tri, ds, valid_idx, device, split, seed)
            hist.append({"epoch": ep, **{k: h[k] for k in ("hit@1", "hit@5", "hit@10")}})
            if best is None or h["hit@5"] > best["hit@5"]:
                best, best_ep = h, ep
    return best, best_ep, hist[-1] if hist else None, hist


def build_features(arm, bank, sf, ds, train_idx, args, seed):
    # The non-learned floors are defined *in the encoder's own space* (that is what they
    # are a floor for), so they take the raw embedding tables rather than a featurisation.
    if arm.startswith("embed_") or arm in FLOOR_ARMS:
        return np.asarray(ds.state_embs), np.asarray(ds.action_embs), {"features": arm}
    return featurize(arm, bank, sf, ds, train_idx, args.dim, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=["ferry", "logistics", "goldminer"])
    ap.add_argument("--splits", nargs="+", default=["problem_grouped", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+",
                    default=["embed_BAAI/bge-m3"] + FEATURE_ARMS + FLOOR_ARMS + ["symbolic"])
    ap.add_argument("--reference_encoder", default="BAAI/bge-m3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--action_weight", type=float, default=2.0)
    ap.add_argument("--projection_dim", type=int, default=128)
    ap.add_argument("--projection_layers", type=int, default=2)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else OUT
    res = json.loads(path.read_text()) if path.exists() else {}
    print(f"device={device}  writing {path}", flush=True)

    for domain in args.domains:
        ds, tri = load_domain(domain, args.reference_encoder, verbose=False)
        bank = TextBank(ds)
        sf = symbolic_frame(ds, tri)
        sch_map, sch_names = schema_index(ds.action_vocab)
        for split in args.splits:
            for seed in args.seeds:
                train_idx, valid_idx = make_split(ds, tri, split, seed)
                for arm in args.arms:
                    key = f"{domain}|{split}|{seed}|{arm}"
                    if key in res:
                        print(f"  skip {key}", flush=True); continue
                    t0 = time.time()

                    if arm == "symbolic":
                        sym = LiftedActionModel().fit(
                            [(sf["s_lits"].iat[i], sf["action"].iat[i], sf["sp_lits"].iat[i])
                             for i in train_idx])
                        sp_l = sf["sp_lits"].to_numpy(); s_l = sf["s_lits"].to_numpy()
                        acts = sf["action"].to_numpy()
                        exact = sum(1 for i in valid_idx
                                    if sym.predict(s_l[i], acts[i]) == sp_l[i])
                        res[key] = {"arm": arm, "domain": domain, "split": split, "seed": seed,
                                    "exact": exact / max(1, len(valid_idx)),
                                    "induction": sym.summary(),
                                    "seconds": round(time.time() - t0, 1)}
                        print(f"  {key:56s} exact={res[key]['exact']:.4f}", flush=True)
                        json.dump(res, open(path, "w"), indent=2, default=str); continue

                    S_np, A_np, info = build_features(arm, bank, sf, ds, train_idx, args, seed)
                    S = torch.as_tensor(S_np, dtype=torch.float32, device=device)
                    A = torch.as_tensor(A_np, dtype=torch.float32, device=device)

                    if arm in FLOOR_ARMS:
                        s_tr = torch.as_tensor(tri["s_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
                        a_tr = torch.as_tensor(tri["a_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
                        p_tr = torch.as_tensor(tri["sp_emb_idx"].to_numpy()[train_idx], device=device, dtype=torch.long)
                        if arm == "identity":
                            model, ai = IdentityTransition().to(device), False
                        elif arm == "offset_grounded":
                            model = OffsetTransition(A.shape[0], S.shape[1], device).fit(S, a_tr, s_tr, p_tr); ai = True
                        else:
                            model = LiftedOffsetTransition(sch_map, len(sch_names), S.shape[1], device); model.fit(S, a_tr, s_tr, p_tr); ai = True
                        h = paper_hit(model, S, A, tri, ds, valid_idx, device, split, seed,
                                      action_index=ai)
                        res[key] = {"arm": arm, "domain": domain, "split": split, "seed": seed,
                                    "best": h, "final": h, "best_epoch": 0,
                                    "featurizer": info, "seconds": round(time.time() - t0, 1)}
                    else:
                        torch.manual_seed(seed)
                        model = ProjectedTransitionModel(
                            ProjectionHead(S.shape[1], args.projection_dim, n_layers=args.projection_layers).to(device),
                            ProjectionHead(A.shape[1], args.projection_dim, n_layers=args.projection_layers).to(device),
                            TransitionMLP(args.projection_dim, args.projection_dim, hidden=args.hidden_size,
                                          n_layers=args.n_layers, use_layer_norm=True).to(device)).to(device)
                        best, best_ep, final, hist = train_with_best(
                            model, S, A, tri, train_idx, valid_idx, ds, args, device, split, seed)
                        res[key] = {"arm": arm, "domain": domain, "split": split, "seed": seed,
                                    "best": best, "final": final, "best_epoch": best_ep,
                                    "featurizer": info, "history": hist,
                                    "seconds": round(time.time() - t0, 1)}
                    b = res[key]["best"]
                    print(f"  {key:56s} best h@1={b['hit@1']:.4f} h@5={b['hit@5']:.4f} "
                          f"(ep {res[key]['best_epoch']}, {res[key]['seconds']:.0f}s)", flush=True)
                    json.dump(res, open(path, "w"), indent=2, default=str)
                    del S, A
                    torch.cuda.empty_cache()
    print(f"\nwrote {path}  ({len(res)} cells)")


if __name__ == "__main__":
    main()
