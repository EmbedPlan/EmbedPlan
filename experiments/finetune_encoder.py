"""Frozen vs LoRA-finetuned encoder, on long-context encoders only.

Two arms, identical in every other respect:

  --lora_rank 0   encoder frozen, only projection heads + transition train.
                  This is EmbedPlan as published, re-run through the fine-tuning
                  code path so the comparison shares one implementation.
  --lora_rank 16  encoder adapted jointly with the rest.

Both are evaluated by re-encoding the entire state pool with the arm's own
encoder and then running the standard retrieval metrics, so the frozen arm also
serves as a check that re-encoding reproduces the cached embeddings.

Encoders are restricted to those whose context window exceeds the prompt length
(state prompts average ~500 tokens, ferry p95 = 1019). all-mpnet-base-v2 caps at
384 and is deliberately excluded: fine-tuning it would be confounded with the
model simply seeing more of its input.

Usage:
    python -m experiments.finetune_encoder --domain ferry --split problem_grouped --lora_rank 0
    python -m experiments.finetune_encoder --domain ferry --split problem_grouped --lora_rank 16
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from embedplan import load_domain, make_split, train_transition
from embedplan.data import build_trajectories
from embedplan.rollout import rollout
from embedplan.config import RESULTS_ROOT
from embedplan.evaluation import matched_pool_eval, pool_sweep
from embedplan.finetune import JointModel, TextBank, TunableEncoder, train_joint
from embedplan.models import (AnchoredTransitionMLP, ProjectionHead,
                              ProjectedTransitionModel, TransitionMLP)
from embedplan.scoring import ABSOLUTE, DELTA, project_pool
from embedplan.utils import resolve_device

OUT_DIR = RESULTS_ROOT / "finetune"

# context window >= prompt length; mpnet (384) is excluded on purpose
LONG_CONTEXT = {
    "BAAI/bge-m3": 8192,
    "Qwen/Qwen2.5-7B-Instruct": 32768,
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
}


def _transition(args):
    """Transition net for the head. 'anchored' starts at the identity predictor and has
    no output LayerNorm, so its output radius is free — which matters under DELTA, where
    a prediction pinned to a fixed norm cannot land on the state manifold."""
    kind = AnchoredTransitionMLP if getattr(args, "transition", "mlp") == "anchored" else TransitionMLP
    return kind(args.projection_dim, args.projection_dim, hidden=args.hidden_size,
                n_layers=args.n_layers, use_layer_norm=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="ferry")
    ap.add_argument("--split", default="problem_grouped", choices=["random", "problem_grouped"])
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--lora_rank", type=int, default=16, help="0 = frozen control arm")
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="projection + transition")
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--action_weight", type=float, default=2.0)
    ap.add_argument("--projection_dim", type=int, default=128)
    ap.add_argument("--projection_layers", type=int, default=2)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, -1])
    ap.add_argument("--max_train", type=int, default=0, help="subsample training triplets (0 = all)")
    # head stage: identical budget for both arms, matching the published pipeline
    ap.add_argument("--head_epochs", type=int, default=400)
    ap.add_argument("--head_batch_size", type=int, default=128)
    ap.add_argument("--head_lr", type=float, default=4e-5)
    ap.add_argument("--warmup_head_epochs", type=int, default=0,
                    help="train the head in the unadapted space before unfreezing the encoder "
                         "(0 = cold joint start, the original schedule)")
    ap.add_argument("--max_trajs", type=int, default=300)
    ap.add_argument("--loss_space", default="absolute", choices=["absolute", "delta"],
                    help="geometry the InfoNCE objective lives in; delta also switches "
                         "eval scoring to DELTA so training and scoring agree")
    ap.add_argument("--transition", default="mlp", choices=["mlp", "anchored"])
    ap.add_argument("--save", action="store_true", help="persist head, LoRA adapter and re-encoded tables")
    args = ap.parse_args()

    if args.encoder not in LONG_CONTEXT:
        raise SystemExit(f"{args.encoder} is not in the long-context allowlist {list(LONG_CONTEXT)}. "
                         "Short-context encoders confound fine-tuning with truncation.")

    torch.manual_seed(args.seed)
    device = resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arm = "frozen" if args.lora_rank == 0 else f"lora{args.lora_rank}"
    if args.lora_rank > 0 and args.warmup_head_epochs > 0:
        arm += f"warm{args.warmup_head_epochs}"
    if args.loss_space != "absolute":
        arm += f"_{args.loss_space}"
    if args.transition != "mlp":
        arm += f"_{args.transition}"
    tag = f"{args.domain}_{args.split}_{args.encoder.split('/')[-1]}_{arm}_seed{args.seed}"
    print(f"=== {tag} ===", flush=True)

    t0 = time.time()
    # the cached table is only used for its shape and the prompt<->row mapping
    ds, tri = load_domain(args.domain, args.encoder, verbose=False)
    bank = TextBank(ds)
    train_idx, valid_idx = make_split(ds, tri, args.split, args.seed)
    if args.max_train and len(train_idx) > args.max_train:
        import numpy as np
        train_idx = np.random.default_rng(args.seed).choice(
            train_idx, args.max_train, replace=False).tolist()
    print(f"train {len(train_idx)}  test {len(valid_idx)}  "
          f"states {len(bank.state_texts)}  actions {len(bank.action_texts)}", flush=True)

    encoder = TunableEncoder(args.encoder, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                             max_length=args.max_length, device=device,
                             gradient_checkpointing=args.lora_rank > 0)
    d = encoder.hidden_size
    model = JointModel(
        encoder,
        ProjectionHead(d, args.projection_dim, n_layers=args.projection_layers).to(device),
        ProjectionHead(d, args.projection_dim, n_layers=args.projection_layers).to(device),
        _transition(args).to(device),
    )

    # Stage 0 (optional): warm the head in the *unadapted* space before unfreezing.
    #
    # LoRA is initialized to a zero delta, so at this point the adapted encoder is
    # bit-identical to the frozen one and a head trained here transfers exactly.
    # Without it the joint stage optimizes the encoder against a head that is
    # nowhere near converged -- measured: joint-stage loss bottoms out around 2.05
    # while the same head reaches 0.59 once the space stops moving -- so the
    # adapter gets shaped to compensate for a weak transition function that is
    # then discarded. Warming first means the encoder's gradient comes from a
    # transition model that already knows how to move.
    if args.lora_rank > 0 and args.warmup_head_epochs > 0:
        print(f"[warmup] encoding pool with the unadapted encoder ...", flush=True)
        S0 = encoder.encode_all(bank.state_texts, args.eval_batch_size, desc="warmup states")
        A0 = encoder.encode_all(bank.action_texts, args.eval_batch_size, desc="warmup actions")
        warm = ProjectedTransitionModel(model.state_projection_head,
                                        model.action_projection_head,
                                        model.transition_model)
        warm_args = argparse.Namespace(**{**vars(args), "epochs": args.warmup_head_epochs,
                                          "batch_size": args.head_batch_size,
                                          "lr": args.head_lr, "dropout": 0.0})
        print(f"[warmup] training head {args.warmup_head_epochs} epochs in the frozen space ...", flush=True)
        train_transition(warm, S0, A0, tri, train_idx, warm_args, device)
        del S0, A0
        torch.cuda.empty_cache()

    if args.lora_rank > 0:
        train_joint(model, bank, tri, train_idx, args, device)
    else:
        print("[frozen arm] skipping joint stage; the encoder cannot move", flush=True)
    train_s = time.time() - t0

    print("re-encoding state pool with the resulting encoder ...", flush=True)
    S = encoder.encode_all(bank.state_texts, args.eval_batch_size, desc="states")
    A = encoder.encode_all(bank.action_texts, args.eval_batch_size, desc="actions")

    # Both arms now get an identically-budgeted head trained from scratch on their
    # own embedding table. Without this the frozen arm would be compared at a few
    # hundred steps against the published pipeline's 400 epochs, and any difference
    # would measure head training rather than the embedding space. With it, the
    # only thing that differs between arms is the space the head is fitted in.
    print(f"training head to convergence on the resulting space "
          f"({args.head_epochs} epochs, batch {args.head_batch_size}) ...", flush=True)
    head_args = argparse.Namespace(**{**vars(args), "epochs": args.head_epochs,
                                      "batch_size": args.head_batch_size,
                                      "lr": args.head_lr, "dropout": 0.0})
    evaluator = ProjectedTransitionModel(
        ProjectionHead(d, args.projection_dim, n_layers=args.projection_layers).to(device),
        ProjectionHead(d, args.projection_dim, n_layers=args.projection_layers).to(device),
        _transition(args).to(device),
    )
    train_transition(evaluator, S, A, tri, train_idx, head_args, device)
    evaluator.eval()
    pool = project_pool(evaluator, S)
    mode = DELTA if args.loss_space == "delta" else ABSOLUTE
    sweep = pool_sweep(evaluator, S, A, tri, valid_idx, pool, device, args.sizes,
                       args.seed, mode=mode)
    matched = matched_pool_eval(evaluator, S, A, tri, ds, valid_idx, device,
                                seed=args.seed, mode=mode)

    out = {"tag": tag, "arm": arm, "args": vars(args),
           "n_train": len(train_idx), "n_test": len(valid_idx),
           "encoder_dim": d, "pool_sweep": sweep, "matched_128": matched,
           "encoder_stage_seconds": round(train_s, 1)}

    # Multi-step rollout in the same space. Under extrapolation the published
    # models collapse here (step Hit@1 0.010 ferry / 0.038 logistics, exact 0.000),
    # so this is where an adapted encoder either pays off or does not.
    trajs = build_trajectories(tri, valid_idx, args.max_trajs, args.seed)
    if trajs:
        print(f"rolling out {len(trajs)} trajectories ...", flush=True)
        out["rollout"] = rollout(evaluator, S, A, trajs, F.normalize(pool, dim=-1),
                                 device, prefix_curve=True)
        r = out["rollout"]
        print(f"  rollout step_hit@1  tf={r['teacher_forced']['step_hit@1']:.3f}  "
              f"cl={r['closed_loop']['step_hit@1']:.3f}  "
              f"fr={r['free_running']['step_hit@1']:.3f}", flush=True)
        print(f"  rollout exact@5     tf={r['teacher_forced']['exact_hit@5']:.3f}  "
              f"cl={r['closed_loop']['exact_hit@5']:.3f}", flush=True)
    else:
        print("no usable test trajectories; skipping rollout", flush=True)

    if args.save:
        ckpt = OUT_DIR / f"ckpt_{tag}.pt"
        torch.save({"head": evaluator.state_dict(), "args": vars(args),
                    "S": S.half().cpu(), "A": A.half().cpu()}, ckpt)
        if args.lora_rank > 0:
            encoder.backbone.save_pretrained(OUT_DIR / f"lora_{tag}")
        print(f"saved artifacts to {ckpt}", flush=True)

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
