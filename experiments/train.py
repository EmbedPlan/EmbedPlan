import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Callable

import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import wandb

from embedplan.data import FactorizedTripletDataset, ProblemGroupedBatchSampler, grouped_split_by_problem, grouped_split_by_plan
from embedplan.evaluation import eval_action_disambiguation, evaluate_hit_across_states
from embedplan.models import *
from embedplan.config import Config, load_config
from embedplan.losses import compute_infonce_loss
from embedplan.utils import EvalConfig, fix_seeds, worker_init_fn


def train_loop(model: nn.Module, train_loader: DataLoader, valid_loader: Optional[DataLoader], device: torch.device,
               epochs: int = 20, lr: float = 2e-3, eval_cfg: Optional[EvalConfig] = None,
               tau: float = 0.1, config: Optional[Dict] = None, use_wandb: bool = False,
               save_path: Optional[Path] = None, action_contrastive_weight: float = 0.0,
               callback: Optional[Callable[[nn.Module, int], None]] = None) -> Dict[str, float]:
    eval_cfg = eval_cfg or EvalConfig()
    training_cfg = config.get('training', {}) if config else {}
    wandb_cfg = config.get('wandb', {}) if config else {}
    patience = training_cfg.get('patience', float('inf'))
    ckpt_freq = training_cfg.get('checkpoint_freq', 100)
    eval_start = training_cfg.get('eval_start_epoch', 50)
    log_freq = wandb_cfg.get('log_freq', 1)

    # Handle eval-only mode (epochs=0)
    if epochs == 0:
        print("Running evaluation on untrained model (epochs=0)...")
        model.eval()
        with torch.no_grad():
            metrics_hit = evaluate_hit_across_states(model, valid_loader, device, eval_cfg)
            metrics_acc = eval_action_disambiguation(model, valid_loader, device, eval_cfg.topk)
            metrics_plan = evaluate_plan_grouped(model, valid_loader, device, k=5)
            metrics = {**metrics_hit, **metrics_acc, **metrics_plan}

        print(f"Untrained model metrics: {metrics}")

        if use_wandb:
            wandb.log({f"untrained/{k}": v for k, v in metrics.items()})

        # if save_path:
        #     torch.save({"metrics": metrics, "model_state_dict": model.state_dict()}, save_path)

        return metrics

    metrics, tbar, shown = {}, trange(1, epochs + 1), ""
    best_metrics, best_hit5 = {}, 0.0
    epochs_without_improvement = 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup = min(10, epochs // 10)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup)

    for epoch in tbar:
        model.train()
        tot_loss, tot_correct, seen = 0.0, 0, 0

        for batch in train_loader:
            s, a, sp = batch["s_emb"].to(device), batch["a_emb"].to(device), batch["sp_emb"].to(device)
            bs = s.size(0)
            pred, s_proj, a_proj = model(s, a, return_projections=True)
            sp_proj = model.state_projection_head(sp)
            loss = compute_infonce_loss(pred, sp_proj, tau)
            if action_contrastive_weight > 0.0:
                n_samples = max(1, int(bs ** 0.5))
                if n_samples > 1:
                    sample_idxs = torch.randperm(bs, device=device)[:n_samples]
                    s_sampled, a_sampled, sp_sampled = s[sample_idxs], a[sample_idxs], sp[sample_idxs]
                    s_repeated = s_sampled.repeat_interleave(n_samples, dim=0)
                    a_tiled = a_sampled.repeat(n_samples, 1)

                    preds_action = model(s_repeated, a_tiled)
                    preds_action = preds_action.view(n_samples, n_samples, -1)

                    preds_norm = F.normalize(preds_action, dim=-1)
                    sp_norm = F.normalize(model.state_projection_head(sp_sampled), dim=-1)  # (N, Dim)

                    # 2. Compute similarity between EVERY action outcome and the TRUE next state
                    # We want preds_norm[i, j] to match sp_norm[i] ONLY when i == j
                    # dot product: (N, N, D) * (N, 1, D) -> (N, N)
                    logits = (preds_norm * sp_norm.unsqueeze(1)).sum(dim=-1) / tau

                    # 3. The label for state i is action i (the diagonal)
                    labels = torch.arange(n_samples, device=device)

                    action_loss = F.cross_entropy(logits, labels)
                    loss = loss + action_contrastive_weight * action_loss
            scores = F.normalize(pred, dim=-1) @ F.normalize(sp_proj, dim=-1).T
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * bs
            seen += bs
            with torch.no_grad():
                tot_correct += (scores.argmax(1) == torch.arange(bs, device=pred.device)).sum().item()

        if scheduler and epoch <= warmup:
            scheduler.step()

        avg_loss, avg_acc = tot_loss / max(1, seen), tot_correct / max(1, seen)
        tbar.set_description(f"epoch {epoch:03d} | loss {avg_loss:.7f}, acc {avg_acc:.4f} | val {shown}")

        if use_wandb and epoch % log_freq == 0:
            log_dict = {"epoch": epoch, "train/loss": avg_loss, "train/accuracy": avg_acc,
                        "train/lr": opt.param_groups[0]['lr']}
            wandb.log(log_dict)

        if (epoch % ckpt_freq == 0 or epoch == epochs) and epoch >= eval_start:
            metrics_plan = evaluate_plan_grouped(model, valid_loader, device, k=5)
            metrics_hit = evaluate_hit_across_states(model, valid_loader, device, eval_cfg)
            metrics_acc = eval_action_disambiguation(model, valid_loader, device, eval_cfg.topk)
            metrics = {**metrics_hit, **metrics_acc, **metrics_plan}
            if metrics.get("hit@5", 0) > best_hit5:
                best_hit5 = metrics["hit@5"]
                best_metrics = {f"best_{k}": v for k, v in metrics.items()}
                best_metrics["best_epoch"] = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += ckpt_freq
            if use_wandb:
                wandb.log({f"val/{k}": v for k, v in metrics.items()} | best_metrics | {"epoch": epoch})

            if epochs_without_improvement >= patience:
                break

        if callback:
            callback(model, epoch)

    return best_metrics if best_metrics else metrics


def eval_untrained_model(model: nn.Module, valid_loader: DataLoader, device: torch.device,
                         eval_cfg: Optional[EvalConfig] = None, use_wandb: bool = False) -> Dict[str, float]:
    """Evaluate an untrained (randomly initialized) model."""
    eval_cfg = eval_cfg or EvalConfig()
    print("Evaluating untrained model...")

    model.eval()
    with torch.no_grad():
        metrics_plan = evaluate_plan_grouped(model, valid_loader, device, k=5)
        metrics_hit = evaluate_hit_across_states(model, valid_loader, device, eval_cfg)
        metrics_acc = eval_action_disambiguation(model, valid_loader, device, eval_cfg.topk)
        metrics = {**metrics_hit, **metrics_acc, **metrics_plan}

    print(f"Untrained model metrics: {metrics}")

    if use_wandb:
        wandb.log({f"untrained/{k}": v for k, v in metrics.items()})

    return metrics


def evaluate_plan_grouped(model: nn.Module, loader: DataLoader, device: torch.device, k: int = 5) -> Dict[str, float]:
    """
    Computes Hit@K for each sample, then groups by plan_id to compute average Hit@K per plan,
    and finally returns the average of those plan averages.
    """
    model.eval()
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    # 1. Prepare all candidate state embeddings (global ranking)
    all_states = torch.tensor(dataset.state_embs, device=device)

    # Check if model has projection head and project candidates if so
    if hasattr(model, "state_projection_head"):
        all_states_proj = []
        chunk_size = 2048
        with torch.no_grad():
            for i in range(0, len(all_states), chunk_size):
                chunk = all_states[i:i+chunk_size]
                all_states_proj.append(model.state_projection_head(chunk))
        all_candidates = torch.cat(all_states_proj)
    else:
        all_candidates = all_states

    all_candidates = F.normalize(all_candidates, dim=-1)

    plan_hits = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            s = batch["s_emb"].to(device)
            a = batch["a_emb"].to(device)
            plan_ids = batch["plan_id"].cpu().numpy()
            target_idxs = batch["sp_emb_idx"].to(device)

            # Forward pass
            pred = model(s, a)
            pred = F.normalize(pred, dim=-1)

            # Compute similarity with all candidates: (B, N_states)
            scores = pred @ all_candidates.T

            # Get top-k indices
            _, topk_indices = torch.topk(scores, k, dim=1)

            # Check if target index is in top-k
            hits = (topk_indices == target_idxs.unsqueeze(1)).any(dim=1).cpu().numpy()

            for pid, hit in zip(plan_ids, hits):
                plan_hits[pid].append(float(hit))

    # Compute average Hit@K per plan
    plan_averages = [np.mean(hits) for hits in plan_hits.values()]
    plan_max = [float(np.min(hits) == 1) for hits in plan_hits.values()]

    # Compute macro average across plans
    macro_avg_hit = np.mean(plan_averages) if plan_averages else 0.0
    macro_max_hit = np.mean(plan_max) if plan_max else 0.0

    return {**{f"plan_avg_hit@{k}": macro_avg_hit}, **{f"plan_max_hit@{k}": macro_max_hit}}


def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="ferry")
    parser.add_argument("--all_domains", action="store_true")
    parser.add_argument("--train_domain", type=str, default=None, help="Domain to train on (if different from test domain)")
    parser.add_argument("--test_domain", type=str, default=None, help="Domain to test on (if different from train domain)")
    parser.add_argument("--save_prefix", type=str, default="results/training/job")
    parser.add_argument("--model_type", type=str, choices=["mlp", "hyper"], default="mlp")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--k_state_infonce", type=float, default=2)
    parser.add_argument("--action_contrastive_weight", type=float, default=0.0)
    parser.add_argument("--use_projection", action="store_true")
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--projection_layers", type=int, default=2)
    parser.add_argument("--split_type", type=str, choices=["random", "problem_grouped", "plan_grouped"], default="problem_grouped")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate untrained model without training")
    return parser.parse_args()


def main():
    args = arguments_parser()

    # If eval_only is set, override epochs to 0
    if args.eval_only:
        args.epochs = 0

    fix_seeds(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    wandb_cfg, training_cfg, results_cfg = config.get('wandb', {}), config.get('training', {}), config.get('results', {})
    use_wandb = wandb_cfg.get('enabled', True) and not args.no_wandb
    wandb_project = args.wandb_project or wandb_cfg.get('project', 'transition-function-prediction')

    # Handle cross-domain training
    if args.train_domain and args.test_domain:
        # Cross-domain mode: train on one domain, test on another
        train_domains = [args.train_domain]
        test_domains = [args.test_domain]
        cross_domain = True
    elif args.all_domains:
        train_domains = Config.domain_names
        test_domains = Config.domain_names
        cross_domain = False
    else:
        # Single domain mode
        train_domains = [args.domain]
        test_domains = [args.domain]
        cross_domain = False

    for train_domain, test_domain in zip(train_domains, test_domains):
        mode_str = "Evaluating untrained" if args.eval_only else "Training"
        if cross_domain:
            print(f"--- {mode_str}: train={train_domain}, test={test_domain} ---")
        else:
            print(f"--- {mode_str}: {train_domain} ---")

        save_prefix = args.save_prefix if args.save_prefix != "results/training/job" else results_cfg.get(
            'training_dir', 'results/training') + "/job"

        if args.eval_only:
            save_path = Path(f"{save_prefix}_untrained_{train_domain}.pt") if args.all_domains else Path(f"{save_prefix}_untrained.pt")
        elif cross_domain:
            save_path = Path(f"{save_prefix}_train_{train_domain}_test_{test_domain}.pt")
        else:
            save_path = Path(f"{save_prefix}_{train_domain}.pt") if args.all_domains else Path(f"{save_prefix}.pt")

        if save_path.exists():
            print(f"Found existing weights at {save_path}, running 1 epoch for wandb logging...")
            args.epochs = 1
            # Continue with normal flow - model will be loaded below if needed, or trained for 1 epoch

        if use_wandb:
            run_name = f"{train_domain}_{args.model_type}_tau{args.tau}"
            if cross_domain:
                run_name = f"train_{train_domain}_test_{test_domain}_{args.model_type}_tau{args.tau}"
            run_name += f"_proj{args.projection_dim}d{args.projection_layers}L" if args.use_projection else "_no-proj"
            if args.eval_only:
                run_name = f"untrained_{run_name}"
            tags = wandb_cfg.get('tags', []) + [train_domain, args.model_type]
            if cross_domain:
                tags.extend(["cross_domain", f"test_{test_domain}"])
            if args.use_projection:
                tags.append("projection")
            if args.eval_only:
                tags.append("untrained")
            wandb.init(project=wandb_project,
                       entity=wandb_cfg.get('entity'), name=run_name, config=vars(args), tags=tags,
                       notes=wandb_cfg.get('notes'), save_code=wandb_cfg.get('save_code', True), reinit=True)

        # Load training dataset
        train_dataset = FactorizedTripletDataset(domain=train_domain, model_name=args.model_name, text_type="original")
        state_dim, action_dim = train_dataset.state_embs.shape[1], train_dataset.action_embs.shape[1]

        # Load test dataset (could be different domain)
        if cross_domain:
            test_dataset = FactorizedTripletDataset(domain=test_domain, model_name=args.model_name, text_type="original")
            # Verify dimensions match
            assert test_dataset.state_embs.shape[1] == state_dim, f"State dim mismatch: {test_dataset.state_embs.shape[1]} != {state_dim}"
            assert test_dataset.action_embs.shape[1] == action_dim, f"Action dim mismatch: {test_dataset.action_embs.shape[1]} != {action_dim}"
        else:
            test_dataset = train_dataset

        if args.use_projection:
            s_proj = ProjectionHead(input_dim=state_dim, output_dim=args.projection_dim,
                                    n_layers=args.projection_layers)
            a_proj = ProjectionHead(input_dim=action_dim, output_dim=args.projection_dim,
                                    n_layers=args.projection_layers)
            trans = model_selection(args.projection_dim, args, args.projection_dim)
            model = ProjectedTransitionModel(s_proj, a_proj, trans)
        else:
            model = model_selection(action_dim, args, state_dim)

        model.to(device)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if args.split_type == "random":
            train_indices, _ = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
            # # Modified to split by plans instead of purely random indices
            # train_indices, valid_indices = grouped_split_by_plan(train_dataset, train_frac=0.8, seed=args.seed)
            train_subset = Subset(train_dataset, train_indices)

            if cross_domain:
                valid_indices = list(range(len(test_dataset)))
                valid_subset = Subset(test_dataset, valid_indices)
            else:
                valid_subset = Subset(test_dataset, valid_indices)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            valid_loader = DataLoader(valid_subset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        elif args.split_type == "plan_grouped":
            train_indices, valid_indices = grouped_split_by_plan(train_dataset, train_frac=0.8, seed=args.seed)
            train_subset = Subset(train_dataset, train_indices)

            if cross_domain:
                valid_indices = list(range(len(test_dataset)))
                valid_subset = Subset(test_dataset, valid_indices)
            else:
                valid_subset = Subset(test_dataset, valid_indices)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            valid_loader = DataLoader(valid_subset, batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        else:  # problem_grouped
            train_idxs, _, _ = grouped_split_by_problem(train_dataset, train_frac=1.0 if cross_domain else 0.8, seed=args.seed)
            train_bs = ProblemGroupedBatchSampler(train_dataset, batch_size=args.batch_size, indices=train_idxs, shuffle_problems=True, shuffle_within_problem=True, seed=args.seed)

            # For test dataset
            if cross_domain:
                # Use all problems from test domain for validation
                valid_idxs = list(range(len(test_dataset)))
            else:
                _, valid_idxs, _ = grouped_split_by_problem(test_dataset, train_frac=0.8, seed=args.seed)

            valid_bs = ProblemGroupedBatchSampler(test_dataset, batch_size=args.val_batch_size, indices=valid_idxs, shuffle_problems=False, shuffle_within_problem=False)
            train_loader = DataLoader(train_dataset, batch_sampler=train_bs, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
            valid_loader = DataLoader(test_dataset, batch_sampler=valid_bs, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

        eval_cfg = EvalConfig()

        # train_loop will handle epochs=0 case by skipping training and only running eval
        train_loop(model, train_loader, valid_loader, device, epochs=args.epochs, lr=args.lr,
                   eval_cfg=eval_cfg, tau=args.tau, config=config, use_wandb=use_wandb,
                   save_path=save_path, action_contrastive_weight=args.action_contrastive_weight)

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
