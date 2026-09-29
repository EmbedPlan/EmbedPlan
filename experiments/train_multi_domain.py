import argparse
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from embedplan.data import FactorizedTripletDataset, ProblemGroupedBatchSampler, grouped_split_by_problem
from embedplan.evaluation import evaluate_hit_across_states
from embedplan.models import ProjectionHead, ProjectedTransitionModel, model_selection
from embedplan.config import Config, load_config
from embedplan.utils import EvalConfig, fix_seeds, worker_init_fn
from experiments.train import train_loop

try:
    import wandb
except ImportError:  # optional: only needed when W&B logging is enabled in the config
    wandb = None


def arguments_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_protocol", choices=["loo", "in_domain"], required=True)
    p.add_argument("--test_domains", nargs="+", default=[])
    p.add_argument("--save_prefix", default="results/training_multi/job")
    p.add_argument("--model_type", choices=["mlp", "hyper"], default="mlp")
    p.add_argument("--model_name", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_layer_norm", action="store_true")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--k_state_infonce", type=float, default=2)
    p.add_argument("--action_contrastive_weight", type=float, default=2.0)
    p.add_argument("--use_projection", action="store_true")
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--projection_layers", type=int, default=4)
    p.add_argument("--split_type", choices=["random", "problem_grouped"], default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--visualize_pca", action="store_true", help="Visualize transitions with PCA")
    p.add_argument("--dim_reduction", choices=["pca", "umap"], default="pca", help="Dimensionality reduction method")
    p.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained model for visualization")
    return p.parse_args()


def visualize_pca(model, loaders, device, save_dir, run_name, method="pca", epoch=None):
    model.eval()
    all_s, all_next_s, all_sp, domains = [], [], [], []
    csv_data = []

    target_domains = ["ferry", "blocksworld", "rovers"]

    with torch.no_grad():
        for domain, loader in loaders.items():
            if domain not in target_domains:
                continue
            batch = next(iter(loader))
            # Sample 5 transitions per domain
            idx = torch.randperm(batch["s_emb"].shape[0])[:5]
            s = batch["s_emb"][idx].to(device)
            a = batch["a_emb"][idx].to(device)
            sp = batch["sp_emb"][idx].to(device)

            if isinstance(model, ProjectedTransitionModel):
                pred_next, curr_s, _ = model(s, a, return_projections=True)
                curr_sp = model.state_projection_head(sp)
            else:
                curr_s = s
                pred_next = model(s, a)
                curr_sp = sp

            all_s.append(curr_s.cpu().numpy())
            all_next_s.append(pred_next.cpu().numpy())
            all_sp.append(curr_sp.cpu().numpy())
            domains.extend([domain] * curr_s.shape[0])

    # Concatenate lists of arrays first to ensure 2D array for PCA
    s_flat = np.concatenate(all_s, axis=0)
    next_s_flat = np.concatenate(all_next_s, axis=0)
    sp_flat = np.concatenate(all_sp, axis=0)

    X = np.concatenate([s_flat, next_s_flat, sp_flat], axis=0)

    if method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("Please install umap-learn: pip install umap-learn")
        reducer = umap.UMAP(n_components=2)
    else:
        reducer = PCA(n_components=2)

    X_reduced = reducer.fit_transform(X)

    n = len(domains)
    s_pca, next_s_pca, sp_pca = X_reduced[:n], X_reduced[n:2*n], X_reduced[2*n:]

    # Prepare visualization directory
    viz_dir = Path(save_dir) / "visualizations" / run_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_epoch_{epoch:03d}" if epoch is not None else "_final"

    # Save data to CSV
    for i in range(n):
        csv_data.append({
            'domain': domains[i],
            'transition_idx': i,
            f's_{method}_x': s_pca[i, 0],
            f's_{method}_y': s_pca[i, 1],
            f'pred_next_{method}_x': next_s_pca[i, 0],
            f'pred_next_{method}_y': next_s_pca[i, 1],
            f'actual_next_{method}_x': sp_pca[i, 0],
            f'actual_next_{method}_y': sp_pca[i, 1],
            'pred_error': np.linalg.norm(next_s_pca[i] - sp_pca[i])
        })

    csv_path = viz_dir / f"{method}_data{suffix}.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Saved {method.upper()} data to {csv_path}")

    unique_doms = sorted(list(set(domains)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_doms)))
    dom_map = dict(zip(unique_doms, colors))

    for show_sp in [True, False]:
        # ACL single-column format: 3.25 inches width, with appropriate height
        plt.figure(figsize=(3.25, 3.5), dpi=300)

        seen_labels = set()
        for i in range(n):
            d = domains[i]
            label = d if d not in seen_labels else None
            if label: seen_labels.add(d)

            # Draw arrows from s to pred_next (dashed)
            plt.plot([s_pca[i, 0], next_s_pca[i, 0]], [s_pca[i, 1], next_s_pca[i, 1]],
                     color=dom_map[d], alpha=0.5, linestyle="--", linewidth=0.8)

            if show_sp:
                # Draw arrows from s to sp (solid)
                plt.plot([s_pca[i, 0], sp_pca[i, 0]], [s_pca[i, 1], sp_pca[i, 1]],
                         color=dom_map[d], alpha=0.5, linestyle="-", linewidth=0.8)

            # Plot current state (circle)
            plt.scatter(s_pca[i, 0], s_pca[i, 1], color=dom_map[d], label=label, marker='o', s=30, edgecolors='black', linewidths=0.5)
            # Plot predicted next state (triangle)
            plt.scatter(next_s_pca[i, 0], next_s_pca[i, 1], color=dom_map[d], marker='^', s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

            if show_sp:
                # Plot actual next state (square)
                plt.scatter(sp_pca[i, 0], sp_pca[i, 1], color=dom_map[d], marker='s', s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

        # Add legend entries for markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Current (s)', markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=5, label='Predicted', markeredgecolor='black', markeredgewidth=0.5),
        ]
        if show_sp:
            legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=5, label='Actual', markeredgecolor='black', markeredgewidth=0.5))

        # Combine domain labels with marker labels
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles + legend_elements, labels + [e.get_label() for e in legend_elements],
                   loc='best', fontsize=6, framealpha=0.9)

        plt.xlabel(f'{method.upper()}1', fontsize=8)
        plt.ylabel(f'{method.upper()}2', fontsize=8)
        plt.tick_params(labelsize=7)
        plt.tight_layout()

        sp_suffix = "" if show_sp else "_no_sp"
        save_path = viz_dir / f"{method}{suffix}{sp_suffix}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {method.upper()} plot to {save_path}")


def main():
    args = arguments_parser()

    fix_seeds(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    use_wandb = cfg.get('wandb', {}).get('enabled', True) and not args.no_wandb
    if use_wandb and wandb is None:
        raise SystemExit("W&B logging is enabled in the config but wandb is not installed: "
                         "pip install wandb, or pass --no_wandb")
    wandb_project = args.wandb_project or cfg.get('wandb', {}).get('project', 'transition-function-prediction')

    train_ds, test_loaders, all_doms = [], {}, Config.domain_names
    if args.eval_protocol == "loo":
        if not args.test_domains:
            raise ValueError("LOO requires --test_domains")
        train_doms, test_doms = [d for d in all_doms if d not in args.test_domains], args.test_domains
        for d in train_doms:
            ds = FactorizedTripletDataset(domain=d, model_name=args.model_name)
            indices, _ = train_test_split(range(len(ds)), train_size=0.5, random_state=args.seed)
            train_ds.append(Subset(ds, indices))
        for d in test_doms:
            ds = FactorizedTripletDataset(domain=d, model_name=args.model_name)
            test_loaders[d] = DataLoader(
                ds, batch_sampler=ProblemGroupedBatchSampler(ds, args.val_batch_size),
                num_workers=4, pin_memory=True)
    else:
        test_doms = all_doms
        for d in all_doms:
            ds = FactorizedTripletDataset(domain=d, model_name=args.model_name)
            if args.split_type == "random":
                t_idx, v_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=args.seed)
            else:
                t_idx, v_idx, _ = grouped_split_by_problem(ds, train_frac=0.8, seed=args.seed)
            train_ds.append(Subset(ds, t_idx))
            test_loaders[d] = DataLoader(Subset(ds, v_idx), batch_size=args.val_batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)

    full_train = ConcatDataset(train_ds)
    ref_ds = train_ds[0].dataset if isinstance(train_ds[0], Subset) else train_ds[0]
    s_dim, a_dim = ref_ds.state_embs.shape[1], ref_ds.action_embs.shape[1]

    run_name = f"{args.eval_protocol}_{args.split_type}_test-{('+'.join(sorted(test_doms)) if len(test_doms) < 5 else 'ALL')}"
    save_path = Path(args.save_prefix) / f"{run_name}.pt"
    weights_path = Path(args.save_prefix) / f"{run_name}_pca_weights.pt"

    if weights_path.exists():
        print(f"Found existing weights at {weights_path}, running 2 epochs...")
        args.epochs = 2

    if use_wandb:
        wandb.init(project=wandb_project,
                   entity=cfg.get('wandb', {}).get('entity'), name=run_name, config=vars(args),
                   tags=[args.eval_protocol, args.split_type, args.model_type] + (
                       [f"LOO_{test_doms[0]}"] if args.eval_protocol == "loo" else []), reinit=True)

    if args.use_projection:
        model = ProjectedTransitionModel(ProjectionHead(s_dim, args.projection_dim, args.projection_layers),
                                         ProjectionHead(a_dim, args.projection_dim, args.projection_layers),
                                         model_selection(args.projection_dim, args, args.projection_dim))
    else:
        model = model_selection(a_dim, args, s_dim)

    model.to(device)

    if weights_path.exists():
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Failed to load weights from {weights_path}: {e}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    def viz_callback(model, epoch):
        if args.visualize_pca and (epoch % 30 == 0 or epoch == args.epochs):
            visualize_pca(model, test_loaders, device, args.save_prefix, run_name, method="pca", epoch=epoch)
            visualize_pca(model, test_loaders, device, args.save_prefix, run_name, method="umap", epoch=epoch)

    train_loop(model, DataLoader(full_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                 worker_init_fn=worker_init_fn), list(test_loaders.values())[0], device,
               epochs=args.epochs, lr=args.lr, eval_cfg=EvalConfig(), tau=args.tau, config=cfg, use_wandb=use_wandb,
               save_path=save_path, action_contrastive_weight=args.action_contrastive_weight, callback=viz_callback)


    torch.save(model.state_dict(), weights_path)

    final = {}
    for d, l in test_loaders.items():
        m = evaluate_hit_across_states(model, l, device, EvalConfig())
        final[d] = m
        print(f"Domain {d}: {m}")
        if use_wandb and not args.visualize_pca: wandb.log({f"final/test_{d}_{k}": v for k, v in m.items()})
    results_path = Path(weights_path).with_suffix(".json")
    results_path.write_text(json.dumps({"args": vars(args), "test_metrics": final}, indent=2))
    print(f"Wrote {results_path}")

    if args.visualize_pca:

        # Final visualization (if not covered by callback)
        visualize_pca(model, test_loaders, device, args.save_prefix, run_name, method="pca", epoch=None)
        visualize_pca(model, test_loaders, device, args.save_prefix, run_name, method="umap", epoch=None)

    if use_wandb and not args.visualize_pca: wandb.finish()


if __name__ == "__main__":
    main()
