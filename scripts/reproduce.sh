#!/usr/bin/env bash
# Re-run the paper's training grids with the settings that produced its tables.
#
#   bash scripts/reproduce.sh main      [ENCODER]   # Interpolation + Extrapolation, 9 domains x 3 seeds
#   bash scripts/reproduce.sh untrained [ENCODER]   # the untrained-network floor
#   bash scripts/reproduce.sh plan      [ENCODER]   # Plan-Variant evaluation
#   bash scripts/reproduce.sh cross     [ENCODER]   # Cross-Domain: train on one domain, test on another
#   bash scripts/reproduce.sh loo       [ENCODER]   # Leave-One-Out: train on 8 domains, test on the 9th
#   bash scripts/reproduce.sh multi     [ENCODER]   # Multi-Domain: one model trained on all nine domains
#
# ENCODER defaults to meta-llama/Llama-3.3-70B-Instruct; the paper also reports
# sentence-transformers/all-mpnet-base-v2, BAAI/bge-m3 and Qwen/Qwen2.5-7B-Instruct.
# Every run writes <prefix>.json and is skipped when that file exists, so the script
# resumes where it stopped. Each run needs one GPU; set PY to choose the interpreter.
set -euo pipefail

PY=${PY:-python}
GRID=${1:?usage: bash scripts/reproduce.sh main|untrained|plan|cross|loo|multi [ENCODER]}
ENC=${2:-meta-llama/Llama-3.3-70B-Instruct}
TAG=${ENC//\//_}
OUT=${EMBEDPLAN_RESULTS:-results}/reproduce/$GRID/$TAG
DOMAINS=(ferry rovers blocksworld depot floortile goldminer grid logistics satellite)
SEEDS=(0 1 2)
mkdir -p "$OUT"

# Shared by every grid: residual MLP in a learned 128-d space, InfoNCE (tau 0.07) plus
# the action-disambiguation term (weight 2), AdamW at 4e-5, batches of 128.
COMMON=(--model_name "$ENC" --use_projection --projection_dim 128 --use_layer_norm
        --n_layers 2 --lr 4e-5 --batch_size 128 --tau 0.07 --action_contrastive_weight 2)

run() { echo "+ $*"; "$PY" "$@"; }

case "$GRID" in
  main)
    for split in problem_grouped random; do for d in "${DOMAINS[@]}"; do for s in "${SEEDS[@]}"; do
      run -m experiments.train "${COMMON[@]}" --model_type mlp --hidden_size 128 --projection_layers 2 \
          --epochs 200 --domain "$d" --split_type "$split" --seed "$s" --save_prefix "$OUT/${d}_${split}_s${s}"
    done; done; done ;;
  untrained)
    for split in problem_grouped random; do for d in "${DOMAINS[@]}"; do for s in "${SEEDS[@]}"; do
      run -m experiments.train "${COMMON[@]}" --model_type mlp --hidden_size 128 --projection_layers 2 \
          --eval_only --domain "$d" --split_type "$split" --seed "$s" --save_prefix "$OUT/${d}_${split}_s${s}"
    done; done; done ;;
  plan)
    for split in problem_grouped plan_grouped; do for d in "${DOMAINS[@]}"; do for s in "${SEEDS[@]}"; do
      run -m experiments.train "${COMMON[@]}" --model_type mlp --hidden_size 128 --projection_layers 4 \
          --epochs 200 --domain "$d" --split_type "$split" --seed "$s" --save_prefix "$OUT/${d}_${split}_s${s}"
    done; done; done ;;
  cross)
    for tr in "${DOMAINS[@]}"; do for te in "${DOMAINS[@]}"; do [[ "$tr" == "$te" ]] && continue
      for s in "${SEEDS[@]}"; do
        run -m experiments.train "${COMMON[@]}" --model_type hyper --hidden_size 256 --projection_layers 4 \
            --epochs 250 --train_domain "$tr" --test_domain "$te" --split_type problem_grouped --seed "$s" \
            --save_prefix "$OUT/${tr}_to_${te}_s${s}"
      done
    done; done ;;
  loo)
    for d in "${DOMAINS[@]}"; do for s in "${SEEDS[@]}"; do
      # train_multi_domain uses --save_prefix as a folder and does not skip finished runs itself
      compgen -G "$OUT/loo_${d}_s${s}/*.json" > /dev/null && { echo "skip loo_${d}_s${s}"; continue; }
      run -m experiments.train_multi_domain "${COMMON[@]}" --model_type mlp --hidden_size 128 \
          --projection_layers 4 --epochs 100 --eval_protocol loo --test_domains "$d" \
          --split_type problem_grouped --seed "$s" --save_prefix "$OUT/loo_${d}_s${s}"
    done; done ;;
  multi)
    for split in problem_grouped random; do for s in "${SEEDS[@]}"; do
      compgen -G "$OUT/multi_${split}_s${s}/*.json" > /dev/null && { echo "skip multi_${split}_s${s}"; continue; }
      run -m experiments.train_multi_domain "${COMMON[@]}" --model_type mlp --hidden_size 128 \
          --projection_layers 4 --epochs 100 --eval_protocol in_domain --split_type "$split" --seed "$s" \
          --save_prefix "$OUT/multi_${split}_s${s}"
    done; done ;;
  *) echo "unknown grid: $GRID" >&2; exit 2 ;;
esac
echo "=== $GRID DONE ($OUT) ==="
