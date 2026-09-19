#!/usr/bin/env bash
# Resume the paper-protocol baseline sweep from wherever it stopped.
#
# Safe to run repeatedly: experiments/baseline_sweep.py writes each cell to its part
# file as soon as it completes and skips cells already present, so cancelled, preempted
# or requeued jobs resume rather than restart. Nothing is recomputed.
#
# Usage:  bash scripts/resume_protocol_sweep.sh [DOMAIN...]
set -u
DOMAINS=${@:-"ferry logistics goldminer"}
mkdir -p logs/protocol results/analysis/protocol_parts
n=0
for d in $DOMAINS; do
  for sp in problem_grouped random; do
    for s in 0 1 2; do
      part="results/analysis/protocol_parts/${d}_${sp}_seed${s}.json"
      have=0
      [ -f "$part" ] && have=$(python -c "import json;print(len(json.load(open('$part'))))" 2>/dev/null || echo 0)
      [ "${have:-0}" -ge 11 ] && { echo "skip  ${d}_${sp}_seed${s} (complete)"; continue; }
      sbatch --job-name="pp_${d:0:4}_${sp:0:4}_$s" \
        --output="logs/protocol/${d}_${sp}_seed${s}.log" \
        --error="logs/protocol/${d}_${sp}_seed${s}.log" \
        scripts/sbatch_job.sh experiments.baseline_sweep \
        --domains "$d" --splits "$sp" --seeds "$s" --out "$part" >/dev/null \
        && { echo "queue ${d}_${sp}_seed${s} (${have}/11 done)"; n=$((n+1)); }
    done
  done
done
echo "submitted $n jobs"
