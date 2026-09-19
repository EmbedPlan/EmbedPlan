#!/usr/bin/env bash
# Run any embedplan module as a SLURM batch job.
#
# Usage:
#   sbatch --job-name=NAME --output=PATH scripts/sbatch_job.sh MODULE [ARGS...]
#
# Example:
#   sbatch scripts/sbatch_job.sh experiments.baseline_sweep --domain ferry --split random
#
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# Drivers write results incrementally and skip completed cells, so a requeued job
# resumes rather than restarting.
#SBATCH --requeue
#
# NOTE: this torch build ships kernels for compute capability sm_75 and above. On a
# heterogeneous cluster, exclude older nodes with --exclude=<nodelist>, otherwise a job
# landing on one dies with cudaErrorNoKernelImageForDevice partway through.
set -u

MODULE=${1:?module, e.g. experiments.baseline_sweep}
shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=${PY:-python}

# torch defaults to one thread per visible core, which oversubscribes a cgroup-limited
# allocation and slows the run down rather than speeding it up.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "=== $MODULE $* on $(hostname) ($(date +%F' '%H:%M)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
$PY -u -m "$MODULE" "$@"
status=$?
echo "=== exit ${status} ($(date +%F' '%H:%M)) ==="
exit $status
