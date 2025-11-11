#!/bin/bash
#SBATCH --job-name=${EXPERIMENT_ID:-proT_parallel_sweep}
#SBATCH --output=parallel_sweep_output_%j.log
#SBATCH --error=parallel_sweep_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4g

set -euo pipefail

echo "[$(date)] Parallel sweep submission job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ───────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ───────────────────────────────────────────────
EXPERIMENT_ID="proT_ishigami_cat"
MAX_CONCURRENT_JOBS=6
WALLTIME="5-00:00:00"
GPU_MEM="24g"
MEM_PER_CPU="10g"
SWEEP_MODE="combination"

# Project root and experiment folder in $HOME
PROJ_HOME="$HOME/proT"
HOME_EXP="$PROJ_HOME/experiments/$EXPERIMENT_ID"

# Scratch locations
RUN_DIR="$SCRATCH/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR"

mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"
echo "[$(date)] Sweep mode      : $SWEEP_MODE"
echo "[$(date)] Max concurrent  : $MAX_CONCURRENT_JOBS"
echo "[$(date)] Walltime        : $WALLTIME"

# ───────────────────────────────────────────────
# COPY INPUTS TO SCRATCH
# ───────────────────────────────────────────────
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

# ───────────────────────────────────────────────
# ENVIRONMENT SETUP
# ───────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source "$HOME/myenv/bin/activate"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────
# SUBMIT PARALLEL SWEEP
# ───────────────────────────────────────────────
cd "$SCRATCH_EXP"

echo "[$(date)] Submitting parallel sweep..."

python -m proT.euler_sweep.cli sweep --exp_id "$EXPERIMENT_ID" --sweep_mode "$SWEEP_MODE" --parallel --cluster --scratch_path "$SCRATCH_EXP" --max_concurrent_jobs "$MAX_CONCURRENT_JOBS" --walltime "$WALLTIME" --gpu_mem "$GPU_MEM" --mem_per_cpu "$MEM_PER_CPU"

deactivate
echo "[$(date)] Python environment deactivated"
echo "[$(date)] Parallel sweep submission completed – results will be in $SCRATCH_EXP"
