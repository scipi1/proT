#!/bin/bash
#SBATCH --job-name=${EXPERIMENT_ID:-proT_optuna_parallel}
#SBATCH --output=optuna_parallel_output_%j.log
#SBATCH --error=optuna_parallel_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4g

set -euo pipefail

echo "[$(date)] Parallel Optuna submission job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ───────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
EXPERIMENT_ID="baseline_proTCVRand_dyconex_sum"
STUDY_NAME="optimization_study"
N_TRIALS=20
MAX_CONCURRENT_JOBS=6
WALLTIME="5-00:00:00"
GPU_TYPE="rtx_4090"
MEM_PER_CPU="23g"
OPTIMIZATION_METRIC="val_mae_mean"
OPTIMIZATION_DIRECTION="minimize"

# Project root and experiment folder in $HOME
PROJ_HOME="$HOME/proT"
HOME_EXP="$PROJ_HOME/experiments/$EXPERIMENT_ID"

# Scratch locations
RUN_DIR="$SCRATCH/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR"

mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Study name      : $STUDY_NAME"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"
echo "[$(date)] Total trials    : $N_TRIALS"
echo "[$(date)] Max concurrent  : $MAX_CONCURRENT_JOBS"
echo "[$(date)] Walltime        : $WALLTIME"
echo "[$(date)] GPU type        : $GPU_TYPE"
echo "[$(date)] Memory per CPU  : $MEM_PER_CPU"

# ───────────────────────────────────────────────────────────────────────
# COPY INPUTS TO SCRATCH
# ───────────────────────────────────────────────────────────────────────
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

# ───────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ───────────────────────────────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source "$HOME/myenv/bin/activate"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────────────────────────────
# SUBMIT PARALLEL OPTUNA OPTIMIZATION
# ───────────────────────────────────────────────────────────────────────
cd "$SCRATCH_EXP"

echo "[$(date)] Submitting parallel Optuna optimization..."

python -m proT.euler_optuna.cli_parallel paramsopt-parallel --exp_id "$EXPERIMENT_ID" --study_name "$STUDY_NAME" --n_trials "$N_TRIALS" --cluster --scratch_path "$SCRATCH_EXP" --max_concurrent_jobs "$MAX_CONCURRENT_JOBS" --walltime "$WALLTIME" --gpu_type "$GPU_TYPE" --mem_per_cpu "$MEM_PER_CPU" --optimization_metric "$OPTIMIZATION_METRIC" --optimization_direction "$OPTIMIZATION_DIRECTION"

deactivate
echo "[$(date)] Python environment deactivated"
echo "[$(date)] Parallel Optuna submission completed – results will be in $SCRATCH_EXP"
