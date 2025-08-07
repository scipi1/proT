#!/bin/bash

#SBATCH --job-name=${EXPERIMENT_ID:-my_job}
#SBATCH --output=my_job_output_%j.log
#SBATCH --error=my_job_error_%j.log
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_4090:4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=23g



echo "Partition : $SLURM_JOB_PARTITION"
echo "GPUs asked: ${SLURM_JOB_GPUS:-none}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$(date)] Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ───────────────────────────────────────────────
# 1)  EXPERIMENT SELECTION
# ───────────────────────────────────────────────
EXPERIMENT_ID="proT_cat_dyconex_optuna"
EXP_TAG="proT_cat_dyconex_optuna"
STUDY_NAME="NA"

# Project root and experiment folder in $HOME
PROJ_HOME="$HOME/prochain_transformer"
HOME_EXP="$PROJ_HOME/experiments/training/$EXPERIMENT_ID"

# Scratch locations
RUN_DIR="$SCRATCH/${EXPERIMENT_ID}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR/experiments/training/$EXPERIMENT_ID"


mkdir -p "$SCRATCH_EXP"                                

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"

# ───────────────────────────────────────────────
# 2)  COPY INPUTS TO SCRATCH (once, by task 0)
# ───────────────────────────────────────────────
if [[ $SLURM_PROCID -eq 0 ]]; then
    rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"
fi
# wait until copy done before others proceed


# ───────────────────────────────────────────────
# 3)  ENVIRONMENT
# ───────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# Activate virtual‑env (use absolute path)
source "$PROJ_HOME/venv/bin/activate"

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
echo "[$(date)] Task $SLURM_PROCID uses GPU $CUDA_VISIBLE_DEVICES"         

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────
# 4)  RUN
# ───────────────────────────────────────────────
cd "$SCRATCH_EXP"

# helper: absolute path to CLI
CLI="$PROJ_HOME/prochain_transformer/cli.py"


# STAGE B — RESUME (all tasks in parallel)
# Each task sees exactly ONE GPU via Slurm’s CUDA_VISIBLE_DEVICES mask

echo "[$(date)] Stage B: parallel resume — $NUM_WORKERS workers"
srun python "$CLI" paramsopt --exp_id "$EXPERIMENT_ID" --cluster True --study_name "$STUDY_NAME" --mode resume --scratch_path "$SCRATCH_EXP" --study_path "$HOME_EXP/optuna"



# ───────────────────────────────────────────────
# 5)  WRAP‑UP
# ───────────────────────────────────────────────
deactivate
echo "[$(date)] Python environment deactivated"

echo "[$(date)] Job finished – results are still in $SCRATCH_EXP"
