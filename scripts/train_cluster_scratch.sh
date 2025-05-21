#!/bin/bash
#SBATCH --job-name=${EXPERIMENT_ID:-my_job}
#SBATCH --output=my_job_output_%j.log
#SBATCH --error=my_job_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=15-00:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpumem:24g

set -euo pipefail                                      

echo "[$(date)] Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ───────────────────────────────────────────────
# 1)  EXPERIMENT SELECTION
# ───────────────────────────────────────────────
EXPERIMENT_ID="dx_250415_200_mean_clip_emb50_mod50"

# Project root and experiment folder in $HOME
PROJ_HOME="$HOME/prochain_transformer"
HOME_EXP="$PROJ_HOME/experiments/training/$EXPERIMENT_ID"

# Scratch locations
RUN_DIR="$SCRATCH/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
SCRATCH_EXP="$RUN_DIR/experiments/training/$EXPERIMENT_ID"

echo "Is it None [$EXPERIMENT_ID $SCRATCH_EXP]"

mkdir -p "$SCRATCH_EXP"                                

echo "[$(date)] Experiment ID   : $EXPERIMENT_ID"
echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"

# ───────────────────────────────────────────────
# 2)  COPY INPUTS TO SCRATCH
# ───────────────────────────────────────────────
rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"

# ───────────────────────────────────────────────
# 3)  ENVIRONMENT
# ───────────────────────────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# Activate virtual‑env (use absolute path)
source "$PROJ_HOME/venv/bin/activate"                  

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────
# 4)  RUN
# ───────────────────────────────────────────────
cd "$SCRATCH_EXP"

echo "[$(date)] Running script…"
python "$PROJ_HOME/prochain_transformer/cli.py" train --exp_id "$EXPERIMENT_ID" --cluster True --scratch_path "$SCRATCH_EXP" --plot_pred_check True

# ───────────────────────────────────────────────
# 5)  WRAP‑UP
# ───────────────────────────────────────────────
deactivate
echo "[$(date)] Python environment deactivated"

echo "[$(date)] Job finished – results are still in $SCRATCH_EXP"
