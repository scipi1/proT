#!/bin/bash
# K-fold training with ONE SLURM ARRAY TASK PER FOLD (folds run in parallel).
# Array version of train_cluster_scratch_job.sh. All tasks share one run dir
# ($SCRATCH/<job name>_<array job id>) holding k_0 ... k_<K-1> and a single
# kfold_summary.json, rebuilt by every task when its fold ends (the last task
# to finish leaves the complete summary).
#
# Usage:  sbatch --job-name=<EXPERIMENT_NAME> scripts/train_cluster_kfold_array.sh
# Set --array to 0-(k_fold-1); re-run a single fold with e.g. --array=3.
# Manual re-aggregation: python $CLI aggregate --scratch_path <SCRATCH_EXP>
#SBATCH --job-name=${EXPERIMENT_ID:-my_job}
#SBATCH --output=%x_%A_%a.log
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpumem:24g

set -euo pipefail

echo "[$(date)] Task started on $(hostname)"
echo "Array job: $SLURM_ARRAY_JOB_ID  task (fold): $SLURM_ARRAY_TASK_ID"

# ---------------------------------------------------------------
# 1)  EXPERIMENT SELECTION  (adapt the paths to your setup)
# ---------------------------------------------------------------
EXPERIMENT_ID="proTEncGRUpervar_smooth2_delta_nocurr"

PROJ_HOME="$HOME/proT"                 # repository root
CLI="$PROJ_HOME/proT/cli.py"           # training CLI
VENV="$PROJ_HOME/venv"                 # virtual environment
HOME_EXP="$PROJ_HOME/experiments/$EXPERIMENT_ID"

# shared by ALL array tasks (array job id, not the per-task job id)
RUN_DIR="$SCRATCH/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
SCRATCH_EXP="$RUN_DIR"
mkdir -p "$SCRATCH_EXP"

echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Scratch folder  : $SCRATCH_EXP"

# ---------------------------------------------------------------
# 2)  COPY INPUTS TO SCRATCH (once, serialized across tasks)
# ---------------------------------------------------------------
(
    flock -x 200
    if [[ ! -f "$SCRATCH_EXP/.inputs_copied" ]]; then
        rsync -av "$HOME_EXP/" "$SCRATCH_EXP/"
        touch "$SCRATCH_EXP/.inputs_copied"
    fi
) 200>"$SCRATCH_EXP/.copy.lock"

# ---------------------------------------------------------------
# 3)  ENVIRONMENT
# ---------------------------------------------------------------
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

source "$VENV/bin/activate"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ---------------------------------------------------------------
# 4)  RUN ONE FOLD
# ---------------------------------------------------------------
cd "$SCRATCH_EXP"
echo "[$(date)] Training fold $SLURM_ARRAY_TASK_ID"
python "$CLI" train --exp_id "$EXPERIMENT_ID" --cluster True --scratch_path "$SCRATCH_EXP" --plot_pred_check False --fold "$SLURM_ARRAY_TASK_ID"

# ---------------------------------------------------------------
# 5)  WRAP-UP
# ---------------------------------------------------------------
deactivate
echo "[$(date)] Fold $SLURM_ARRAY_TASK_ID finished - results in $SCRATCH_EXP"
