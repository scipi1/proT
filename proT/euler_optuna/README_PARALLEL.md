# Parallel Optuna Optimization

This document describes the NEW parallel Optuna optimization system that efficiently runs hyperparameter optimization on SLURM clusters using job arrays.

## Problem Solved

The original `optuna_resume_more_tasks.sh` was inefficient because it:
- Requested multiple GPUs in ONE job (e.g., 2× RTX 4090)
- Required all resources simultaneously before starting
- Had poor resource utilization with internal parallelism via `srun`
- Jobs often never started due to unavailable resources

The NEW parallel system solves this by:
- Requesting ONE GPU per trial
- Submitting independent array jobs that schedule flexibly
- Each trial runs as a separate SLURM job
- Much better cluster scheduler efficiency

## Architecture

```
scripts/optuna_parallel.sh (lightweight submission job)
    ├── Creates Optuna study database (if needed)
    ├── Creates SLURM array script with CLI args
    └── Submits array job (0 to N_TRIALS-1)
        └── Each array task:
            ├── Loads base config FRESH from experiment directory
            ├── Loads study from database
            ├── Asks for next trial (Optuna handles coordination)
            ├── Samples parameters with fresh config
            ├── Runs training with updated config
            ├── Reports results
            └── Exits
```

**Key Design**: Workers load config fresh from disk (no caching) to ensure correct dimensions after `update_config` is called.

## New Files Created

All new files - NO modifications to existing code:

1. **proT/euler_optuna/optuna_worker.py** - Worker script for array tasks
2. **proT/euler_optuna/optuna_parallel.py** - Parallel execution functions
3. **proT/euler_optuna/cli_parallel.py** - CLI for parallel mode
4. **scripts/optuna_parallel.sh** - Submission script

## Usage

### 1. Configure the Submission Script

Edit `scripts/optuna_parallel.sh`:

```bash
EXPERIMENT_ID="baseline_LSTM_ishigami_cat"
STUDY_NAME="optimization_study"
N_TRIALS=100                    # Total trials to run
MAX_CONCURRENT_JOBS=6           # Max parallel trials
WALLTIME="5-00:00:00"          # Time per trial
GPU_TYPE="rtx_4090"            # GPU type
MEM_PER_CPU="23g"              # Memory per trial
OPTIMIZATION_METRIC="val_mae_mean"
OPTIMIZATION_DIRECTION="minimize"
```

### 2. Submit the Job

```bash
sbatch scripts/optuna_parallel.sh
```

This lightweight job (1 hour, 4GB RAM, 1 CPU):
1. Copies experiment data to SCRATCH
2. Creates the Optuna study
3. Submits the array job
4. Exits

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View logs
tail -f $SCRATCH/${EXPERIMENT_ID}_*/optuna/slurm_logs/optuna_*.out

# Check study progress
# The Optuna database is at: $SCRATCH/${EXPERIMENT_ID}_*/optuna/study.db
```

### 4. View Results

After completion, view the best trial:

```bash
cd experiments/${EXPERIMENT_ID}
cat best_trial.yaml
```

## Key Differences from Sweep Parallel

| Feature | Sweep Parallel | Optuna Parallel |
|---------|---------------|-----------------|
| **Combinations** | Pre-generated (all known upfront) | Dynamic (Optuna asks/tells) |
| **Sampling** | Grid/random combinations | Intelligent (TPE, Sobol, etc.) |
| **Learning** | No (independent) | Yes (learns from completed trials) |
| **Pruning** | No | Yes (can stop bad trials early) |
| **Use Case** | Systematic grid search | Smart exploration |

## Resource Allocation

Each trial requests:
- **1 GPU** (e.g., `--gpus=rtx_4090:1`)
- **1 task** (`--ntasks=1`)
- **23g per CPU** (`--mem-per-cpu=23g`)
- **5 days max** (`--time=5-00:00:00`)

This is MUCH more scheduler-friendly than requesting 2 GPUs simultaneously!

## Concurrent Trial Coordination

- All trials share a single SQLite database
- Optuna's `ask()` method is thread-safe with proper timeout
- Each worker gets a unique trial to run
- No conflicts or duplicate trials
- Storage: `sqlite:///path/to/study.db?timeout=60`

## Comparison with Original Script

### Old: optuna_resume_more_tasks.sh
```bash
#SBATCH --ntasks-per-node=2      # Needs 2 tasks
#SBATCH --gpus=rtx_4090:2        # Needs 2 GPUs
#SBATCH --mem-per-cpu=23g        # 46GB total

# Hard to schedule - needs full allocation at once
srun python ...  # Internal parallelism
```

### New: optuna_parallel.sh + array jobs
```bash
# Submission job (lightweight)
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4g

# Each array task (flexible)
#SBATCH --array=0-99%6           # 100 trials, max 6 concurrent
#SBATCH --ntasks=1               # Just 1 task
#SBATCH --gpus=rtx_4090:1        # Just 1 GPU
#SBATCH --mem-per-cpu=23g        # 23GB total

# Easy to schedule - small independent jobs
python -m proT.euler_optuna.optuna_worker ...
```

## Advanced Usage

### Dry Run
Test without submitting:
```bash
python -m proT.euler_optuna.cli_parallel paramsopt-parallel \
    --exp_id baseline_LSTM_ishigami_cat \
    --study_name test_study \
    --n_trials 10 \
    --cluster \
    --no-submit_jobs
```

### Resume a Study
If trials fail or you want to run more:
```bash
# Just submit the script again with same STUDY_NAME
# It will load the existing study and continue
sbatch scripts/optuna_parallel.sh
```

### Custom Parameters
```bash
python -m proT.euler_optuna.cli_parallel paramsopt-parallel \
    --exp_id my_experiment \
    --study_name my_study \
    --n_trials 50 \
    --max_concurrent_jobs 10 \
    --walltime "2-00:00:00" \
    --gpu_type "rtx_3090" \
    --mem_per_cpu "16g" \
    --optimization_metric "val_r2_mean" \
    --optimization_direction "maximize" \
    --cluster
```

## Troubleshooting

### Dimension Mismatch Errors
**Fixed in current version!** The system now loads config fresh on each trial (no caching) to ensure `update_config` correctly computes dimensions like `d_in`, `d_emb` based on sampled hyperparameters.

### Study Already Exists
If you see "Loaded existing study", the system automatically resumes. To start fresh:
```bash
rm $SCRATCH/${EXPERIMENT_ID}_*/optuna/study.db
```

### Worker Crashes
Check individual worker logs:
```bash
tail -f $SCRATCH/${EXPERIMENT_ID}_*/optuna/slurm_logs/optuna_*_5.err  # Task 5
```

### No Trials Running
Check if jobs are pending:
```bash
squeue -u $USER
```

If all jobs are completed but trials < N_TRIALS, some workers may have failed. Check logs.

## Benefits Summary

✅ **Efficient Scheduling**: Small jobs schedule faster
✅ **Better Resource Utilization**: No wasted allocation
✅ **Fault Tolerance**: Failed trials don't affect others
✅ **Flexible**: Easy to add more trials later
✅ **Smart Optimization**: Optuna learns from completed trials
✅ **No Code Changes**: Existing experiments work as-is

## See Also

- Original CLI: `proT/euler_optuna/cli.py` (unchanged)
- Sweep parallel: `scripts/sweep_parallel.sh` (similar architecture)
- Optuna docs: https://optuna.readthedocs.io/
