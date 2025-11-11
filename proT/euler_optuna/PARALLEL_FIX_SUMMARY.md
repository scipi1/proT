# Parallel Optuna Trial Limit Fix

## Problem
The parallel Optuna implementation was running more trials than specified in the study configuration. The issue occurred because:
1. Workers always called `study.ask()` without checking trial limits
2. The parallel script created SLURM array jobs based on CLI parameters, ignoring the study's configured limit
3. The study database in SCRATCH was updated but not synced back to HOME

## Solution
Implemented a three-part fix that respects trial limits for both new and existing studies:

### 1. Store Trial Limit in Study Metadata (`optuna_opt.py`)
- When creating a new study, the configured `n_trials` from `optuna.yaml` is now stored as study metadata
