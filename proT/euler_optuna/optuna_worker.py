"""Minimal Optuna Worker - loads config fresh, runs one trial"""

import sys
import os
import re
import click
from pathlib import Path
from os.path import join, dirname, abspath
import optuna
import torch
from omegaconf import OmegaConf
from functools import partial

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

from proT.euler_optuna.optuna_opt import objective_extended
from proT.euler_optuna.cli import sample_params_for_optuna, train_function_for_optuna, get_metrics_for_optuna


@click.command()
@click.option("--exp_dir", required=True)
@click.option("--home_exp_dir", required=True)
@click.option("--study_name", required=True)
@click.option("--data_dir", required=True)
@click.option("--optimization_metric", default="val_mae_mean")
@click.option("--optimization_direction", default="minimize")
@click.option("--task_id", required=True, type=int)
@click.option("--cluster", is_flag=True, default=False)
def main(exp_dir, home_exp_dir, study_name, data_dir, optimization_metric, 
         optimization_direction, task_id, cluster):
    """Run one Optuna trial."""
    print(f"[Worker {task_id}] Starting")
    
    # Load config fresh from home directory
    config_files = [f for f in os.listdir(home_exp_dir) if re.match(r'config.*\.yaml', f)]
    if len(config_files) != 1:
        raise ValueError(f"Expected 1 config file, found {len(config_files)}")
    
    base_config = OmegaConf.load(join(home_exp_dir, config_files[0]))
    print(f"[Worker {task_id}] Loaded config: {config_files[0]}")
    
    # Load study
    storage = f"sqlite:///{join(exp_dir, 'optuna', 'study.db')}?timeout=60"
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"[Worker {task_id}] Loaded study ({len(study.trials)} trials)")
    
    # Get trial limit with fallback logic
    n_trials_total = None
    
    # Try to read from study metadata (new studies)
    if hasattr(study, '_study_id'):
        try:
            n_trials_total = study.user_attrs.get("n_trials_total")
            if n_trials_total is not None:
                print(f"[Worker {task_id}] Trial limit from study metadata: {n_trials_total}")
        except Exception:
            pass
    
    # Fallback: read from optuna.yaml (existing studies)
    if n_trials_total is None:
        optuna_files = [f for f in os.listdir(home_exp_dir) if re.match(r'optuna.*\.yaml', f)]
        if len(optuna_files) == 1:
            optuna_config = OmegaConf.load(join(home_exp_dir, optuna_files[0]))
            n_trials_total = optuna_config.get("n_trials", None)
            if n_trials_total is not None:
                print(f"[Worker {task_id}] Trial limit from optuna.yaml: {n_trials_total}")
    
    # Check if study has reached its limit
    if n_trials_total is not None:
        current_trials = len(study.trials)
        if current_trials >= n_trials_total:
            print(f"[Worker {task_id}] Study limit reached ({current_trials}/{n_trials_total}). Exiting gracefully.")
            return
        print(f"[Worker {task_id}] Progress: {current_trials}/{n_trials_total} trials")
    else:
        print(f"[Worker {task_id}] Warning: Could not determine trial limit. Proceeding anyway.")
    
    # Create objective
    objective = partial(
        objective_extended,
        sample_params=lambda t: sample_params_for_optuna(t, base_config),
        train_function=train_function_for_optuna,
        get_metrics=get_metrics_for_optuna,
        config=base_config,
        exp_path=Path(exp_dir),
        data_dir=Path(data_dir),
        experiment_tag=f"optuna_{study_name}",
        cluster=cluster,
        optimization_metric=optimization_metric,
        optimization_direction=optimization_direction
    )
    
    # Set CUDA
    if cluster and torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # Run trial
    trial = study.ask()
    print(f"[Worker {task_id}] Running trial {trial.number}")
    value = objective(trial)
    study.tell(trial, value)
    print(f"[Worker {task_id}] Completed: {value}")


if __name__ == "__main__":
    main()
