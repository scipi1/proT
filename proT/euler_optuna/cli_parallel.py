"""
Simplified CLI for Parallel Optuna Hyperparameter Optimization

This module provides a command to run Optuna hyperparameter optimization
in parallel using SLURM job arrays. It reuses all working functions from cli.py.

USAGE EXAMPLE

python -m proT.euler_optuna.cli_parallel paramsopt-parallel \
    --exp_id baseline_LSTM_ishigami_cat \
    --study_name optimization_study \
    --n_trials 20 \
    --max_concurrent_jobs 6 \
    --cluster
"""

import sys
import os
from os.path import abspath, join, dirname

import click

# Setup paths
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

# Import only what we need from cli.py
from proT.euler_optuna.cli import SAMPLING_BOUNDS, SAMPLING_PROFILES
from proT.euler_optuna.optuna_parallel import run_parallel_optuna


@click.group()
def cli():
    """Parallel Optuna Optimization CLI."""
    pass


@click.command()
@click.option("--exp_id", required=True, help="Experiment folder containing config")
@click.option("--cluster", default=False, is_flag=True, help="Running on cluster?")
@click.option("--study_name", default="optimization_study", help="Optuna study name")
@click.option("--n_trials", required=True, type=int, help="Total number of trials")
@click.option("--scratch_path", default=None, help="SCRATCH path (for cluster)")
@click.option("--study_path", default=None, help="Path to store study database")
@click.option("--optimization_metric", default="val_mae_mean", help="Metric to optimize")
@click.option("--optimization_direction", default="minimize", type=click.Choice(['minimize', 'maximize']))
@click.option("--max_concurrent_jobs", type=int, default=6, help="Max concurrent SLURM jobs")
@click.option("--walltime", default="5-00:00:00", help="SLURM walltime per trial")
@click.option("--gpu_type", default="rtx_4090", help="GPU type")
@click.option("--mem_per_cpu", default="23g", help="Memory per CPU")
@click.option("--sampling_profile", default="baseline", type=click.Choice(['baseline']))
def paramsopt_parallel(exp_id, cluster, study_name, n_trials, scratch_path,
                      study_path, optimization_metric, optimization_direction, 
                      max_concurrent_jobs, walltime, gpu_type, mem_per_cpu, 
                      sampling_profile):
    """
    Run Optuna optimization in PARALLEL using SLURM job arrays.
    
    This creates/resumes a study and submits array jobs where each task runs one trial.
    """
    print(f"Parallel Optuna: exp_id={exp_id}, study={study_name}")
    print(f"Trials: {n_trials}, Concurrent: {max_concurrent_jobs}")
    print(f"Profile: {sampling_profile}")
    
    # Set sampling bounds
    global SAMPLING_BOUNDS
    SAMPLING_BOUNDS = SAMPLING_PROFILES[sampling_profile]
    
    # Paths
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments", exp_id)
        home_exp_dir = exp_dir
    else:
        exp_dir = scratch_path
        home_exp_dir = join(ROOT_DIR, "experiments", exp_id)
    
    data_dir = join(ROOT_DIR, "data", "input")
    
    if not os.path.exists(home_exp_dir):
        raise ValueError(f"Experiment directory does not exist: {home_exp_dir}")
    
    print(f"Experiment dir: {exp_dir}")
    print(f"Home dir: {home_exp_dir}")
    
    # SLURM parameters
    slurm_params = {
        'max_concurrent_jobs': max_concurrent_jobs,
        'walltime': walltime,
        'gpu_type': gpu_type,
        'mem_per_cpu': mem_per_cpu
    }
    
    # Run parallel optimization
    experiment_id = os.path.basename(home_exp_dir)
    
    run_parallel_optuna(
        exp_dir=exp_dir,
        home_exp_dir=home_exp_dir,
        experiment_id=experiment_id,
        study_name=study_name,
        n_trials=n_trials,
        data_dir=data_dir,
        scratch_path=scratch_path,
        slurm_params=slurm_params,
        cluster=cluster,
        optimization_metric=optimization_metric,
        optimization_direction=optimization_direction,
        study_path=study_path
    )


cli.add_command(paramsopt_parallel, name='paramsopt-parallel')


if __name__ == "__main__":
    cli()
