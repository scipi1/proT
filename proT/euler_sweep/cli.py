"""
CLI for proT Parameter Sweeps

This module provides a command-line interface for running parameter sweeps
with proT models using the euler_sweep framework.

USAGE EXAMPLES

Sequential Sweep (Local or Cluster)
------------------------------------
python -m proT.euler_sweep.cli sweep \
    --exp_id my_experiment \
    --sweep_mode combination

Parallel Sweep (Cluster with SLURM)
-----------------------------------
python -m proT.euler_sweep.cli sweep \
    --exp_id my_experiment \
    --sweep_mode combination \
    --parallel \
    --cluster \
    --scratch_path $SCRATCH/my_experiment

DIRECTORY STRUCTURE
-------------------
experiments/training/my_experiment/
├── config.yaml              # Main configuration
└── sweeper/
    ├── sweep.yaml          # Sweep definition
    └── runs/               # Created by sweeper
        ├── combinations/   # Combination mode results
        └── sweeps/         # Independent mode results
"""

# Standard library imports
import sys
import os
from os.path import abspath, join, exists, dirname

# Third-party imports
import click

# Setup paths
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

# proT-specific imports
from proT.training.trainer import trainer
from proT.training.experiment_control import update_config
from proT.euler_sweep.sweeper import run_sequential_sweep, run_parallel_sweep


# =============================================================================
# TRAINING WRAPPER FOR proT
# =============================================================================

def train_function_for_sweep(
    config,
    save_dir,
    data_dir,
    cluster,
    experiment_tag="sweep",
    **kwargs
):
    """
    Wrapper for proT's trainer function.
    
    This function:
    1. Updates the config with proT-specific preprocessing
    2. Calls proT's trainer with k-fold cross-validation
    3. Returns DataFrame with metrics from all folds
    
    Args:
        config: OmegaConf configuration
        save_dir: Directory to save outputs
        data_dir: Directory containing training data
        cluster: Whether running on cluster
        experiment_tag: Tag for experiment tracking
        **kwargs: Additional arguments
        
    Returns:
        pd.DataFrame with columns like val_mae, val_r2, test_mae, etc.
    """
    # Update config (proT-specific preprocessing)
    config_updated = update_config(config)
    
    # Call proT's trainer
    df_metric = trainer(
        config=config_updated,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
        resume_ckpt=None,
        plot_pred_check=True,
        debug=False
    )
    
    return df_metric


# =============================================================================
# CLI COMMANDS
# =============================================================================

@click.group()
def cli():
    """proT Parameter Sweep CLI."""
    pass


@click.command()
@click.option(
    "--exp_id",
    required=True,
    help="Experiment folder name (in experiments/training/)"
)
@click.option(
    "--sweep_mode",
    required=True,
    type=click.Choice(['independent', 'combination']),
    help="Sweep mode: independent (one param at a time) or combination (all combinations)"
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Enable parallel execution with SLURM job arrays"
)
@click.option(
    "--cluster",
    is_flag=True,
    default=False,
    help="Running on cluster (affects resource settings)"
)
@click.option(
    "--scratch_path",
    default=None,
    help="SCRATCH path for cluster execution (experiment will run here)"
)
@click.option(
    "--max_concurrent_jobs",
    type=int,
    default=6,
    help="Maximum concurrent SLURM jobs (parallel mode only)"
)
@click.option(
    "--walltime",
    default="5-00:00:00",
    help="SLURM walltime limit (parallel mode only)"
)
@click.option(
    "--gpu_mem",
    default="24g",
    help="GPU memory requirement (parallel mode only)"
)
@click.option(
    "--mem_per_cpu",
    default="10g",
    help="CPU memory requirement (parallel mode only)"
)
@click.option(
    "--submit_jobs",
    is_flag=True,
    default=True,
    help="Actually submit SLURM jobs (False for dry run)"
)
@click.option(
    "--exp_tag",
    default="sweep",
    help="Experiment tag for tracking"
)
def sweep(
    exp_id,
    sweep_mode,
    parallel,
    cluster,
    scratch_path,
    max_concurrent_jobs,
    walltime,
    gpu_mem,
    mem_per_cpu,
    submit_jobs,
    exp_tag
):
    """
    Run parameter sweeps with proT models.
    
    Sequential Mode (default):
        Runs combinations one after another. Suitable for local execution
        or small sweeps on cluster.
    
    Parallel Mode (--parallel):
        Uses SLURM job arrays for parallel execution. Only available on
        cluster with SLURM scheduler.
    
    Examples:
        Sequential local sweep:
            python -m proT.euler_sweep.cli sweep --exp_id my_exp --sweep_mode combination
        
        Parallel cluster sweep:
            python -m proT.euler_sweep.cli sweep --exp_id my_exp --sweep_mode combination --parallel --cluster
    """
    print(f"proT Parameter Sweep")
    print(f"Mode: {sweep_mode}, Parallel: {parallel}, Cluster: {cluster}")
    print("=" * 60)
    
    # Determine experiment directory
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments", exp_id)
        home_exp_dir = exp_dir
    else:
        exp_dir = scratch_path
        home_exp_dir = join(ROOT_DIR, "experiments", exp_id)
    
    data_dir = join(ROOT_DIR, "data", "input")
    
    # Verify experiment directory exists
    if not exists(home_exp_dir):
        print(f"ERROR: Experiment directory not found: {home_exp_dir}")
        print("\nCreate it with:")
        print(f"  mkdir -p {home_exp_dir}/sweeper")
        print(f"  # Add config.yaml to {home_exp_dir}/")
        print(f"  # Add sweep.yaml to {home_exp_dir}/sweeper/")
        sys.exit(1)
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Home directory: {home_exp_dir}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)
    print()
    
    # Run sweep based on mode
    if parallel:
        # Parallel execution requires cluster
        if not cluster:
            print("ERROR: Parallel execution requires --cluster flag")
            print("SLURM job arrays are only available on cluster environments")
            sys.exit(1)
        
        print(f"Running PARALLEL {sweep_mode} sweep...")
        print(f"Max concurrent jobs: {max_concurrent_jobs}")
        print(f"Walltime: {walltime}")
        print()
        
        # SLURM parameters
        slurm_params = {
            'max_concurrent_jobs': max_concurrent_jobs,
            'walltime': walltime,
            'gpu_mem': gpu_mem,
            'mem_per_cpu': mem_per_cpu
        }
        
        run_parallel_sweep(
            exp_dir=exp_dir,
            home_exp_dir=home_exp_dir,
            sweep_mode=sweep_mode,
            train_fn_module="proT.euler_sweep.cli",
            train_fn_name="train_function_for_sweep",
            data_dir=data_dir,
            scratch_path=scratch_path,
            slurm_params=slurm_params,
            cluster=cluster,
            submit_jobs=submit_jobs,
            experiment_tag=exp_tag
        )
    
    else:
        # Sequential execution
        print(f"Running SEQUENTIAL {sweep_mode} sweep...")
        print()
        
        run_sequential_sweep(
            exp_dir=exp_dir,
            sweep_mode=sweep_mode,
            train_fn=train_function_for_sweep,
            data_dir=data_dir,
            cluster=cluster,
            experiment_tag=exp_tag
        )
        
        print()
        print("=" * 60)
        print("Sweep completed!")
        print("=" * 60)
        
        if sweep_mode == "independent":
            print(f"Results: {exp_dir}/sweeper/runs/sweeps/")
        else:
            print(f"Results: {exp_dir}/sweeper/runs/combinations/")
        
        print("=" * 60)


# =============================================================================
# Register commands
# =============================================================================

cli.add_command(sweep)


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    cli()
