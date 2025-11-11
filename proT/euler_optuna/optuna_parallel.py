"""Minimal Parallel Optuna - no caching, CLI args only"""

import subprocess
import os
from pathlib import Path
from os import makedirs
from os.path import exists, join
import optuna


def run_parallel_optuna(exp_dir, home_exp_dir, experiment_id, study_name, n_trials, data_dir,
                       scratch_path=None, slurm_params=None, cluster=True,
                       optimization_metric="val_mae_mean", optimization_direction="minimize", study_path=None):
    """Create study, generate SLURM script, submit jobs."""
    
    # Defaults
    if slurm_params is None:
        slurm_params = {}
    slurm_params.setdefault('max_concurrent_jobs', 6)
    slurm_params.setdefault('walltime', '5-00:00:00')
    slurm_params.setdefault('gpu_type', 'rtx_4090')
    slurm_params.setdefault('mem_per_cpu', '23g')
    
    print(f"\nParallel Optuna: {experiment_id} / {study_name}")
    
    # Load study
    if study_path is None:
        study_path = Path(exp_dir) / "optuna"
        study_path.mkdir(parents=True, exist_ok=True)
    
    storage = f"sqlite:///{join(study_path, 'study.db')}?timeout=60"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded study: {len(study.trials)} trials completed")
    except KeyError:
        raise ValueError(f"Study '{study_name}' not found. Create it first with: "
                        f"python -m proT.euler_optuna.cli paramsopt --exp_id {experiment_id} "
                        f"--study_name {study_name} --mode create")
    
    # Determine trial limit with fallback logic
    n_trials_total = None
    
    # Try to read from study metadata (new studies)
    try:
        n_trials_total = study.user_attrs.get("n_trials_total")
        if n_trials_total is not None:
            print(f"Trial limit from study metadata: {n_trials_total}")
    except Exception:
        pass
    
    # Fallback: read from optuna.yaml (existing studies)
    if n_trials_total is None:
        import re
        from omegaconf import OmegaConf
        optuna_files = [f for f in os.listdir(home_exp_dir) if re.match(r'optuna.*\.yaml', f)]
        if len(optuna_files) == 1:
            optuna_config = OmegaConf.load(join(home_exp_dir, optuna_files[0]))
            n_trials_total = optuna_config.get("n_trials", None)
            if n_trials_total is not None:
                print(f"Trial limit from optuna.yaml: {n_trials_total}")
    
    # Use CLI parameter as fallback if nothing else works
    if n_trials_total is None:
        n_trials_total = n_trials
        print(f"Warning: Could not read trial limit from study or config. Using CLI parameter: {n_trials_total}")
    
    # Calculate remaining trials
    current_trials = len(study.trials)
    remaining_trials = n_trials_total - current_trials
    
    if remaining_trials <= 0:
        print(f"\nStudy is complete or exceeded limit!")
        print(f"  Current: {current_trials} trials")
        print(f"  Target: {n_trials_total} trials")
        print(f"No new jobs will be submitted.")
        return
    
    print(f"\nPlan:")
    print(f"  Total trials configured: {n_trials_total}")
    print(f"  Already completed: {current_trials}")
    print(f"  Remaining to submit: {remaining_trials}")
    print(f"  Max concurrent jobs: {slurm_params['max_concurrent_jobs']}")
    
    # Generate script with remaining trials count
    script_path = _gen_script(exp_dir, home_exp_dir, experiment_id, study_name, data_dir,
                              remaining_trials, slurm_params, optimization_metric, optimization_direction, cluster)
    print(f"Script: {script_path}")
    
    # Submit
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True, cwd=exp_dir)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"\nSubmitted! Job ID: {job_id}")
        print(f"Monitor: squeue -u $USER")
        print(f"Logs: {exp_dir}/optuna/slurm_logs/\n")
        with open(join(exp_dir, "optuna", "job_id.txt"), 'w') as f:
            f.write(job_id)
    else:
        print(f"Error: {result.stderr}")


def _gen_script(exp_dir, home_exp_dir, experiment_id, study_name, data_dir, n_trials,
                slurm_params, optimization_metric, optimization_direction, cluster):
    """Generate minimal SLURM script."""
    
    logs_dir = join(exp_dir, "optuna", "slurm_logs")
    if not exists(logs_dir):
        makedirs(logs_dir)
    
    cluster_flag = "--cluster" if cluster else ""
    
    script = f"""#!/bin/bash
#SBATCH --job-name=opt_{experiment_id}
#SBATCH --output={logs_dir}/opt_%A_%a.out
#SBATCH --error={logs_dir}/opt_%A_%a.err
#SBATCH --array=0-{n_trials-1}%{slurm_params['max_concurrent_jobs']}
#SBATCH --ntasks=1
#SBATCH --time={slurm_params['walltime']}
#SBATCH --gpus={slurm_params['gpu_type']}:1
#SBATCH --mem-per-cpu={slurm_params['mem_per_cpu']}

set -euo pipefail
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source "$HOME/myenv/bin/activate"

cd "{exp_dir}"
python -m proT.euler_optuna.optuna_worker \\
    --exp_dir "{exp_dir}" \\
    --home_exp_dir "{home_exp_dir}" \\
    --study_name "{study_name}" \\
    --data_dir "{data_dir}" \\
    --optimization_metric "{optimization_metric}" \\
    --optimization_direction "{optimization_direction}" \\
    --task_id $SLURM_ARRAY_TASK_ID {cluster_flag}
"""
    
    script_path = join(exp_dir, "optuna", "run_optuna_array.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    
    return script_path
