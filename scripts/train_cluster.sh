#!/bin/bash
#SBATCH --job-name=my_job            # Job name
#SBATCH --output=my_job_output.log   # Standard output and error log
#SBATCH --error=my_job_error.log     # Separate error log (optional)
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --time=5-00:00:00            # Walltime (hh:mm:ss)
#SBATCH --gpus=1 
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpumem:24g

# Load necessary modules or set up environment
module load stack/2024-06
module load gcc/12.2.0 
module load python_cuda/3.11.6

# Activate venv
source prochain_transformer/myenv/bin/activate


# Run your program
# sbatch --wrap="python prochain_transformer/prochain_transformer/cli.py train --exp_id dyconex_250312_test_cluster --debug True --cluster True"

python prochain_transformer/prochain_transformer/cli.py train --exp_id dx_250318 --cluster True

deactivate