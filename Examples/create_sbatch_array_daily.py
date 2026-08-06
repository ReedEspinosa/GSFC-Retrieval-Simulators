#!/usr/bin/env python3
import os

# Define the target timeframes
target_year = 2006
target_months = [3, 8, 9, 12]  # March, August, September, December
days_per_month = 14            # First two weeks

# Calculate total number of daily jobs needed (4 months * 14 days = 56 jobs)
total_jobs = len(target_months) * days_per_month

# SLURM Job Settings
account = "s1180"
maxCPU = 100          # Cores/Tasks requested per job
walltime = "12:00:00" # Increased walltime to allow processing 24 hours sequentially

# Create logs directory
os.makedirs("logs", exist_ok=True)

slurm_script_name = "submit_osse_array.sh"

slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=OSSE_Retrieval
#SBATCH --account={account}
#SBATCH --ntasks={maxCPU}          # Requesting resources by tasks/cores, not nodes
#SBATCH --time={walltime}
#SBATCH --array=0-{total_jobs - 1} # Array indices matching the total days (56)
#SBATCH --output=logs/osse_%A_%a.out
#SBATCH --error=logs/osse_%A_%a.err

# Load required modules
module load comp/intel

# Activate the Python 3.11+ virtual environment 
source /home/dgiles/nobackup/AIST/software/.venv/bin/activate

# Execute the python processing script
python runRetrievalOSSE_daily.py
"""

with open(slurm_script_name, "w") as f:
    f.write(slurm_script_content)

print(f"Successfully created '{slurm_script_name}' for {total_jobs} daily jobs.")
print(f"Running for Year: {target_year}")
print(f"Months: {target_months} (First 14 days of each month)")
print(f"To submit the jobs to DISCOVER, run:\n    sbatch {slurm_script_name}")
