#!/usr/bin/env python3
import os
from datetime import datetime

# Define your date range here
start_date = datetime(2006, 8, 1)
end_date = datetime(2006, 8, 3)

# Calculate total number of hourly jobs needed
total_hours = int((end_date - start_date).total_seconds() // 3600)

# SLURM Job Settings
account = "s1180"  # Updated account
maxCPU = 100       # Cores/Tasks requested
walltime = "02:00:00" # Walltime per hourly job

# Create logs directory
os.makedirs("logs", exist_ok=True)

slurm_script_name = "submit_osse_array.sh"

slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=OSSE_Retrieval
#SBATCH --account={account}
#SBATCH --ntasks={maxCPU}          # Requesting resources by tasks/cores, not nodes
#SBATCH --time={walltime}
#SBATCH --array=0-{total_hours - 1} # Array indices matching the total hours
#SBATCH --output=logs/osse_%A_%a.out
#SBATCH --error=logs/osse_%A_%a.err

# Load required modules
module load comp/intel

# Activate the Python 3.11+ virtual environment 
# (This ensures numpy, scipy, pandas, pyyaml, netCDF4, and matplotlib are available)
source /home/dgiles/nobackup/AIST/software/.venv/bin/activate

# Execute the python processing script
python runRetrievalOSSE_single.py
"""

with open(slurm_script_name, "w") as f:
    f.write(slurm_script_content)

print(f"Successfully created '{slurm_script_name}' for {total_hours} hourly jobs.")
print(f"To submit the jobs to DISCOVER, run:\n    sbatch {slurm_script_name}")
