#!/bin/bash
#SBATCH --job-name=OSSE_Retrieval
#SBATCH --account=s1180
#SBATCH --ntasks=100          # Requesting resources by tasks/cores, not nodes
#SBATCH --time=12:00:00
#SBATCH --array=0-55 # Array indices matching the total days (56)
#SBATCH --output=logs/osse_%A_%a.out
#SBATCH --error=logs/osse_%A_%a.err

# Load required modules
module load comp/intel

# Activate the Python 3.11+ virtual environment 
source /home/dgiles/nobackup/AIST/software/.venv/bin/activate

# Execute the python processing script
python runRetrievalOSSE_daily.py
