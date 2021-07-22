#!/usr/local/bin/bash
#SBATCH --job-name=GPM
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH -o log_GPM/output.%A-%a
#SBATCH -e log_GPM/error.%A-%a
#SBATCH --array=10-129

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalOSSE_GPM.py ${SLURM_ARRAY_TASK_ID}
exit 0
