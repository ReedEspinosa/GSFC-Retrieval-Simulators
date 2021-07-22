#!/usr/local/bin/bash
#SBATCH --job-name=SS450
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=10-79,90-129

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalOSSE.py ${SLURM_ARRAY_TASK_ID}
exit 0
