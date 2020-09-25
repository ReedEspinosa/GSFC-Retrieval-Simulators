#!/usr/local/bin/bash
#SBATCH --job-name=GOSSE
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=02:00:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalOSSE.py 0
python runRetrievalOSSE.py 1
python runRetrievalOSSE.py 2
python runRetrievalOSSE.py 3
exit 0
