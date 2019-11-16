#!/usr/local/bin/bash
#SBATCH --job-name=SIM9ConCases22
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:30:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-188 

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID}
exit 0
