#!/usr/local/bin/bash
#SBATCH --job-name=GRASP_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:10:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-545

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID}
exit 0
