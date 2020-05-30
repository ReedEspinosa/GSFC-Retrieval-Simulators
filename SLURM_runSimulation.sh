#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=1:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-230

date
hostname
echo "---Running Sims n="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 0 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 1 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 2 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 3 &
wait
exit 0
