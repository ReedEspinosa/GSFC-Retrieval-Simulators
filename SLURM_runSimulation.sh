#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:50:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-230

date
hostname
echo "---Running Sims n="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 4 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 5 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 6 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 7 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 8 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 9 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 10 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 11 &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) 12 &
wait
exit 0
