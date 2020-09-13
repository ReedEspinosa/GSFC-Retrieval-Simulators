#!/usr/local/bin/bash
#SBATCH --job-name=GCC
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=00:30:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-400:7

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+2)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+3)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+4)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+5)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+6)) &
wait
exit 0
