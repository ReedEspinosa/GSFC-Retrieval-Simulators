#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-225

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}"+m*225, for m=0,1,...,7---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+225)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+450)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+675)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+900)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1125)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1350)) &
wait
exit 0
