#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-25

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}"+m*26, for m=0,1,...,7---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+26)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+52)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+78)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+104)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+130)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+156)) &
wait
exit 0
