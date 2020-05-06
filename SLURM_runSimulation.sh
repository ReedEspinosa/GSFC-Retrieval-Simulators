#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=1:29:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=6-22:2

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}" (and others)"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1)) &
wait
exit 0
