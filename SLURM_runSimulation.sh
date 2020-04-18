#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:19:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-180

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}"+m, for m=0"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
wait
exit 0
