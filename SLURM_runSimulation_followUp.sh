#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=15,51,63,69,71,87,93,159,197,213,215

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}" (and others)"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
wait
exit 0
