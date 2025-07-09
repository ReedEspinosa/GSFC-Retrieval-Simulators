#!/usr/local/bin/bash
#SBATCH --job-name=aosVr21
#SBATCH --nodes=1 
#SBATCH --time=0:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-2

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}" (and no others)"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
wait
exit 0
