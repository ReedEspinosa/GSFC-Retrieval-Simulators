#!/usr/local/bin/bash
#SBATCH --job-name=GOSSE
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=00:04:30
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-9

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py 0 ${SLURM_ARRAY_TASK_ID}
python runRetrievalSimulation.py 0 1${SLURM_ARRAY_TASK_ID}
python runRetrievalSimulation.py 0 2${SLURM_ARRAY_TASK_ID}
python runRetrievalSimulation.py 0 3${SLURM_ARRAY_TASK_ID}
python runRetrievalSimulation.py 0 4${SLURM_ARRAY_TASK_ID}
python runRetrievalSimulation.py 0 5${SLURM_ARRAY_TASK_ID}
exit 0
