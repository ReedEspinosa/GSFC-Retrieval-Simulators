#!/usr/local/bin/bash
#SBATCH --job-name=BASIC
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=00:25:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=1-2

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+0)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+3)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+6)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+9)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+12)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+15)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+18)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+21)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+24)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+27)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+31)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+34)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+37)) &
python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+41)) &
wait
exit 0
