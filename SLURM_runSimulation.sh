#!/usr/local/bin/bash
#SBATCH --job-name=GOSSE
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=01:30:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-240:14

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
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+7)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+8)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+9)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+10)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+11)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+12)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+13)) &
wait
exit 0
