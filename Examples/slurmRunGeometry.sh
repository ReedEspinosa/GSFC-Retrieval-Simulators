#!/usr/local/bin/bash
#SBATCH --job-name=MULTI
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=00:55:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-2

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+0)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+3)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+6)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+9)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+12)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+15)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+18)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+21)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+24)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+27)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+31)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+34)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+37)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+40)) megaharp01 30 1 &
wait
exit 0