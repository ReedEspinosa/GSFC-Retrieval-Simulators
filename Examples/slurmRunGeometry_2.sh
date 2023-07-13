#!/usr/local/bin/bash
#SBATCH --job-name=MEGAH
#SBATCH --nodes=1 --constraint="sky|cas"
#SBATCH --time=02:59:00
#SBATCH -o ./job/28-Jul-2022-megaharp01out_2.%A-%a
#SBATCH -e ./job/28-Jul-2022-megaharp01err_2.%A-%a
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
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+30)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+33)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+36)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+39)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+42)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+45)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+48)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+51)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+54)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+57)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+60)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+63)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+66)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+69)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+72)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+75)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+78)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+81)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+84)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+87)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+90)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+93)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+96)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+99)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+102)) megaharp01 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+105)) megaharp01 30 1 &
wait
exit 0
