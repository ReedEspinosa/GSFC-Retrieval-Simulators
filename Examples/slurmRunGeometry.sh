#!/usr/local/bin/bash
#SBATCH --job-name=HARP2
#SBATCH --nodes=1 --constraint="sky|cas"
#SBATCH --time=02:55:00
#SBATCH -o ./job/28-Jul-2022-harp02out.%A-%a
#SBATCH -e ./job/28-Jul-2022-harp02err.%A-%a
#SBATCH --array=0-2

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+0)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+3)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+6)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+9)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+12)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+15)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+18)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+21)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+24)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+27)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+30)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+33)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+36)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+39)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+42)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+45)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+48)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+51)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+54)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+57)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+60)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+63)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+66)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+69)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+72)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+75)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+78)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+81)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+84)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+87)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+90)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+93)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+96)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+99)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+102)) harp02 30 1 &
python runRetrievalSimulationSlurm.py $((${SLURM_ARRAY_TASK_ID}+105)) harp02 30 1 &
wait
exit 0
