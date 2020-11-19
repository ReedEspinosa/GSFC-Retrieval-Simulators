#!/usr/local/bin/bash
#SBATCH --job-name=SITA
#SBATCH --nodes=1 --constraint=sky
#SBATCH --time=01:10:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-180

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}", nAng0="$nAng"---"
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+0)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+1)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+2)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+3)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+4)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+5)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+6)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+7)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+8)) &
python runRetrievalSimulation.py ${SLURM_ARRAY_TASK_ID} $(($nAng+9)) &
wait
exit 0
