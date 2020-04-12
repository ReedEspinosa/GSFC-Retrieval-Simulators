#!/usr/local/bin/bash
#SBATCH --job-name=G_Sim
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=0:29:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-143

date
hostname
echo "---Running Sims N="${SLURM_ARRAY_TASK_ID}"+m*144, for m=0,1,...,13---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+144)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+288)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+432)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+576)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+720)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+864)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1008)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1152)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1296)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1440)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1584)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1728)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1872)) &
wait
exit 0
