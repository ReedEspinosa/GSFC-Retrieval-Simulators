#!/usr/local/bin/bash
#SBATCH --job-name=GCC
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=01:20:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}", nAng="$nAng"---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $nAng &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+1)) $nAng &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+2)) $nAng &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+3)) $nAng &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+4)) $nAng &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+5)) $nAng &
#python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+6)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+7)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+8)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+9)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+10)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+11)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+12)) $nAng &
# python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+13)) $nAng &
wait
exit 0
