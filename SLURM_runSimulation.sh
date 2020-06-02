#!/usr/local/bin/bash
#SBATCH --job-name=GSS7
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=01:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0-131

STARTnAng=$1
date
hostname
echo "---Running Sims n="${SLURM_ARRAY_TASK_ID}"  STARTnAng="$STARTnAng"---"

python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((0+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((1+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((2+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((3+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((4+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((5+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((6+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((7+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((8+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((9+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((10+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((11+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((12+$STARTnAng)) &
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0)) $((13+$STARTnAng)) &
wait
exit 0
