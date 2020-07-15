#!/usr/local/bin/bash
#SBATCH --job-name=GOSSE
#SBATCH --nodes=1 --constraint=hasw
#SBATCH --time=01:59:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --array=0

date
hostname
echo "---Running Sims---"
python runRetrievalOSSE.py
exit 0
