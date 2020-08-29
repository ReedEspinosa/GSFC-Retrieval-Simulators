#!/usr/local/bin/bash

STARTnAng=0
date
hostname
while [ $STARTnAng -lt 100 ]
do
    if [ $(squeue -u wrespino | wc -l) -lt 12 ]
    then
        echo "<><><><><><>"
        date
        echo "Running Command: sbatch --export=ALL,nAng='$STARTnAng' SLURM_runSimulation.sh"
        sbatch --export=ALL,nAng='$STARTnAng' SLURM_runSimulation.sh
        squeue -u wrespino
        STARTnAng=$(($STARTnAng+1))
        echo "<><><><><><>"
    fi
    sleep 10
done
exit 0
