#!/usr/local/bin/bash

STARTnAng=10
date
hostname
while [ $STARTnAng -lt 100 ]
do
    if [ $(squeue -u wrespino | grep SITA | wc -l) -lt 1 ]
    then
        echo "<><><><><><>"
        date
        echo "Running Command: sbatch --export=ALL,nAng=$STARTnAng SLURM_runSimulation.sh"
        sbatch --export=ALL,nAng=$STARTnAng SLURM_runSimulation.sh
        squeue -u wrespino
        STARTnAng=$(($STARTnAng+10))
        echo "<><><><><><>"
    fi
    sleep 30
done
exit 0
