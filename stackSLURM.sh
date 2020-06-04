#!/usr/local/bin/bash

STARTnAng=0
date
hostname
while [ $STARTnAng -lt 73 ]
do
    if [ $(squeue -u wrespino | wc -l) -lt 2 ]
    then
        echo "<><><><><><>"
        date
        echo "Running Command: sbatch SLURM_runSimulation.sh $STARTnAng"
        sbatch SLURM_runSimulation.sh $STARTnAng
        squeue -u wrespino
        STARTnAng=$(($STARTnAng+28))
        echo "<><><><><><>"
    fi
    sleep 30
done
exit 0
