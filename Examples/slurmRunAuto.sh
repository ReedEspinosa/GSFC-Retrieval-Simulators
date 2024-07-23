#!/bin/bash

# This script is used to run the slurm_run.py script on the nyx or discover machines
# It will run the script a number of times based on the machine it is running on

# if the argument is --dry-run, then just print the commands that would be run

# Prompt user: "Are you want to run the slurm_run.py script? (yes/no)"
# Read user input
read -p "Are you want to run the slurm_run.py script? (yes/no) " -n 3 -r

if [ "$REPLY" != "yes" ]; then
    echo "Exiting (use --dry-run to see the commands that would be run)"
    exit 1
fi

if [ "$1" == "--dry-run" ]; then
    echo "Dry run"
    # Set the dry run flag
    set dry_run

else
    # Unset the dry run flag
    unset dry_run
fi

if hostname | grep -q 'nyx'; then
    echo "Running on nyx"
    # Set the number of runs
    runs=4

elif hostname | grep -q 'discover'; then
    echo "Running on discover"
    # Set the number of runs
    runs=2
else
    echo "Running on unknown machine"
    exit 2
fi

# loop over the number of runs
for x in $(seq 0 $runs);
do
    # Run the python script using slurm
    if [ -z $dry_run ]; then
        echo "python slurm_run.py $x --dry-run"
        echo "python slurm_run.py $x Triangular --dry-run"
    else
        python slurm_run.py $x
        python slurm_run.py $x Triangular
    fi
done

