#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

import os
import sys
import functools
import subprocess
from datetime import datetime, timedelta

# add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths
parentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 
sys.path.append(parentDir) 
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))
grandParentDir = os.path.dirname(parentDir)
sys.path.append(os.path.join(grandParentDir, "gsfc-grasp-python-interface"))

# Import necessary simulation modules
import simulateRetrieval as rs
from readOSSEnetCDF import osseData
from architectureMap import returnPixel, addError

# =========================================================================
# CONFIGURATION
# =========================================================================
# Define your start and end dates here
start_date = datetime(2020, 8, 1)  # Modify to your actual start date
end_date = datetime(2020, 8, 3)    # Modify to your actual end date

# SLURM Job Settings
account = "1180"
maxCPU = 10 # This will dictate the number of tasks/cores requested
walltime = "02:00:00" # Walltime per individual hour-job

# Simulation / Paths Configuration
dirGRASP = '/home/dgiles/nobackup/AIST/grasp/build/bin/grasp'
krnlPath = '/home/dgiles/nobackup/AIST/software/grasp/src/retrieval/internal_files/'
osseDataPath = '/home/dgiles/nobackup/AIST/software/GMAO_Nature_Run/'
bckYAMLpath = os.path.join(parentDir, 'ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_2modes_V1.0.0.yml')

rndIntialGuess = False
random = False
orbit = 'ss450'
maxSZA = 60
oceanOnly = True
noiseFree = True
customOutDir = "/home/dgiles/nobackup/AIST/software/OSSE_Test_Run/" # Define this if needed, or set to None

# =========================================================================
# SLURM ARRAY SUBMISSION LOGIC
# =========================================================================
# Calculate total number of hours to process
total_hours = int((end_date - start_date).total_seconds() // 3600)

if "SLURM_ARRAY_TASK_ID" not in os.environ:
    # We are NOT in a SLURM job yet. Generate the batch file and submit.
    slurm_script_name = "submit_osse_array.sh"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Generate the SLURM submission script requesting tasks (cores) instead of nodes
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=OSSE_Retrieval
#SBATCH --account={account}
#SBATCH --ntasks={maxCPU}          # Requesting resources by tasks/cores, not nodes
#SBATCH --time={walltime}
#SBATCH --array=0-{total_hours - 1} # Array indices matching the total hours
#SBATCH --output=logs/osse_%A_%a.out
#SBATCH --error=logs/osse_%A_%a.err

# Execute the python script
python {os.path.abspath(__file__)}
"""
    with open(slurm_script_name, "w") as f:
        f.write(slurm_script_content)
        
    print(f"Submitting SLURM array job to NCCS DISCOVER with {total_hours} tasks...")
    subprocess.run(["sbatch", slurm_script_name])
    sys.exit(0)

# =========================================================================
# SINGLE TASK PROCESSING LOGIC (RUNS INSIDE SLURM JOB)
# =========================================================================
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

# Determine the specific date and hour this task is responsible for
current_date = start_date + timedelta(hours=task_id)
year = current_date.year
month = current_date.month
day = current_date.day
hour = current_date.hour

print(f"\n=======================================================")
print(f"Processing OSSE Data for: {year:04d}-{month:02d}-{day:02d} {hour:02d}:00Z (Task ID: {task_id})")
print(f"=======================================================")

try:
    # create osseData instance w/ pixels from specified date/time
    od = osseData(osseDataPath, orbit, year, month, day, hour, random=random, wvls=None, pixInd=None,
                  lidarVersion=None, maxSZA=maxSZA, oceanOnly=oceanOnly, loadPSD=True, verbose=True)
    
    # extract the simulated observations and pack them in GRASP_scripts rslts dictionary format
    fwdData = od.osse2graspRslts()
    
    # build file name to save the results. Adding a timestamp string to keep filenames unique.
    time_str = f"{year:04d}{month:02d}{day:02d}_{hour:02d}00Z"
    
    # Assuming `vrsn` was defined somewhere in your script
    vrsn = 'v1' 
    base_filename = od.fpDict['savePath'] % (vrsn, f'example_{time_str}', 'polarimeter07')
    
    if customOutDir: 
        # Create the YYYY/MM/DD folder structure string
        daily_out_dir = os.path.join(customOutDir, f"{year:04d}", f"{month:02d}", f"{day:02d}")
        # Make sure the directory exists (exist_ok=True prevents errors if it's already there)
        os.makedirs(daily_out_dir, exist_ok=True)
        
        # Set the save path to be inside the newly created daily directory
        savePath = os.path.join(daily_out_dir, os.path.basename(base_filename))
    else:
        savePath = base_filename
        
    print('-- Running simulation for ' + savePath + ' --')
    
    # define a new instance of the simulation class
    simA = rs.simulation()
    
    # run the retrievals
    simA.runSim(fwdData, bckYAMLpath, maxCPU=maxCPU, maxT=20, savePath=savePath,
                binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True,
                rndIntialGuess=rndIntialGuess, radianceNoiseFun=None, # Ensure radNoiseFun is defined if you use it
                workingFileSave=True, dryRun=False, verbose=True)
    
    simA.saveSim_netCDF(savePath[:-4], verbose=True)
    
except Exception as e:
    print(f"FAILED to process {year:04d}-{month:02d}-{day:02d} {hour:02d}:00Z. Error: {e}")


