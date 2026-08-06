#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import functools
from datetime import datetime

# Check if running within SLURM array
if "SLURM_ARRAY_TASK_ID" not in os.environ:
    print("Error: SLURM_ARRAY_TASK_ID not found. This script must be submitted via SLURM array.")
    sys.exit(1)

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

# =========================================================================
# DATETIME DECODING LOGIC
# =========================================================================
year = 2006
target_months = [3, 8, 9, 12]

tasks_per_day = 24
tasks_per_month = 14 * 24  # 336 tasks per month (14 days * 24 hours)

# Calculate exactly which month, day, and hour this task_id represents
month_index = task_id // tasks_per_month
month = target_months[month_index]

# Find the remaining tasks for the specific month to calculate day and hour
remainder_in_month = task_id % tasks_per_month
day = (remainder_in_month // tasks_per_day) + 1  # +1 because days are 1-indexed (1 to 14)
hour = remainder_in_month % tasks_per_day

# =========================================================================
# CONFIGURATION
# =========================================================================
vrsn = 120
wvls = [0.355, 0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65]
pixInd = None
customOutDir = '/home/dgiles/nobackup/AIST/software/OSSE_Test_Run'
noiseFree = True
maxCPU = 100
rndIntialGuess = False
random = False
orbit = 'ss450'
maxSZA = 60
oceanOnly = True

# Add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths
parentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) 
sys.path.append(parentDir) 
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))
grandParentDir = os.path.dirname(parentDir)
sys.path.append(os.path.join(grandParentDir, "gsfc-grasp-python-interface"))

import simulateRetrieval as rs
from readOSSEnetCDF import osseData
from architectureMap import returnPixel, addError

# Updated GRASP executable path
dirGRASP = '/home/dgiles/nobackup/AIST/software/grasp/build/bin/grasp'
krnlPath = '/home/dgiles/nobackup/AIST/software/grasp/src/retrieval/internal_files/'
osseDataPath = '/home/dgiles/nobackup/AIST/software/GMAO_Nature_Run/'
bckYAMLpath = os.path.join(parentDir, 'ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_2modes_V1.0.0.yml')

radNoiseFun = None if noiseFree else functools.partial(addError, 'polar07')

# =========================================================================
# EXECUTION
# =========================================================================
print(f"\n=======================================================")
print(f"Processing OSSE Data for: {year:04d}-{month:02d}-{day:02d} {hour:02d}:00Z (Task ID: {task_id})")
print(f"=======================================================")

try:
    # create osseData instance w/ pixels from specified date/time
    od = osseData(osseDataPath, orbit, year, month, day, hour, random=random, wvls=wvls, pixInd=pixInd,
                  lidarVersion=None, maxSZA=maxSZA, oceanOnly=oceanOnly, loadPSD=True, verbose=True)
    
    # extract the simulated observations and pack them in GRASP_scripts rslts dictionary format
    fwdData = od.osse2graspRslts()
    
    # build file name to save the results. Adding a timestamp string to keep filenames unique.
    time_str = f"{year:04d}{month:02d}{day:02d}_{hour:02d}00Z"
    base_filename = od.fpDict['savePath'] % (vrsn, f'example_{time_str}', 'polarimeter07')
    
    if customOutDir: 
        daily_out_dir = os.path.join(customOutDir, f"{year:04d}", f"{month:02d}", f"{day:02d}")
        os.makedirs(daily_out_dir, exist_ok=True)
        savePath = os.path.join(daily_out_dir, os.path.basename(base_filename))
    else:
        savePath = base_filename
        
    print('-- Running simulation for ' + savePath + ' --')
    
    # define a new instance of the simulation class
    simA = rs.simulation()
    
    # run the retrievals
    simA.runSim(fwdData, bckYAMLpath, maxCPU=maxCPU, maxT=20, savePath=savePath,
                binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True,
                rndIntialGuess=rndIntialGuess, radianceNoiseFun=radNoiseFun,
                workingFileSave=True, dryRun=False, verbose=True)
    
    simA.saveSim_netCDF(savePath[:-4], verbose=True)
    
except Exception as e:
    print(f"FAILED to process {year:04d}-{month:02d}-{day:02d} {hour:02d}:00Z. Error: {e}")

