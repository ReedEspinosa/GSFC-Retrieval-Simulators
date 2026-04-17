#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

# import some standard modules
import os
import sys
import functools
from datetime import datetime, timedelta

# add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths (assumes GRASP_scripts and GSFC-Retrieval-Simulators are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # obtain [THIS_FILE_PATH]/../ in POSIX
sys.path.append(parentDir) # that should be GSFC-Retrieval-Simulators – add it to Python path
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))
grandParentDir = os.path.dirname(parentDir)# [THIS_FILE_PATH]/../../ in POSIX (this is folder that contains GRASP_scripts and GSFC-Retrieval-Simulators
sys.path.append(os.path.join(grandParentDir, "GSFC-GRASP-Python-Interface"))

# import top level class that peforms the actual retrieval simulation; defined in [THIS_FILE_PATH]/../simulateRetrieval.py
import simulateRetrieval as rs

# import the class that is used to read Patricia's OSSE data; defined in ...GSFC-Retrieval-Simulators/readOSSEnetCDF.py
from readOSSEnetCDF import osseData

# import returnPixel and addError functions with instrument definitions from ...GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel, addError

dirGRASP = '/home/dgiles/AIST_GRASP/grasp/build/bin/grasp'
krnlPath = '/home/dgiles/AIST_GRASP/grasp/src/retrieval/internal_files/'
osseDataPath = '/home/dgiles/AIST_GRASP/testCase_Aug01_0000Z_VersionJune2020/testCase_Aug01_0000Z_VersionJune2020/'

# define other paths not having to do with the python code itself
bckYAMLpath = os.path.join(parentDir, 'ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_2modes_V1.0.0.yml') # location of retrieval YAML file (V1.0.0 for VLIDORTMatch branch of GRASP)

# if retrievals are divided up into multiple calls to GRASP, ensure the number of simultaneous processes is always ≤maxCPU
maxCPU = 100

# randomize initial guess in YAML file before retrieving
rndIntialGuess = False

# Must be False to use specific days and hours defined in the loop below
random = False

# simulated orbit to use – gpm OR ss450
orbit = 'ss450'

# filter out pixels with mean SZA above this value (degrees)
maxSZA = 60

# true to skip retrievals on land pixels
oceanOnly = True

# If true no noise will be added to simulated measurements, else noise is added according to architectureMap.py
noiseFree = True

# general version integer to distinguish output files of different runs
vrsn = 120

# wavelengths (μm); if we only want specific λ set it here, otherwise use every λ found in the netCDF files
wvls = [0.355, 0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65]

# specific pixels to run; set to None to run all pixels (computationally heavy)
pixInd = None

# Base output directory
customOutDir = '/home/dgiles/AIST_GRASP/OSSE_Test_Run'

# Set noise model to added to polarimeter measurements, polar07 error is defined in addError() method of architectureMap.py
radNoiseFun = None if noiseFree else functools.partial(addError, 'polar07')

# =============================================================================
# Time loop for running retrievals across a date range
# =============================================================================

# Define your start and end dates here
start_date = datetime(2006, 8, 1)
end_date = datetime(2006, 8, 3) # Example: Runs up to (but not including) Aug 3

current_date = start_date

while current_date < end_date:
    for hour in range(24): # Iterate through each hour (0 to 23)
        year = current_date.year
        month = current_date.month
        day = current_date.day
        
        print(f"\n=======================================================")
        print(f"Processing OSSE Data for: {year:04d}-{month:02d}-{day:02d} {hour:02d}:00Z")
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
                        rndIntialGuess=rndIntialGuess, radianceNoiseFun=radNoiseFun,
                        workingFileSave=True, dryRun=False, verbose=True)
            
            simA.saveSim_netCDF(savePath[:-4], verbose=True)
            
        except Exception as e:
            print(f"FAILED to process {year:04d}-{month:02d}-{day:02d} {hour:02d}:00Z. Error: {e}")
            
    # Move to the next day
    current_date += timedelta(days=1)

