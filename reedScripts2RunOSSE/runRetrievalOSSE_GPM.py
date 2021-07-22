#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

# import some standard modules
import os
import sys
import functools
import numpy as np

nArg = int(sys.argv[1])

# add GRASP_scripts, MADCAP_Scripts and ACCP subfolder to path (assumes GRASP_scripts and MADCAP_scripts are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GRASP_Scripts – add it to Python path
grandParentDir = os.path.dirname(parentDir)# THIS_FILE_PATH/../../ in POSIX (this is folder that contains GRASP_scripts and MADCAP_scripts
sys.path.append(os.path.join(grandParentDir, "MADCAP_scripts"))
sys.path.append(os.path.join(grandParentDir, "MADCAP_scripts","ACCP_ArchitectureAndCanonicalCases"))

# import top level class that peforms the actual retrieval simulation; defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import the class that is used to read Patricia's OSSE data; defined in ...MADCAP_scripts/readOSSEnetCDF.py
from readOSSEnetCDF import osseData

# import returnPixel and addError functions with instrument definitions from ...MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel, addError

# define other paths not having to do with the python code itself
basePath = os.environ['NOBACKUP']
#bckYAMLpath = os.path.join(basePath, 'MADCAP_scripts','ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_2modes.yml') # location of retrieval YAML file
#bckYAMLpath = os.path.join(basePath, 'MADCAP_scripts','ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_16bins.yml') # location of retrieval YAML file
bckYAMLpath = os.path.join(basePath, 'MADCAP_scripts','ACCP_ArchitectureAndCanonicalCases','settings_BCK_POLAR_3modes.yml') # location of retrieval YAML file
dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp') # location of the GRASP binary to use for retrievals
krnlPath = os.path.join(basePath, 'local/share/grasp/kernels') # location of GRASP kernel files
osseDataPath = '/discover/nobackup/projects/gmao/osse2/pub/c1440_NR/OBS/A-CCP/' # base path for A-CCP OSSE data (contains gpm and ss450 folders)
# if retrievals are divided up into multiple calls to GRASP, ensure the number of simultaneous processes is always ≤maxCPU
maxCPU = 28

# randomize initial guess in YAML file before retrieving
rndIntialGuess = False

# True to use 10k randomly chosen samples for defined month, or False to use specific day and hour defined below
random = True

# point in time to pull OSSE data (if random==true then day and hour below can be ignored)
year = 2006
month = int(np.floor(nArg/10))
day = 1
hour = 0
#day = int(20+np.floor(nArg/24))
#hour = np.mod(nArg,24)

# simulated orbit to use – gpm OR ss450
orbit = 'gpm'

# filter out pixels with mean SZA above this value (degrees)
maxSZA = 60

# true to skip retrievals on land pixels
oceanOnly = False

# If true no noise will be added to simulated measurements, else noise is added according to architectureMap.py (or Ed's simulations for lidar data)
noiseFree = False

# general version integer to distinguish output files of different runs
chnkInd = np.mod(nArg, 10)
vrsn = 300+chnkInd

# wavelengths (μm); if we only want specific λ set it here, otherwise use every λ found in the netCDF files
wvls = [0.355, 0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65]

# specific pixels to run; set to None to run all pixels (likely very slow)
#pixInd = [2560+i*3 for i in range(252)] # Aug 1, 1400Z, stripe up atlantic w/ marine, BB, marine, dust, dustymarine(?)
# pixInd = None
#pixInd = [3122, 3122] # works to test with Aug 2, 1000Z
chunkSize=1000
pixInd = np.linspace(chnkInd*chunkSize, (chnkInd+1)*chunkSize-1, chunkSize, dtype=np.int)

# save output here instead of within osseDataPath (None to disable)
#customOutDir = os.path.join(basePath, 'synced', 'Working', 'OSSE_Test_Run')
customOutDir = '/discover/nobackup/wrespino/OSSE_results_working'

# create osseData instance w/ pixels from specified date/time (detail on these arguments in comment near top of osseData class's __init__ near readOSSEnetCDF.py:30)
od = osseData(osseDataPath, orbit, year, month, day, hour, random=random, wvls=wvls, pixInd=pixInd,
              lidarVersion=None, maxSZA=maxSZA, oceanOnly=oceanOnly, loadPSD=True, verbose=True)
# extract the simulated observations and pack them in GRASP_scripts rslts dictionary format
fwdData = od.osse2graspRslts()

# build file name to save the results
savePath = od.fpDict['savePath'] % (vrsn, 'example', 'polarimeter07')
if customOutDir: savePath = os.path.join(customOutDir, os.path.basename(savePath))
print('-- Running simulation for ' + os.path.basename(savePath) + ' --') # we print this because savePath has useful information about OSSE data
# Set noise model to added to polarimeter measurements, polar07 error is defined in addError() method of architectureMap.py
radNoiseFun = None if noiseFree else functools.partial(addError, 'polar07')

# define a new instance of the simulation class
simA = rs.simulation()

# run the retrievals (these arguments are explained at the top of the runSim method definition near simulateRetrieval.py:35)
simA.runSim(fwdData, bckYAMLpath, maxCPU=maxCPU, maxT=20, savePath=savePath,
            binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True,
            rndIntialGuess=rndIntialGuess, radianceNoiseFun=radNoiseFun,
            workingFileSave=False, dryRun=False, verbose=True)

simA.saveSim_netCDF(savePath[:-4], verbose=True)
