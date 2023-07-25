#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulations on user defined scenes and instruments """

# import some basic stuff
import os
import sys
import pprint

# add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths (assumes GRASP_scripts and GSFC-Retrieval-Simulators are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath("__file__"))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GSFC-Retrieval-Simulators – add it to Python path
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))

# import top level class that peforms the retrieval simulation, defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import returnPixel function with instrument definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel

# import setupConCaseYAML function with simulated scene definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/canonicalCaseMap.py
from canonicalCaseMap import setupConCaseYAML


# <><><> BEGIN BASIC CONFIGURATION SETTINGS <><><>

# Full path to save simulation results as a Python pickle
savePath = './job/exampleSimulationTest#1.pkl'

# Full path grasp binary
# binGRASP = '/usr/local/bin/grasp'
binGRASP = '../../GRASP_GSFC/build_megaharp01/bin/grasp'

# Full path grasp precomputed single scattering kernels
krnlPath = './src/retrieval/internal_files'

# Directory containing the foward and inversion YAML files you would like to use
ymlDir = os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases")
fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_CustomBins.yml') # foward YAML file
bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex_flatRI_darkOcean.yml') # inversion YAML file

# Other non-path related settings
Nsims = 5 # the number of inversions to perform, each with its own random noise
maxCPU = 1 # the number of processes to lssaunch, effectivly the # of CPU cores you want to dedicate to the simulation
conCase = 'dark_ocean_coarse_campex_tria_flatfine_flatcoarse_flight#01_layer#00' # conanical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
SZA = 30 # solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
Phi = 0 # relative azimuth angle, φsolar-φsensor
τFactor = 1 # scaling factor for total AOD
instrument = 'uvswirmap01' # polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options

# %% <><><> END BASIC CONFIGURATION SETTINGS <><><>

# create a dummy pixel object, conveying the measurement geometry, wavlengths, etc. (i.e. information in a GRASP SDATA file)
nowPix = returnPixel(instrument, sza=SZA, relPhi=Phi, nowPix=None)

# generate a YAML file with the forward model "truth" state variable values for this simulated scene
cstmFwdYAML = setupConCaseYAML(conCase, nowPix, fwdModelYAMLpath, caseLoadFctr=τFactor)

# Define a new instance of the simulation class for the instrument defined by nowPix (an instance of the pixel class)
simA = rs.simulation(nowPix)

# run the simulation, see below the definition of runSIM in simulateRetrieval.py for more input argument explanations
simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
            binPathGRASP=binGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, \
            rndIntialGuess=True, dryRun=False, workingFileSave=True, verbose=True)

# print some results to the console/terminal
wavelengthIndex = 2
wavelengthValue = simA.rsltFwd[0]['lambda'][wavelengthIndex]
print('RMS deviations (retrieved-truth) at wavelength of %5.3f μm:' % wavelengthValue)
pprint.pprint(simA.analyzeSim(0)[0])

# save simulated truth data to a NetCDF file
#simA.saveSim_netCDF(savePath[:-4], verbose=True)
