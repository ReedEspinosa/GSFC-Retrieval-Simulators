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
binGRASP = '../../GRASP_GSFC/build_uvswirmap/bin/grasp'

# Full path grasp precomputed single scattering kernels
krnlPath = './src/retrieval/internal_files'

# Directory containing the foward and inversion YAML files you would like to use
ymlDir = os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases")
fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml') # foward YAML file
bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex_flatRI_darkOcean.yml') # inversion YAML file

# Other non-path related settings
Nsims = 5 # the number of inversions to perform, each with its own random noise
maxCPU = 1 # the number of processes to lssaunch, effectivly the # of CPU cores you want to dedicate to the simulation
conCase = 'dark_ocean_fixedcoarse_campex_bi_flatfine_flatcoarse_flight#01_layer#00' # conanical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
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

# %% <><><> TEST AREA <><><>

# We want to cross check the results of the simulation with the results of the inversion
# The chi2 values should be cross checked, since we are specifying the noise for Q and U instead of Q/I and U/I
import numpy as np
def chi2calc(x,y,sigma, noiseType='absolute'):
    ''' calculate chi2 for two arrays of data, x and y, with noise of sigma
    Parameters
    ----------
    Input
    -----
    x : array_like
        true array of data
    y : array_like
        retrieved array of data
    sigma : scalar
        noise value
    noiseType : str
        'absolute' or 'relative' noise
    
    Returns
    -------
    chi2 : scalar
        chi2 value
    
        '''
    if noiseType == 'absolute':
        # calculate chi2 for absolute noise
        return np.sum(((x-y)/sigma)**2)/(len(x)-1)
    elif noiseType == 'relative':
        # calculate chi2 for relative noise
        return np.sum((((x-y)/x)/sigma)**2)/(len(x)-1)

# A simple test case to check the chi2calc function
c1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8, 9, 10, 11, 12])
c2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1])
c3 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 9.0, 11.1, 10.8])

print(chi2calc(c1,c2,sigma=0.1))
print(chi2calc(c1,c3,sigma=0.1, noiseType='relative'))

# Now we want to check the chi2 values for the simulation and inversion'

# calculate chi2 based for different wavelengths
I_noise = 0.03 # 3% noise in I
Q_noise = 0.005 # 0.5% noise in Q/I
U_noise = 0.005 # 0.5% noise in U/I
for i in range(len(simA.rsltFwd[0]['lambda'])):
    for j in range (Nsims):
        print('chi2 for wavelength %5.3f μm:' % simA.rsltFwd[0]['lambda'][i])
        print('chi2 in I:       %1.4f' % chi2calc(simA.rsltBck[j]['meas_I'][:,i], simA.rsltBck[j]['fit_I'][:,i], I_noise, noiseType='relative'))
        print('chi2 in I(/n):   %1.4f' % chi2calc(simA.rsltFwd[0]['fit_I'][:,i], simA.rsltBck[j]['fit_I'][:,i], I_noise, noiseType='relative'))
        print('chi2 in Q/I:     %1.4f' % chi2calc(simA.rsltBck[j]['meas_QoI'][:,i], simA.rsltBck[j]['fit_QoI'][:,i], Q_noise))
        print('chi2 in U/I:     %1.4f' % chi2calc(simA.rsltBck[j]['meas_UoI'][:,i], simA.rsltBck[j]['fit_UoI'][:,i], U_noise))

# %% <><><> END TEST AREA <><><>