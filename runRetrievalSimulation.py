#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using the A-CCP canonical cases and corresponding architectures defined in the ACCP_ArchitectureAndCanonicalCases directory within this repo """

import os
import sys
import itertools
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import simulateRetrieval as rs
from miscFunctions import checkDiscover
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML
from ACCP_functions import selectGeometryEntryModis

assert sys.version_info.major==3, 'This script requires Python 3'
if checkDiscover(): # DISCOVER
    n = int(sys.argv[1]) # (0,1,2,...,N-1)
    nAng = int(sys.argv[2])
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/TASNPP_simulation00/Test11_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    geomFile = os.path.join(basePath, 'synced/Working/NASA_Ames_MOD_angles-SZA-VZA-PHI.txt')
    Nsims = 4 # number of runs (if initial guess is not random this just varies the random noise)
    maxCPU = 4 # number of cores to divide above Nsims over... we might need to do some restructuring here
else: # MacBook Air
    n = 0
    nAng = 11
    saveStart = '/Users/wrespino/Desktop/TEST_V03_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
    geomFile = '/Users/wrespino/Synced/Proposals/ROSES_TASNPP_Yingxi_2020/retrievalSimulation/NASA_Ames_MOD_angles-SZA-VZA-PHI.txt'
    krnlPath = None
    Nsims = 2
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml') # will get bumped to 4 modes if needed
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')

instruments = ['modis']
conCases = ['HuamboVegetation', 'NasaAmesVegetation'] # a1,a2,b1,..,o2 #180
τFactor = [1.0] #1 - Syntax error on this line? Make sure you are running python 3!
rndIntialGuess = 0.90 # initial guess falls in middle 25% of min/max range
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>

# parse input argument n to instrument/case
paramTple = list(itertools.product(*[instruments, conCases, τFactor]))[n] 

# SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nAng, orbit=orbitNow, verbose=verbose)
SZA, phi, vza = selectGeometryEntryModis(geomFile, nAng)

# building pickle save path 
savePathInputTuple = paramTple[0:3] + (SZA, phi, n, nAng)
savePath = saveStart + '%s_%s_tFct%5.3f_sza%d_phi%d_n%d_nAng%d.pkl' % savePathInputTuple 
print('-- Processing ' + os.path.basename(savePath) + ' --')

# setup forward and back YAML objects and now pixel
nowPix = returnPixel(paramTple[0], sza=SZA, relPhi=phi, vza=vza, nowPix=None, concase=paramTple[1])
print('n = %d, nAng = %d, Nλ = %d' % (n, nAng, nowPix.nwl))
fwdModelYAMLpath = fwdModelYAMLpathLID if 'lidar' in paramTple[0].lower() else fwdModelYAMLpathPOL
bckYAML = bckYAMLpathLID if 'lidar' in paramTple[0].lower() else bckYAMLpathPOL
fwdYAML = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[2])
# bckYAML = boundBackYaml(bckYAMLpath, paramTple[1], nowPix, verbose=verbose)

# run simulation    
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(fwdYAML, bckYAML, Nsims, maxCPU=maxCPU, savePath=savePath, \
                               binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
                               lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=False, \
                               workingFileSave=False, fixRndmSeed=False, verbose=verbose)

