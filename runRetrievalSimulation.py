#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using the A-CCP canonical cases and corresponding architectures defined in the ACCP_ArchitectureAndCanonicalCases directory within this repo """

import os
import sys
import itertools
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
import simulateRetrieval as rs
from miscFunctions import checkDiscover
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML
from ACCP_functions import selectGeometryEntryModis, selectGeomSabrina

assert sys.version_info.major==3, 'This script requires Python 3'
if checkDiscover(): # DISCOVER
    n = int(sys.argv[1]) # (0,1,2,...,N-1)
#     nAng = int(sys.argv[2])
    nAng = 0
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/AOS/Phase-A/PLRA_RequirementsAndTraceability/GSFC_ValidationSimulationsData/V0/Run-02_')
    ymlDir = os.path.join(basePath, 'GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    geomFile = os.path.join(basePath, 'synced/AOS/Phase-A/Orbital-Viewing-Geometry-Simulations/AOS_Solstice_nc4_Files_no_view_angles/AOS_1330_LTAN_442km_alt/MAAP-GeometrySubSample_AOS_1330_LTAN_442km_alt_2023Aug12.nc4')
    Nangles = 660
#     Nangles = 4
    Nsims = 1 # number of runs (if initial guess is not random this just varies the random noise)
    maxCPU = 36 # number of cores to divide above Nsims over... we might need to do some restructuring here
else: # MacBook Air
    n = 0
    nAng = 11 # Sabrina's files have 132 x 5 = 660 angles
    saveStart = '/Users/wrespino/Desktop/TEST_V01_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
    # geomFile = '/Users/wrespino/Synced/Proposals/ROSES_TASNPP_Yingxi_2020/retrievalSimulation/NASA_Ames_MOD_angles-SZA-VZA-PHI.txt'
    geomFile = '/Users/wrespino/Synced/AOS/Phase-A/Orbital-Viewing-Geometry-Simulations/AOS_Solstice_nc4_Files_no_view_angles/AOS_1330_LTAN_442km_alt/MAAP-GeometrySubSample_AOS_1330_LTAN_442km_alt_2023Aug12.nc4'
    krnlPath = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/src/retrieval/internal_files'
    Nangles = 4 # This many angles will be processed by this call
    Nsims = 1 # number of runs (if initial guess is not random this just varies the random noise)
    maxCPU = 2 # number of cores to divide above Nsims (times Nangles?) over... we might need to do some restructuring here
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml') # will get bumped to 4 modes if needed
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')
#bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_DTarbsorb.yml')

instruments = ['polarAOS']
conCases = ['case08a', 'case08d','case08g'] # a1,a2,b1,..,o2 #180
τFactor = ['randLogNrm0.2'] #1 - Syntax error on this line? Make sure you are running Python 3!
rndIntialGuess = False # initial guess falls in middle 25% of min/max range
maxSZA = 75
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>

# parse input argument n to instrument/case
paramTple = list(itertools.product(*[instruments, conCases, τFactor]))[n] 

# building pickle save path 
saveNmTuple = paramTple[0:2] + (0 if type(paramTple[2])==str else paramTple[2],) + (n,nAng)
savePath = saveStart + '%s_%s_tFct%5.3f_n%d_nAng%d.pkl' % saveNmTuple 
print('-- Processing ' + os.path.basename(savePath) + ' --')

# setup forward and back YAML objects and now pixel
nowPix = []
for i in range(nAng, nAng+Nangles):
    if 'polaraos' in paramTple[0].lower():
        sza, phi, vza = selectGeomSabrina(geomFile, i)
    elif 'modis' in paramTple[0].lower():
        sza, phi, vza = selectGeometryEntryModis(geomFile, i)
    else:
        assert False, 'You probably want selectGeometryEntry() for this instrument but call needs to be added here'
    if sza <= maxSZA and sza>0: # this means nowPix may have slightly fewer than Nangles elements
        nowPix.append(returnPixel(paramTple[0], sza=sza, relPhi=phi, vza=vza, nowPix=None, concase=paramTple[1]))

print('n = %d, nAng = %d, len(nowPix) = %d, Nλ = %d' % (n, nAng, len(nowPix), nowPix[0].nwl))
fwdModelYAMLpath = fwdModelYAMLpathLID if 'lidar' in paramTple[0].lower() else fwdModelYAMLpathPOL
bckYAML = bckYAMLpathLID if 'lidar' in paramTple[0].lower() else bckYAMLpathPOL
fwdYAML = [setupConCaseYAML(paramTple[1], np, fwdModelYAMLpath, caseLoadFctr=paramTple[2]) for np in nowPix]

# run simulation    
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
simA.runSim(fwdYAML, bckYAML, Nsims, maxCPU=maxCPU, savePath=savePath, \
            binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
            lightSave=False, rndIntialGuess=rndIntialGuess, dryRun=False, \
            workingFileSave=False, fixRndmSeed=False, verbose=verbose)

