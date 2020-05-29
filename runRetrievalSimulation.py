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
from ACCP_functions import selectGeometryEntry

# n = int(sys.argv[1]) # (0,1,2,...,N-1)
n = 12
# nAng = int(sys.argv[2]) # index of angles to select from PCA
nAng = 12

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM15_pre613SeminarApr2020/DRS_V01_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    rawAngleDir = ''
    PCAslctMatFilePath = ''
#     Nsims = 1
#     maxCPU = 1
    Nsims = 56
    maxCPU = 28
else: # MacBook Air
    saveStart = '/Users/wrespino/Desktop/TEST_V01_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    rawAngleDir = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    krnlPath = None
    Nsims = 2
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_4modes.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')


conCases = ['case06'+caseLet for caseLet in ['a','b','c','d','e','f']] # 6
# conCases = ['case06d']
τFactor = [1.0] #1
orbits = ['SS', 'GPM'] # 2
instruments = ['polar07', 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07',
               'polar0700', 'Lidar0900+polar0700','Lidar0500+polar0700','Lidar0600+polar0700'] # 8 N=96
# instruments = ['Lidar0600+polar0700'] # 8 N=42
rndIntialGuess = True # randomly vary the initial guess of retrieved parameters
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>
# AUTOMATED INPUT PREP
paramTple = list(itertools.product(*[instruments, conCases, orbits, τFactor]))[n] 
SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nAng, orbit=paramTple[2], verbose=verbose)
savePath = saveStart + '%s_%s_orb%s_tFct%4.2f_sza%d_phi%d_n%d_nAng%d.pkl' % (paramTple + (SZA, phi, n, nAng))
print('-- Processing ' + os.path.basename(savePath) + ' --')
if 'lidar' in paramTple[0].lower(): # Use LIDAR YAML file
    fwdModelYAMLpath = fwdModelYAMLpathLID
    bckYAMLpath = bckYAMLpathLID
else: # Use Polarimeter YAML file
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    bckYAMLpath = bckYAMLpathPOL
# RUN SIMULATION
nowPix = returnPixel(paramTple[0], sza=SZA, relPhi=phi, nowPix=None, \
                     concase=paramTple[1], orbit=paramTple[2]) # these last two (concase & orbit) are only needed if using a lidar w/ Kathy's noise model
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[3])
nowPix.land_prct = landPrct
print('n= %d, Nλ = %d' % (n,nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                               binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
                               lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=False, \
                               workingFileSave=True, fixRndmSeed=False, verbose=verbose)

