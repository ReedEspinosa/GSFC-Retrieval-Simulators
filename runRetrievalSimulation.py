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

n = int(sys.argv[1]) # (0,1,2,...,N-1)
# n = 0
nAng = int(sys.argv[2]) # index of angles to select from PCA
# nAng = 2

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM16_SITA_JuneAssessment/DRS_V00_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    rawAngleDir = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs')
    PCAslctMatFilePath = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat')
    lidErrDir = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized')
#     Nsims = 1
#     maxCPU = 1
    Nsims = 7
    maxCPU = 7
else: # MacBook Air
    saveStart = '/Users/wrespino/Desktop/TEST_V02_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    rawAngleDir = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    lidErrDir = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized'
    krnlPath = None
    Nsims = 2
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_4modes.yml')
bckYAMLpathLIDveg = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_VEG_10Vbins_4modes.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')
bckYAMLpathPOLveg = os.path.join(ymlDir, 'settings_BCK_POLAR_VEG_2modes.yml')

casLets = list(map(chr, range(97, 108))) # 'a' - 'k'
conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['', 'Desert', 'Vegetation']] # 11x3=33
# conCases = ['case06aVegetation']
τFactor = [1.0] #1
# orbits = ['SS', 'GPM'] # 2
orbits = ['SS'] # 1
instruments = ['polar07', 'Lidar09','Lidar05','Lidar06', \
               'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=231
# instruments = ['lidar0600'] # 8 N=42
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
    bckYAMLpath = bckYAMLpathLIDveg if 'Vegetation' in paramTple[1] else bckYAMLpathLID
else: # Use Polarimeter YAML file
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    bckYAMLpath = bckYAMLpathPOLveg if 'Vegetation' in paramTple[1] else bckYAMLpathPOL
# RUN SIMULATION
nowPix = returnPixel(paramTple[0], sza=SZA, relPhi=phi, nowPix=None, \
                     concase=paramTple[1], orbit=paramTple[2], lidErrDir=lidErrDir) # these last two (concase & orbit) are only needed if using a lidar w/ Kathy's noise model
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[3])
nowPix.land_prct = landPrct
print('n = %d, nAng = %d, Nλ = %d' % (n, nAng, nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                               binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
                               lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=False, \
                               workingFileSave=True, fixRndmSeed=True, verbose=verbose)

