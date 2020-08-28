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
from ACCP_functions import selectGeometryEntry, readSharonsLidarProfs, boundBackYaml
from runGRASP import graspYAML
import numpy as np
import tempfile

if checkDiscover(): # DISCOVER
    n = int(sys.argv[1]) # (0,1,2,...,N-1)
#    nAng = int(sys.argv[2]) # index of angles to select from PCA
#     run1: ***nSLURM=0-239***, stackSLURM -> 0, 14
#     run2: ***nSLURM=0-239***, stackSLURM -> 28, 42
#     ...
#    nAng = int(n/120)*14+nAng
    nAng = 0

    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM16_SITA_JuneAssessment/DRS_V10_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    rawAngleDir = os.path.join(basePath, 'synced/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs')
    PCAslctMatFilePath = os.path.join(basePath, 'synced/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat')
    lidErrDir = os.path.join(basePath, 'synced/A-CCP/Assessment_8K_Sept2020/accp_lidar_uncertainties_20200821_day_50kmH_500mV')
    simBuildPtrn = os.path.join(basePath, 'synced/A-CCP/Assessment_8K_Sept2020/Case_Definitions/simprofile_vACCP_case%s_*.csv') #%s for case str (e.g. '8b2') and wildcard * for creation time stamp
    Nsims = 4 # number of runs (if initial guess is not random this just varies the random noise)
    maxCPU = 2 # number of cores to divide above Nsims over... we might need to do some restructuring here
else: # MacBook Air
    n = 239
    nAng = 11
    saveStart = '/Users/wrespino/Desktop/TEST_V03_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    rawAngleDir = '/Users/wrespino/Synced/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '/Users/wrespino/Synced/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    lidErrDir = '/Users/wrespino/Synced/A-CCP/Assessment_8K_Sept2020/accp_lidar_uncertainties_20200821_day_50kmH_500mV'
    simBuildPtrn = '/Users/wrespino/Synced/A-CCP/Assessment_8K_Sept2020/Case_Definitions/simprofile_vACCP_case%s_*.csv' #%s for case str (e.g. '8b2') and wildcard * for creation time stamp
    krnlPath = None
    Nsims = 2
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml') # will get bumped to 4 modes if needed
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')

instruments = ['Lidar050+polar07','Lidar090+polar07','Lidar090+polar07GPM','Lidar060+polar07',
                'polar07', 'Lidar090','Lidar050','Lidar060'] # 8 N=30*1*8=240
spaSetup = 'variableFineLofted+variableCoarseLofted+variableFine+variableCoarse'
# τFactor = [0.07, 0.08, 0.09, 0.1, 0.11] #5
# conCases = [spaSetup+surf for surf in ['', 'Desert']] # 2
conCases = ['case08%c%d' % (let,num) for let in map(chr, range(97, 112)) for num in [1,2]] # a1,a2,b1,..,o2 #30
τFactor = [1.0] #1

rndIntialGuess = 0.25 # initial guess falls in middle 25% of min/max range
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>

# parse input argument n to instrument/case
paramTple = list(itertools.product(*[instruments, conCases, τFactor]))[n] 
# Pull PCA geometry for GPM or SS and read sim_builder profiles
if 'GPM' in paramTple[0]:
    instrmntNow = paramTple[0].replace('GPM','')
    orbitNow = 'GPM'
else:
    instrmntNow = paramTple[0].replace('SS','') # SS [default] in instrument string is optional
    orbitNow = 'SS'
SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nAng, orbit=orbitNow, verbose=verbose)
if 'case08' in paramTple[1] and 'lidar' in instrmntNow.lower():
    layAlt, profs = readSharonsLidarProfs(simBuildPtrn % paramTple[1].replace('case0',''), verbose)
else:
    layAlt, profs = (None, None)
# building pickle save path 
savePath = saveStart + '%s_%s_tFct%4.2f_orb%s_sza%d_phi%d_n%d_nAng%d.pkl' % (paramTple + (orbitNow, SZA, phi, n, nAng))
savePath = savePath.replace(spaSetup, 'SPA')
print('-- Processing ' + os.path.basename(savePath) + ' --')
# setup forward and back YAML objects and now pixel
nowPix = returnPixel(instrmntNow, sza=SZA, relPhi=phi, nowPix=None, concase=paramTple[1], \
                     orbit=orbitNow, lidErrDir=lidErrDir, lidarLayers=layAlt) #(concase & orbit are only needed if using a lidar w/ Kathy's noise model
print('n = %d, nAng = %d, Nλ = %d' % (n, nAng, nowPix.nwl))
fwdModelYAMLpath = fwdModelYAMLpathLID if 'lidar' in instrmntNow.lower() else fwdModelYAMLpathPOL
bckYAMLpath = bckYAMLpathLID if 'lidar' in instrmntNow.lower() else bckYAMLpathPOL
fwdYAML = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[2], simBldProfs=profs)
bckYAML = boundBackYaml(bckYAMLpath, paramTple[1], nowPix, profs, verbose=verbose)
# run simulation    
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(fwdYAML, bckYAML, Nsims, maxCPU=maxCPU, savePath=savePath, \
                               binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
                               lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=False, \
                               workingFileSave=False, fixRndmSeed=True, verbose=verbose)

