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

assert sys.version_info.major==3, 'This script requires Python 3'
if checkDiscover(): # DISCOVER
    n = int(sys.argv[1]) # (0,1,2,...,N-1)
    if n>=90: sys.exit()
    nAng = int(sys.argv[2])
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM16_SITA_JuneAssessment/DRS_V01_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    rawAngleDir = os.path.join(basePath, 'synced/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs')
    PCAslctMatFilePath = os.path.join(basePath, 'synced/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat')
    lidErrDir = os.path.join(basePath, 'synced/A-CCP/Assessment_8K_Sept2020/accp_lidar_uncertainties_2020*_%s_50kmH_500mV')
    simBuildPtrn = os.path.join(basePath, 'synced/A-CCP/Assessment_8K_Sept2020/Case_Definitions/simprofile_vACCP_case%s_*.csv') #%s for case str (e.g. '8b2') and wildcard * for creation time stamp
    Nsims = 4 # number of runs (if initial guess is not random this just varies the random noise)
    maxCPU = 4 # number of cores to divide above Nsims over... we might need to do some restructuring here
else: # MacBook Air
    n = 239
    nAng = 11
    saveStart = '/Users/wrespino/Desktop/TEST_V03_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    rawAngleDir = '/Users/wrespino/Synced/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '/Users/wrespino/Synced/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    lidErrDir = '/Users/wrespino/Synced/A-CCP/Assessment_8K_Sept2020/accp_lidar_uncertainties_20200821_%s_50kmH_500mV'
    simBuildPtrn = '/Users/wrespino/Synced/A-CCP/Assessment_8K_Sept2020/Case_Definitions/simprofile_vACCP_case%s_*.csv' #%s for case str (e.g. '8b2') and wildcard * for creation time stamp
    krnlPath = None
    Nsims = 2
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml') # will get bumped to 4 modes if needed
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')
spaSetup = 'variableFineLofted+variableCoarseLofted+variableFine+variableCoarse'

instruments = ['Lidar090','Lidar050','Lidar060']
#instruments = ['Lidar090+polar07','Lidar050+polar07','Lidar090+polar07GPM','Lidar060+polar07',
#                'polar07', 'Lidar090','Lidar050','Lidar060'] # 8 N=30*1*3=90
conCases = ['case08%c%d' % (let,num) for let in map(chr, range(97, 112)) for num in [1,2]] # a1,a2,b1,..,o2 #30
# conCases = ['case08i1', 'case08i2']
τFactor = [1.0] #1 - Syntax error on this line? Make sure you are running python 3!
observeMode = 'night' # 'night' or 'day' (for lidar error file only)
cirrus = 1 # 0 -> no cirrus, 1 or 2 -> cirrus with τ=0.5 and 1.0, respectively (for lidar error file only)
rndIntialGuess = 0.90 # initial guess falls in middle 25% of min/max range
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>

# parse input argument n to instrument/case
paramTple = list(itertools.product(*[instruments, conCases, τFactor]))[n] 
# Pull PCA geometry for GPM or SS and read sim_builder profiles
orbitNow = 'GPM' if 'GPM' in paramTple[0] else 'SS'
instrumentLabel = paramTple[0]
instrmntNow = paramTple[0].replace('SS','').replace('GPM','')
SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nAng, orbit=orbitNow, verbose=verbose)
if 'case08' in paramTple[1] and 'lidar' in instrmntNow.lower():
    layAlt, profs = readSharonsLidarProfs(simBuildPtrn % paramTple[1].replace('case0',''), verbose)
else:
    layAlt, profs = (None, None)
# building pickle save path 
lidErrDir = lidErrDir % observeMode + ('' if cirrus==0 else '_Cirrus%d' % cirrus)
instrumentLabel = paramTple[0] + 'Night' if observeMode=='night' else paramTple[0] # paramTple[0] only used in save path
if cirrus>0: instrumentLabel = instrumentLabel + ('Cirrus%d' % cirrus) 
savePathInputTuple = (instrumentLabel,) + paramTple[1:3] + (orbitNow, SZA, phi, n, nAng)
savePath = saveStart + '%s_%s_tFct%4.2f_orb%s_sza%d_phi%d_n%d_nAng%d.pkl' % savePathInputTuple 
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
                               workingFileSave=True, fixRndmSeed=True, verbose=verbose)

