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
from runGRASP import graspYAML
import numpy as np
import tempfile


if checkDiscover(): # DISCOVER
    n = int(sys.argv[1]) # (0,1,2,...,N-1)
    nAng = int(sys.argv[2]) # index of angles to select from PCA
#     run1: ***nSLURM=0-239***, stackSLURM -> 0, 14
#     run2: ***nSLURM=0-239***, stackSLURM -> 28, 42
#       ...
    nAng = int(n/120)*14+nAng
    n = n%120 

        
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM16_SITA_JuneAssessment/DRS_V10_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    rawAngleDir = os.path.join(basePath, 'synced/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs')
    PCAslctMatFilePath = os.path.join(basePath, 'synced/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat')
#     lidErrDir = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized_5kmH_500mV')
    lidErrDir = os.path.join(basePath, 'synced/A-CCP/lidarUncertainties/organized')
#     Nsims = 1
#     maxCPU = 1
    Nsims = 2
    maxCPU = 2
else: # MacBook Air
    n = 19
    nAng = 11
    saveStart = '/Users/wrespino/Desktop/TEST_V03_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    rawAngleDir = '/Users/wrespino/Synced/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '/Users/wrespino/Synced/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    lidErrDir = '/Users/wrespino/Synced/A-CCP/lidarUncertainties/organized'
    krnlPath = None
    Nsims = 2
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_4modes.yml')
bckYAMLpathLIDveg = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_VEG_10Vbins_4modes.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')
bckYAMLpathPOLveg = os.path.join(ymlDir, 'settings_BCK_POLAR_VEG_2modes.yml')

# casLets = list(map(chr, range(97, 106))) # 'a' - 'i'
# conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['', 'Desert', 'Vegetation']] # 9x3=27
τFactor = [0.07, 0.08, 0.09, 0.1, 0.11] #5
spaSetup = 'variableFineLofted+variableCoarseLofted+variableFine+variableCoarse'
conCases = [spaSetup+surf for surf in ['', 'Desert']] # 3
# orbits = ['SS', 'GPM'] # 2
orbits = ['SS'] # 1
instruments = ['Lidar09+polar07','Lidar09+polar07GPM','Lidar05+polar07','Lidar06+polar07', \
                'polar07', 'Lidar09','Lidar05','Lidar06'] # 8 N=5*3*8=120

rndIntialGuess = True # randomly vary the initial guess of retrieved parameters
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>
# AUTOMATED INPUT PREP
paramTple = list(itertools.product(*[instruments, conCases, orbits, τFactor]))[n] 

if 'GPM' in paramTple[0]:
    instrmntNow = paramTple[0].replace('GPM','')
    orbitNow = 'GPM'
else:
    instrmntNow = paramTple[0]
    orbitNow = 'SS'
        
SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nAng, orbit=orbitNow, verbose=verbose)
savePath = saveStart + '%s_%s_orb%s_tFct%4.2f_sza%d_phi%d_n%d_nAng%d.pkl' % (paramTple + (SZA, phi, n, nAng))
savePath = savePath.replace(spaSetup, 'SPA')
print('-- Processing ' + os.path.basename(savePath) + ' --')
if 'lidar' in instrmntNow.lower(): # Use LIDAR YAML file
    fwdModelYAMLpath = fwdModelYAMLpathLID
    bckYAMLpathOrg = bckYAMLpathLIDveg if 'Vegetation' in paramTple[1] else bckYAMLpathLID
    # Δn = ±0.02 and Δk = ±0.001 
    randomID = hex(np.random.randint(0, 2**63-1))[2:] # needed to prevent identical FN w/ many parallel runs
    newFn = 'settingsBckYAML_conCase%s_tuned%s.yml' % (paramTple[1], randomID)
    bckYAMLpath = os.path.join(tempfile.gettempdir(), newFn)
    bckYAMLObj = graspYAML(baseYAMLpath=bckYAMLpathOrg, workingYAMLpath=bckYAMLpath)
else: # Use Polarimeter YAML file
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    bckYAMLpath = bckYAMLpathPOLveg if 'Vegetation' in paramTple[1] else bckYAMLpathPOL
# RUN SIMULATION
nowPix = returnPixel(instrmntNow, sza=SZA, relPhi=phi, nowPix=None, \
                     concase=paramTple[1], orbit=orbitNow, lidErrDir=lidErrDir) # these last two (concase & orbit) are only needed if using a lidar w/ Kathy's noise model
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[3])
nowPix.land_prct = landPrct

if 'lidar' in instrmntNow.lower(): # implement ACCP SIT-A specific RI contraints 
    fwdYAMLObj = graspYAML(baseYAMLpath=cstmFwdYAML)
    # adjust BRDF ranges, which will change for lidar05/09 w/o 355 nm
    if not 'lidar06' in paramTple[0].lower():
        val = bckYAMLObj.access('surface_land_brdf_ross_li.1.min')
        bckYAMLObj.access('surface_land_brdf_ross_li.1.min', val[1:])
        val = bckYAMLObj.access('surface_land_brdf_ross_li.1.max')
        bckYAMLObj.access('surface_land_brdf_ross_li.1.max', val[1:])
    # implement ACCP SIT-A specific RI contraints
    val = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.2.value'))
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.2.min', [max(val-0.001, 1e-8)])
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.2.max', [val+0.001])
    val = np.mean(fwdYAMLObj.access('real_part_of_refractive_index_spectral_dependent.2.value'))
    bckYAMLObj.access('real_part_of_refractive_index_constant.2.min', [val-0.02])
    bckYAMLObj.access('real_part_of_refractive_index_constant.2.max', [val+0.02])
    val = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.4.value'))
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.4.min', [max(val-0.001, 1e-8)])
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.4.max', [val+0.001])
    val = np.mean(fwdYAMLObj.access('real_part_of_refractive_index_spectral_dependent.4.value'))
    bckYAMLObj.access('real_part_of_refractive_index_constant.4.min', [val-0.02])
    bckYAMLObj.access('real_part_of_refractive_index_constant.4.max', [val+0.02])
    # break with procedure starts here... (these are fine mode)
    val = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.1.value'))
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.1.max', [val+0.001])
    val = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.3.value'))
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.3.max', [val+0.001])

print('n = %d, nAng = %d, Nλ = %d' % (n, nAng, nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                               binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
                               lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=False, \
                               workingFileSave=False, fixRndmSeed=True, verbose=verbose)

