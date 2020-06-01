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
        
#     run1: nSLURM=0-197 -> n=0-98, nAng=0,14 (last nAng will be 27); no filename offset (this is SS 1st collection)
    nAng = int(n/99)*14+nAng
    n = n%99
    
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM16_SITA_JuneAssessment/DRS_V04_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    rawAngleDir = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs')
    PCAslctMatFilePath = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat')
    lidErrDir = os.path.join(basePath, 'synced/Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized')
#     Nsims = 1
#     maxCPU = 1
    Nsims = 2
    maxCPU = 2
else: # MacBook Air
    n = 28
    nAng = 2
    saveStart = '/Users/wrespino/Desktop/TEST_V03_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    rawAngleDir = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    lidErrDir = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized'
    krnlPath = None
    Nsims = 1
    maxCPU = 1
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml')
bckYAMLpathLIDveg = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_VEG_10Vbins_2modes.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')
bckYAMLpathPOLveg = os.path.join(ymlDir, 'settings_BCK_POLAR_VEG_2modes.yml')

casLets = list(map(chr, range(97, 108))) # 'a' - 'k'
conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['', 'Desert', 'Vegetation']] # 11x3=33
spaSetup = 'variableFineLofted+variableCoarseLofted+variableFine+variableCoarse'
# conCases = [spaSetup+surf for surf in ['', 'Desert', 'Vegetation']] # 3
τFactor = [1.0] #3
# orbits = ['SS', 'GPM'] # 2
orbits = ['SS'] # 1
# instruments = ['polar07', 'Lidar09','Lidar05','Lidar06', \
#                 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=3*3*7=63
instruments = ['Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 3 N=99
rndIntialGuess = True # randomly vary the initial guess of retrieved parameters
verbose = True
# more specific simulation options in runSim call below... 

# <><><><>END INPUTS<><><><>
# AUTOMATED INPUT PREP
paramTple = list(itertools.product(*[instruments, conCases, orbits, τFactor]))[n] 
SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nAng, orbit=paramTple[2], verbose=verbose)
savePath = saveStart + '%s_%s_orb%s_tFct%4.2f_sza%d_phi%d_n%d_nAng%d.pkl' % (paramTple + (SZA, phi, n, nAng))
savePath = savePath.replace(spaSetup, 'SPA')
print('-- Processing ' + os.path.basename(savePath) + ' --')
if 'lidar' in paramTple[0].lower(): # Use LIDAR YAML file
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
nowPix = returnPixel(paramTple[0], sza=SZA, relPhi=phi, nowPix=None, \
                     concase=paramTple[1], orbit=paramTple[2], lidErrDir=lidErrDir) # these last two (concase & orbit) are only needed if using a lidar w/ Kathy's noise model
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[3])
nowPix.land_prct = landPrct

if 'lidar' in paramTple[0].lower(): # implement ACCP SIT-A specific RI contraints 
    fldPath='imaginary_part_of_refractive_index_constant.2.value'
    fwdYAMLObj = graspYAML(baseYAMLpath=cstmFwdYAML)
    # COARSE IRI
    val1 = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.2.value'))
    val2 = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.4.value'))
    newVal = min(val1, val2)-0.001
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.2.min', [max(newVal, 1e-8)])
    newVal = max(val1, val2)+0.001
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.2.max', [newVal])
    # COARSE RRI
    val1 = np.mean(fwdYAMLObj.access('real_part_of_refractive_index_spectral_dependent.2.value'))
    val2 = np.mean(fwdYAMLObj.access('real_part_of_refractive_index_spectral_dependent.4.value'))
    newVal = min(val1, val2)-0.02
    bckYAMLObj.access('real_part_of_refractive_index_constant.2.min', [newVal])
    newVal = max(val1, val2)+0.02
    bckYAMLObj.access('real_part_of_refractive_index_constant.2.max', [newVal])
    # FINE IRI MAX (break with procedure starts here... )
    val1 = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.1.value'))
    val2 = np.mean(fwdYAMLObj.access('imaginary_part_of_refractive_index_spectral_dependent.3.value'))
    newVal = max(val1, val2)+0.001
    bckYAMLObj.access('imaginary_part_of_refractive_index_constant.1.max', [newVal])

print('n = %d, nAng = %d, Nλ = %d' % (n, nAng, nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                               binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, \
                               lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=False, \
                               workingFileSave=True, fixRndmSeed=True, verbose=verbose)

