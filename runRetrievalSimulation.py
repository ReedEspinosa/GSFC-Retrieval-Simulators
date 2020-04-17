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

n = int(sys.argv[1]) # (0,1,2,...,N-1)
# missingCases=[660, 804, 829, 969, 976, 979, 1076, 1131, 1152, 1168, 1419, 1467, 1571]
# n=missingCases[n]
if n>=180: sys.exit()

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM15_pre613SeminarApr2020/COMBO04_n%d_' % n)
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    Nsims = 40
    maxCPU = 4
else: # MacBook Air
    saveStart = '/Users/wrespino/Desktop/testLIDAR_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    Nsims = 2
    maxCPU = 1
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_3lambda_POL.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_IQU_3lambda_POL.yml')


conCases = ['variableFineLofted+variableCoarse',
            'variableFine+variableCoarseLofted',
            'variableFineLofted+variableCoarseLofted+variableFine+variableCoarse'
            ] # 3
SZAs = [0.1, 10, 22, 40, 60] # 5 (GRASP doesn't seem to be wild about θs=0)
Phis = [0] # 1 
#τFactor = [0.04, 0.08, 0.10, 0.12, 0.14, 0.18, 0.35] #7 
τFactor = [0.12, 0.14, 0.35] #3
# τFactor = [0.01  , 0.02  , 0.025 , 0.03  , 0.035 , 0.045 , 0.0875] #7 (cut by a quarter because we are using four modes)
# instruments = ['misr', 'modisMisr', 'modisMisrPolar','Lidar05+modisMisrPolar','Lidar09+modisMisrPolar','modis','Lidar05','Lidar09'] #8 N=1575
instruments = ['Lidar05+modisMisrPolar','Lidar09+modisMisrPolar','Lidar05','Lidar09'] #4 N=180
rndIntialGuess = True # randomly vary the intial guess of retrieved parameters

paramTple = list(itertools.product(*[instruments,conCases,SZAs,Phis,τFactor]))[n] 
savePath = saveStart + '%s_case-%sAOD_sza%d_phi%d_tFct%4.2f_V1.pkl' % paramTple

print('-- Processing ' + os.path.basename(savePath) + ' --')

# RUN SIMULATION
if 'lidar' in paramTple[0].lower():
    fwdModelYAMLpath = fwdModelYAMLpathLID
    bckYAMLpath = bckYAMLpathLID
else:
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    bckYAMLpath = bckYAMLpathPOL
    
nowPix = returnPixel(paramTple[0], sza=paramTple[2], relPhi=paramTple[3], nowPix=None)
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[4])
nowPix.land_prct = landPrct
print('n= %d, Nλ = %d' % (n,nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)
