#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import simulateRetrieval as rs
from miscFunctions import checkDiscover
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML


n = int(sys.argv[1]) # (0,1,2,...,N-1)
#n=0

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    fwdModelYAMLpathPOL = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_POL.yml')
    fwdModelYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_LIDAR.yml')
    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR.yml')
    bckYAMLpathPOL = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL.yml')
    bckYAMLpathPOLnonSph = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL_nonsph.yml')
    #if n<2:
    saveStart = os.path.join(basePath, 'synced/Working/SIM13_lidarTest/SIM42_')
    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR.yml')
    #elif n<4:
    #    saveStart = os.path.join(basePath, 'synced/Working/SIM13_lidarTest/SIM26_')
    #    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR2.yml')
    #else:
    #    saveStart = os.path.join(basePath, 'synced/Working/SIM13_lidarTest/SIM27_')
    #    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR3.yml')
    #bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR_oneRI.yml')
    #bckYAMLpathPOL = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL_oneRI.yml')
    Nsims = 56
    maxCPU = 28
else: # MacBook Air
    fwdModelYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_LIDAR.yml'
    bckYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR.yml'
    #bckYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR_oneRI.yml'
    fwdModelYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_POL.yml'
    bckYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL.yml'
    #bckYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL_oneRI.yml'
    saveStart = '/Users/wrespino/Desktop/testLIDAR_' # end will be appended
    dirGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
    #dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    Nsims = 1
    maxCPU = 1



#instruments = ['lidar09+img02', 'lidar05+img02', 'img02', 'img01'] # 4
##instruments = ['lidar09+img02visnir', 'lidar05+img02visnir', 'img02visnir', 'img01visnir'] # 4
##conCases = ['variablenonsph', 'variablefine', 'variablefinenonsph'] #4
#SZAs = [0.1, 30, 60] # 3 (GRASP doesn't seem to be wild about θs=0)
#Phis = [0] # 1 
#τFactor = [0.04, 0.08, 0.12, 0.18, 0.35] #5 N=240

#instruments = ['lidar05+polar07', 'lidar09+polar07'] #2
instruments = ['lidar05']
#conCases = ['case02a','case02b','case02c','case03','case07']
conCases = []
for caseLet in ['a','b','c','d','e','f']:
#    conCases.append('case06'+caseLet)
    conCases.append('case06'+caseLet+'monomode') #17 total
#    if caseLet in ['e','f']:
#        conCases.append('case06'+caseLet+'nonsph')
#        conCases.append('case06'+caseLet+'monomode'+'nonsph') #21 total
#conCases = ['case06amonomode']
SZAs = [0.1, 30, 60] # 3 (GRASP doesn't seem to be wild about θs=0)
# SZAs = [30]
Phis = [0] # 1 
τFactor = [1.0] #3 N=189 Nodes

rndIntialGuess = True # randomly vary the intial guess of retrieved parameters


sizeMat = [1,1,1,1, len(instruments), len(conCases), len(SZAs), len(Phis), len(τFactor)]
ind = [n//np.prod(sizeMat[i:i+4])%sizeMat[i+4] for i in range(5)]
paramTple = (instruments[ind[0]], conCases[ind[1]], SZAs[ind[2]], Phis[ind[3]], τFactor[ind[4]])
savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple
print('-- Processing ' + os.path.basename(savePath) + ' --')

# RUN SIMULATION
if 'lidar' in instruments[ind[0]].lower():
    fwdModelYAMLpath = fwdModelYAMLpathLID
    if ('case06cmonomode' in conCases[ind[1]].lower()) or ('case06dmonomode' in conCases[ind[1]].lower()):
        bckYAMLpath = bckYAMLpathLID[:-4]+'fine.yml'
    elif ('case06emonomode' in conCases[ind[1]].lower()) or ('case06fmonomode' in conCases[ind[1]].lower()):
        bckYAMLpath = bckYAMLpathLID[:-4]+'coarse.yml'
    else:
        bckYAMLpath = bckYAMLpathLID
else:
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    if 'nonsph' in conCases[ind[1]].lower():
        bckYAMLpath = bckYAMLpathPOLnonSph
    if ('case06cmonomode' in conCases[ind[1]].lower()) or ('case06dmonomode' in conCases[ind[1]].lower()):
        bckYAMLpath = bckYAMLpathPOL[:-4]+'fine.yml'
    elif ('case06emonomode' in conCases[ind[1]].lower()) or ('case06fmonomode' in conCases[ind[1]].lower()):
        bckYAMLpath = bckYAMLpathPOL[:-4]+'coarse.yml'
    else:
        bckYAMLpath = bckYAMLpathPOL
nowPix = returnPixel(paramTple[0], sza=paramTple[2], landPrct=100, relPhi=paramTple[3], nowPix=None)
cstmFwdYAML, landPrct = setupConCaseYAML(conCases[ind[1]], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[4])
nowPix.land_prct = landPrct
print('n= %d, Nλ = %d' % (n,nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for this architecture
# runs the simulation for given set of conditions, releaseYAML=True -> auto adjust back yaml Nλ to match insturment
simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)
