#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM14_lidarPolACCP/SIM43V2_2mode_')
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    Nsims = 56
    maxCPU = 28
else: # MacBook Air
    saveStart = '/Users/wrespino/Desktop/testLIDAR_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    Nsims = 2
    maxCPU = 1
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
# bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_4modes.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_old3Lambda_2mode.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_3lambda_POL.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_IQU_3lambda_POL.yml')


#instruments = ['lidar09+img02', 'lidar05+img02', 'img02', 'img01'] # 4
##instruments = ['lidar09+img02visnir', 'lidar05+img02visnir', 'img02visnir', 'img01visnir'] # 4
conCases = ['variable'] #4
SZAs = [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # 13 (GRASP doesn't seem to be wild about θs=0)
Phis = [0] # 1 
τFactor = [0.04, 0.08, 0.12, 0.18, 0.35] #5 N=240

instruments = ['lidar05+polar07', 'lidar09+polar07'] #2
#conCases = ['case02a','case02b','case02c','case03','case07']
# conCases = []
# for caseLet in ['a','b','c','d','e','f']:
#     conCases.append('case06'+caseLet)
#    conCases.append('case06'+caseLet+'monomode') #17 total
#    if caseLet in ['e','f']:
#        conCases.append('case06'+caseLet+'nonsph')
#        conCases.append('case06'+caseLet+'monomode'+'nonsph') #21 total
# SZAs = [0.1, 30, 60] # 3 (GRASP doesn't seem to be wild about θs=0)
# SZAs = [30] # 3 (GRASP doesn't seem to be wild about θs=0)
# Phis = [0] # 1 
# τFactor = [1.0] #3 N=189 Nodes

rndIntialGuess = False # randomly vary the intial guess of retrieved parameters

paramTple = list(itertools.product(*[instruments,conCases,SZAs,Phis,τFactor]))[n] 
savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple
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
simA = rs.simulation(nowPix) # defines new instance for this architecture
# runs the simulation for given set of conditions, releaseYAML=True -> auto adjust back yaml Nλ to match insturment
simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)
