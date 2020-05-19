#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using the A-CCP canonical cases and corresponding architectures defined in the ACCP_ArchitectureAndCanonicalCases directory within this repo """

import os
import sys
import itertools
import shutil 
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import simulateRetrieval as rs
from miscFunctions import checkDiscover
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML

n = int(sys.argv[1]) # (0,1,2,...,N-1)
# n=0

dryRun = False # set everything up but don't actually retrieve (probably used with fullSave=True)
fullSave = True # archive all the GRASP working directories into a zip file saved along side the pkl file 

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM15_pre613SeminarApr2020/CONCASE4MODEV05_n%d_' % n)
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
#    Nsims = 1
#    maxCPU = 1
    Nsims = 56
    maxCPU = 28
else: # MacBook Air
    saveStart = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/canonicalCases4Lille/CASE_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    Nsims = 4
    maxCPU = 2
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_4modes.yml')
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_3lambda_POL.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes.yml')


# conCases = []
# for caseLet in ['a','b','c','d','e','f']:
#     conCases.append('case06'+caseLet) # 6
conCases = ['case06d']
SZAs = [30] # 3 (GRASP doesn't seem to be wild about θs=0)
Phis = [0] # 1
τFactor = [1] #2
# instruments = ['polar07', 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07',
#                'polar0700', 'Lidar0900+polar0700','Lidar0500+polar0700','Lidar0600+polar0700'] # 8 N=42
instruments = ['Lidar0900+polar0700','Lidar0500+polar0700','Lidar0600+polar0700'] # 8 N=42
rndIntialGuess = False # randomly vary the intial guess of retrieved parameters

paramTple = list(itertools.product(*[instruments,conCases,SZAs,Phis,τFactor]))[n] 
savePath = saveStart + '%s_%s_sza%d_phi%d_tFct%4.2f_V1.pkl' % paramTple

print('-- Processing ' + os.path.basename(savePath) + ' --')

# Select Correct YAML file
if 'lidar' in paramTple[0].lower():
    fwdModelYAMLpath = fwdModelYAMLpathLID
    bckYAMLpath = bckYAMLpathLID
else:
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    bckYAMLpath = bckYAMLpathPOL
# RUN SIMULATION
nowPix = returnPixel(paramTple[0], sza=paramTple[2], relPhi=paramTple[3], nowPix=None)
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[4])
nowPix.land_prct = landPrct
print('n= %d, Nλ = %d' % (n,nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
gObjFwd, gObjBck = simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess, dryRun=dryRun)
# Pack all working directories into a ZIP
if fullSave: # TODO: build zip from original tmp folders without making extra copies to disk, see first answer here: https://stackoverflow.com/questions/458436/adding-folders-to-a-zip-file-using-python
    fullSaveDir = savePath[0:-4]
    if os.path.exists(fullSaveDir): shutil.rmtree(fullSaveDir)
    os.mkdir(fullSaveDir)
    shutil.copytree(gObjFwd.dirGRASP, os.path.join(fullSaveDir,'forwardCalculation'))
    for i, gb in enumerate(gObjBck):
        shutil.copytree(gb.dirGRASP, os.path.join(fullSaveDir,'inversion%02d' % i))
    shutil.make_archive(fullSaveDir, 'zip', fullSaveDir)
    shutil.rmtree(fullSaveDir) 

