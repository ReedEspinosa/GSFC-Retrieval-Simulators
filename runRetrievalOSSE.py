#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
import simulateRetrieval as rs
from miscFunctions import checkDiscover
from architectureMap import returnPixel

archName = 'polar07+lidar09'
rndIntialGuess = True # randomize initial guess before retrieving
if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR.yml')
    bckYAMLpathPOL = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL.yml')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    saveStart = os.path.join(basePath, 'synced/Working/OSSE_NR_20060101')
    maxCPU = 28
else: # MacBook Air
    bckYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_LIDAR.yml'
    bckYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_3lambda_POL.yml'
    dirGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open/build/bin/grasp'
    krnlPath = None
    saveStart = '/Users/wrespino/Desktop/OSSE_NR_20060101' # end will be appended
    maxCPU = 1

YAMLpth = bckYAMLpathLID if 'lidar' in archName.lower() else bckYAMLpathPOL
savePath = saveStart + '_case-%s_V1.pkl' % archName
print('-- Processing ' + os.path.basename(savePath) + ' --')

nowPix = returnPixel(archName)

fwdData = list() 
# TODO: a function/class to load variables from netCDF into above fwdData (list of dicts)
#    standard format but see code starting at simulateRetrieval.py:44 for fields needed
#    this would ultimatly produce a rslts dictionary so we might want to make it a method in graspRun?

simA = rs.simulation(nowPix) # defines new instance for this architecture
# runs the simulation for given set of conditions, releaseYAML=True -> auto adjust back yaml Nλ to match insturment
simA.runSim(fwdData, YAMLpth, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)
# TODO: anaylze simulation results currently can't handle more than one fwd observation set...
    # I think it might even break with just one set now too due to changes in simulateRetrieval.py
