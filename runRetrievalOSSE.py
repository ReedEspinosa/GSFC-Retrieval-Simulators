#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
import simulateRetrieval as rs
from readOSSEnetCDF import osseData
from miscFunctions import checkDiscover
from architectureMap import returnPixel
from MADCAP_functions import hashFileSHA1

archName = 'polar07'
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
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    saveStart = '/Users/wrespino/Desktop/OSSE_NR_20060101' # end will be appended
    maxCPU = 2
    fpDict = {
        'polarNc4FP':'/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/OSSE_NR_20060101_0100z_V1/gpm-polar07-g5nr.lc.vlidort.20060101_0100z_%dd00nm.nc4',
        'asmNc4FP':  '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/OSSE_NR_20060101_0100z_V1/gpm-g5nr.lb2.asm_Nx.20060101_0100z.nc4',
        'metNc4FP':  '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/OSSE_NR_20060101_0100z_V1/gpm-g5nr.lb2.met_Nv.20060101_0100z.nc4',
        'verbose':   True,
            }

YAMLpth = bckYAMLpathLID if 'lidar' in archName.lower() else bckYAMLpathPOL

# derive save file path then setup and run retrievals
yamlTag = '_YAML%s_' % hashFileSHA1(YAMLpth)[0:8]
savePath = saveStart + yamlTag + archName + '_V1.pkl'
print('-- Processing ' + os.path.basename(savePath) + ' --')
nowPix = returnPixel(archName)
fpDict['wvls'] = [mv['wl'] for mv in nowPix.measVals]
simA = rs.simulation(nowPix) # defines new instance corresponding to this architecture
od = osseData(fpDict)
fwdData = od.osse2graspRslts(4)
simA.runSim(fwdData, YAMLpth, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)

""" TODO:
    - we need to add code to osseData to read LIDAR measurments
    - anaylze simulation results currently can't handle more than one fwd observation set...
        I think it might even break with just one set now too due to changes in simulateRetrieval.py
"""