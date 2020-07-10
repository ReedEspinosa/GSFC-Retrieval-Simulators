#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

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

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes.yml')
    bckYAMLpathPOL = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes.yml')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    osseDataPath = '/discover/nobackup/projects/gmao/osse2/pub/c1440_NR/OBS/A-CCP/'
    maxCPU = 28
else: # MacBook Air
    bckYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes.yml'
    bckYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes.yml'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    maxCPU = 2
    osseDataPath = '/Users/wrespino/Synced/MADCAP_CAPER/testCase_Aug01_0000Z_VersionJune2020/'
rndIntialGuess = True # randomize initial guess before retrieving

year = 2006
month = 8
day = 1
hour = 0
orbit = 'gpm' # gpm OR ss450
archName = 'polar07' # name of instrument to pull from returnPixel()
vrsn = 1 # general version tag to distinguish runs

# choose YAML flavor, derive save file path and setup/run retrievals
YAMLpth = bckYAMLpathLID if 'lidar' in archName.lower() else bckYAMLpathPOL
yamlTag = 'YAML%s' % hashFileSHA1(YAMLpth)[0:8]
nowPix = returnPixel(archName)
wvls = [mv['wl'] for mv in nowPix.measVals]
simA = rs.simulation(nowPix) # defines new instance corresponding to this architecture
od = osseData(osseDataPath, orbit, year, month, day, hour, random=False, wvls=wvls, verbose=True)
savePath = od.fpDict['savePath'] % (vrsn, yamlTag, archName)
print('-- Generating ' + os.path.basename(savePath) + ' --')
fwdData = od.osse2graspRslts(NpixMax=6, newLayers=None) # TODO: this should be the lidar bin heights to use in back SDATA file
simA.runSim(fwdData, YAMLpth, maxCPU=maxCPU, maxT=20, savePath=savePath, binPathGRASP=dirGRASP, 
            intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, 
            rndIntialGuess=rndIntialGuess, verbose=True)

""" TODO:
    - we need to add code to osseData to read LIDAR measurments
"""