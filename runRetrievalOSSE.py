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
vrsn = 1 # version of this script/processing (appended to save filename)

tmStr = '%04d%02d%02d_%02d00z' % (year, month, day, hour)
dtTple = (year, month, day)
pathFrmt = os.path.join(osseDataPath, orbit.upper(), 'Level%s', 'Y%04d','M%02d', 'D%02d', '%s')
fpDict = {
    'polarNc4FP': pathFrmt % (('C',)+dtTple+(orbit+'-polar07-g5nr.lc.vlidort.'+tmStr+'_%dd00nm.nc4',)),
    'asmNc4FP': pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.asm_Nx.'+tmStr+'.nc4',)),
    'metNc4FP': pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.met_Nv.'+tmStr+'.nc4',)),
    'aerNc4FP': pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.aer_Nv.'+tmStr+'.nc4',)),
    'lcExt': pathFrmt % (('C',)+dtTple+(orbit+'-g5nr.lc.ext.'+tmStr+'.%dnm.nc4',)),
#    'lc2Lidar':* (string) - gpm-lidar-g5nr.lc2.YYYYMMDD_HH00z.%dnm path to file w/ simulated [noise added] lidar measurements
    'verbose':   True,
        }

# choose YAML flavor, derive save file path and setup/run retrievals
YAMLpth = bckYAMLpathLID if 'lidar' in archName.lower() else bckYAMLpathPOL
yamlTag = 'YAML%s' % hashFileSHA1(YAMLpth)[0:8]
savePath = pathFrmt % (('E',)+dtTple+('%s-g5nr.le%02d.GRASP.%s.%s.%s.pkl' % (orbit,vrsn,yamlTag,archName,tmStr),))
print('-- Generating ' + os.path.basename(savePath) + ' --')
sys.exit()
nowPix = returnPixel(archName)
fpDict['wvls'] = [mv['wl'] for mv in nowPix.measVals]
simA = rs.simulation(nowPix) # defines new instance corresponding to this architecture
od = osseData(fpDict)
fwdData = od.osse2graspRslts()
simA.runSim(fwdData, YAMLpth, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)

""" TODO:
    - we need to add code to osseData to read LIDAR measurments
"""