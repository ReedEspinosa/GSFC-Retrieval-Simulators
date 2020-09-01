#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

import os
import sys
import re
import numpy as np
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
import simulateRetrieval as rs
import functools
import itertools
from readOSSEnetCDF import osseData
from miscFunctions import checkDiscover
from architectureMap import returnPixel, addError
from MADCAP_functions import hashFileSHA1
from runGRASP import graspYAML

if checkDiscover(): # DISCOVER
    inInt = int(sys.argv[1])
    n = inInt%4
    m = int(inInt/4)
    basePath = os.environ['NOBACKUP']
    bckYAMLpathLID = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes.yml')
    bckYAMLpathPOL = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes.yml')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    osseDataPath = '/discover/nobackup/projects/gmao/osse2/pub/c1440_NR/OBS/A-CCP/'
    maxCPU = 28
else: # MacBook Air
    n=0
    bckYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes.yml'
    bckYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes.yml'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    maxCPU = 3
    osseDataPath = '/Users/wrespino/Synced/MADCAP_CAPER/testCase_Aug01_0000Z_VersionJune2020/'
rndIntialGuess = False # randomize initial guess before retrieving

year = 2006
month = 8
day = 1 + int(m/24)
hour = m%24
orbit = 'gpm' # gpm OR ss450
maxSZA = 50
oceanOnly = True
archNames = ['polar07', 'polar07+lidar09'] # name of instrument (never 100x, e.g. don't use 'polar0700' or 'lidar0900' – that is set w/ noiseFree below)
hghtBins = np.r_[15000:-1:-500] # centers of lidar bins (meters)
vrsn = 10 # general version tag to distinguish runs
wvls = [0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65] # (μm) if we only want specific λ set it here, otherwise use all netCDF files found
noiseFrees = [True, False] # do not add noise to the observations
# firstPix2Process = 0+m*maxCPU
firstPix2Process = 5*m # should provide some SZA variation
lastPix2Process = firstPix2Process+maxCPU

customOutDir = os.path.join(basePath, 'synced', 'Working', 'SIM_OSSE_Test') # save output here instead of within osseDataPath (None to disable)
#customOutDir = '/Users/wrespino/Desktop/'
verbose=True

archName, noiseFree = list(itertools.product(*[archNames,noiseFrees]))[n]
# choose YAML flavor, derive save file path and setup/run retrievals
YAMLpth = bckYAMLpathLID if 'lidar' in archName.lower() else bckYAMLpathPOL
yamlTag = 'YAML%s-n%dm%d' % (hashFileSHA1(YAMLpth)[0:8], n, m)
lidMtch = re.match('[A-z0-9]+\+lidar0([0-9])', archName.lower())
lidVer = int(lidMtch[1])*100**noiseFree if lidMtch else None
od = osseData(osseDataPath, orbit, year, month, day, hour, random=False, wvls=wvls, 
              lidarVersion=lidVer, maxSZA=maxSZA, oceanOnly=oceanOnly, loadDust=False, verbose=verbose)
saveArchNm = archName+'NONOISE' if noiseFree else archName
savePath = od.fpDict['savePath'] % (vrsn, yamlTag, saveArchNm)
if customOutDir: savePath = os.path.join(customOutDir, os.path.basename(savePath))
print('-- Generating ' + os.path.basename(savePath) + ' --')
fwdData = od.osse2graspRslts(NpixMin=firstPix2Process, NpixMax=lastPix2Process, newLayers=hghtBins)
radNoiseFun = None if noiseFree else functools.partial(addError, 'polar07')

simA = rs.simulation() # defines new instance corresponding to this architecture

# val_n = 10**(-n)
# val_m = m
yamlObj = graspYAML(YAMLpth, newTmpFile=('BCK_n%dm%d' % (n, m)))
# for ch in [3,4]:
#     for md in [1,2]:
#         fldPath = 'retrieval.constraints.characteristic[%d].mode[%d].single_pixel.smoothness_constraints.lagrange_multiplier' % (ch,md)
#         yamlObj.access(fldPath, newVal=val_n)
#         fldPath = 'retrieval.constraints.characteristic[%d].mode[%d].single_pixel.smoothness_constraints.difference_order' % (ch,md)
#         yamlObj.access(fldPath, newVal=val_m)
simA.runSim(fwdData, yamlObj, maxCPU=maxCPU, maxT=20, savePath=savePath, 
            binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, 
            rndIntialGuess=rndIntialGuess, radianceNoiseFun=radNoiseFun, verbose=verbose)








