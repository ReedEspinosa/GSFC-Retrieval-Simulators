#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using OSSE results and the osseData class """

import os
import sys
import re
import numpy as np
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
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
    nn = inInt
    basePath = os.environ['NOBACKUP']
    bckYAMLpathLID = os.path.join(basePath, 'GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes.yml')
    bckYAMLpathPOL = os.path.join(basePath, 'GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes.yml')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    osseDataPath = '/discover/nobackup/projects/gmao/osse2/pub/c1440_NR/OBS/A-CCP/'
    maxCPU = 2
else: # MacBook Air
    basePath = '/Users/wrespino/'
    inInt = int(sys.argv[1])
    nn = inInt
    bckYAMLpathLID = '/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLARandLIDAR_10Vbins_2modes.yml'
    bckYAMLpathPOL = '/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_BCK_POLAR_2modes.yml'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    maxCPU = 1
    osseDataPath = '/Users/wrespino/Synced/MADCAP_CAPER/testCase_Aug01_0000Z_VersionJune2020/'
rndIntialGuess = False # randomize initial guess before retrieving

year = 2006
month = 8
random = True # if true then day and hour below can be ignored
day = 1
hour = 0
orbit = 'ss450' # gpm OR ss450
maxSZA = 950
oceanOnly = False
archNames = ['polar07','polar07+lidar09','polar07+lidar05','polar07+lidar06'] # name of instrument (never 100x, e.g. don't use 'polar0700' or 'lidar0900' – that is set w/ noiseFree below)
hghtBins = np.round(np.cumsum(np.diff(np.logspace(-2,np.log2(2e4),25, base=2))+120)) # 25 bins, starting at 120m with exponentially increasing seperation up to 20km
vrsn = 100 # general version tag to distinguish runs
wvls = [0.355, 0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65] # (μm) if we only want specific λ set it here, otherwise use all netCDF files found
noiseFrees = [True, False] # do not add noise to the observations
pixInd = [7375, 1444, 1359, 929, 4654, 6574, 2786, 6461, 6897, 2010] # SS Aug 2006

customOutDir = os.path.join(basePath, 'synced', 'Working', 'OSSE_4_Lille') # save output here instead of within osseDataPath (None to disable)
# customOutDir = '/Users/wrespino/Desktop/lilleDump'
verbose=True

archName, noiseFree = list(itertools.product(*[archNames,noiseFrees]))[nn]
# choose YAML flavor, derive save file path and setup/run retrievals
YAMLpth = bckYAMLpathLID if 'lidar' in archName.lower() else bckYAMLpathPOL
yamlTag = 'YAML%s-n%dpixStrt%d' % (hashFileSHA1(YAMLpth)[0:8], nn, len(pixInd))
lidMtch = re.match('[A-z0-9]+\+lidar0([0-9])', archName.lower())
lidVer = int(lidMtch[1])*100**noiseFree if lidMtch else None
od = osseData(osseDataPath, orbit, year, month, day, hour, random=random, wvls=wvls, 
              lidarVersion=lidVer, maxSZA=maxSZA, oceanOnly=oceanOnly, loadDust=False, verbose=verbose)
saveArchNm = archName+'NONOISE' if noiseFree else archName
savePath = od.fpDict['savePath'] % (vrsn, yamlTag, saveArchNm)
if customOutDir: savePath = os.path.join(customOutDir, os.path.basename(savePath))
print('-- Generating ' + os.path.basename(savePath) + ' --')
fwdData = od.osse2graspRslts(pixInd=pixInd, newLayers=hghtBins)
radNoiseFun = None if noiseFree else functools.partial(addError, 'polar07')

simA = rs.simulation() # defines new instance corresponding to this architecture

yamlObj = graspYAML(YAMLpth, newTmpFile=('BCK_n%d' % nn))
# PLAY WITH SMOOTHNESS
# val_n = 0.1*3**(1-nn)
# val_m = mm+1
# for ch in [1]:
#     for md in [1,2]:
#         fldPath = 'retrieval.constraints.characteristic[%d].mode[%d].single_pixel.smoothness_constraints.lagrange_multiplier' % (ch,md)
#         yamlObj.access(fldPath, newVal=val_n)
#         fldPath = 'retrieval.constraints.characteristic[%d].mode[%d].single_pixel.smoothness_constraints.difference_order' % (ch,md)
#         yamlObj.access(fldPath, newVal=val_m)
# PLAY WITH VOLUME/PSD
# val_n = [0.05 + nn*0.02, 0.3]
# val_m = [0.2  + mm*0.05, 0.75]
# fldPath = 'retrieval.constraints.characteristic[3].mode[1].initial_guess.min'
# yamlObj.access(fldPath, newVal=val_n)
# fldPath = 'retrieval.constraints.characteristic[3].mode[1].initial_guess.max'
# yamlObj.access(fldPath, newVal=val_m)
# PLAY WITH NOISES
# val_m = 0.05*(mm+1)
# fldPath = 'retrieval.noises.noise[3].standard_deviation'
# yamlObj.access(fldPath, newVal=val_m)

simA.runSim(fwdData, yamlObj, maxCPU=maxCPU, maxT=20, savePath=savePath, 
            binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, 
            rndIntialGuess=rndIntialGuess, radianceNoiseFun=radNoiseFun,
            workingFileSave=True, dryRun=False, verbose=verbose)





