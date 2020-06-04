#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:06:44 2020

@author: wrespino
"""
import numpy as np
import os
import sys
from glob import glob
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
from simulateRetrieval import simulation
from miscFunctions import matplotlibX11, norm2absExtProf
matplotlibX11()


instruments = ['Lidar09','Lidar05','Lidar06', 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=189
case = 'case06a'
surface = 'Ocean'
# surface = 'Vegetation'
# surface = 'Desert'


simRsltFile = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/DRS_V02_%s_%s%s_orbSS_tFct1.00_multiAngles_n*_nAngALL.pkl'
trgtλLidar = 0.532 # μm, note if this lands on a wavelengths without profiles no lidar data will be plotted
χthresh = 15

figP, axQ = plt.subplots(figsize=(6,6))
for k,inst in enumerate(instruments):
    posFiles = glob(simRsltFile % (inst, case, surface.replace('Ocean','')))
    assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
    simA = simulation(picklePath=posFiles[0])
    simA.conerganceFilter(χthresh=15.0, verbose=True)
    lIndL = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλLidar))    
    extRMSE = simA.analyzeSimProfile(wvlnthInd=lIndL)[0]
    if ['polar07'] == barVal:
        rmse, bias = simB.analyzeSim(lInd, fineModesFwd=fineIndFwd, fineModesBck=fineIndBck)
        rmse['aodMode_PBLFT'] = np.nan
        rmse['rEffMode_PBLFT'] = np.nan
        bias['aodMode_PBLFT'] = [[np.nan]]
        bias['rEffMode_PBLFT'] = [[np.nan]]
    else:
        rmse, bias = simB.analyzeSim(lInd, fineModesFwd=fineIndFwd, fineModesBck=fineIndBck, hghtCut=2100)
        rmse['aodMode_PBLFT'] = rmse['aodMode_PBLFT'][0]
        rmse['rEffMode_PBLFT'] = rmse['rEffMode_PBLFT'][0]
    harvest[n, :], harvestQ, rmseVal = af.normalizeError(simB.rsltFwd[0], rmse, lInd, totVars, bias)


