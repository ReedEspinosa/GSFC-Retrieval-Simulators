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
import matplotlib.pyplot as plt


instruments = ['Lidar09','Lidar05','Lidar06', 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=189
case = 'SPA'
# surface = 'Desert'
# surface = 'Vegetation'
surface = 'Vegetation'


simRsltFile = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/DRS_V09_%s_%s%s_orbSS_tFct0.10_multiAngles_n*_nAngALL.pkl'
trgtλLidar = 0.532 # μm, note if this lands on a wavelengths without profiles no lidar data will be plotted

figP, axQ = plt.subplots(figsize=(6,6))
for k,inst in enumerate(instruments):
    posFiles = glob(simRsltFile % (inst, case, surface.replace('Ocean','')))
    assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
    simA = simulation(picklePath=posFiles[0])
    NfwdModes = len(simA.rsltFwd[0]['aodMode'][:,0])
    simA.conerganceFilter(χthresh=50.0, verbose=True)
    lIndL = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλLidar))
    extRMSE = simA.analyzeSimProfile(wvlnthInd=lIndL)[0]
    extTrue = np.zeros(len(extRMSE['βext']))
    for i in range(NfwdModes):
        # βprof = norm2absExtProf(simA.rsltFwd[0]['βext'][i,:], simA.rsltFwd[0]['range'][i,:], simA.rsltFwd[0]['aodMode'][i,lIndL])
        βprof = 0
        extTrue = extTrue + βprof   
    # if k==0: axQ.plot(1e6*extTrue, simA.rsltFwd[0]['range'][0,:]/1e3, color=0.5*np.ones(3), linewidth=2)
    barOffset = 50*(k-len(instruments)/2)
    vertLevs = (simA.rsltFwd[0]['range'][0,:]+barOffset)/1e3
    lnSty = 'None'
    axQ.errorbar(1e6*extTrue, vertLevs, xerr=1e6*extRMSE['βext'], linestyle=lnSty, elinewidth=3)
lgHnd = axQ.legend(tuple(instruments))
lgHnd.draggable()
axQ.set_xlabel('Retrieved Extinction RMSE ($Mm^{-1}Sr^{-1}$)')
axQ.set_ylabel('Height (km)')
axQ.set_ylim([-0.25,4.25])
axQ.set_xlim([-150,150])
axQ.set_title('%s - %s' % (case, surface))





