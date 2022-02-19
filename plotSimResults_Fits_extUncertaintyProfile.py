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
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, 'GSFC-Retrieval-Simulators','ACCP_ArchitectureAndCanonicalCases'))
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
from simulateRetrieval import simulation
from miscFunctions import matplotlibX11, norm2absExtProf
from ACCP_functions import findFineModes

matplotlibX11()
import matplotlib.pyplot as plt


instruments = ['Lidar090','Lidar050', 'Lidar090+polar07','Lidar050+polar07', 'polar07'] # polarimeter can not go first!
case = 'case08b1'
χthresh = 20.0
forceχ2Calc = True

simRsltFile = '/Users/wrespino/Synced/Working/SIM17_SITA_SeptAssessment/DRS_V01_%s_%s_tFct1.00_orb*_multiAngles_n*_nAngALL.pkl'
trgtλLidar = 0.532 # μm, note if this lands on a wavelengths without profiles no lidar data will be plotted

figP, axQ = plt.subplots(figsize=(6,6))
for k,inst in enumerate(instruments):
    posFiles = glob(simRsltFile % (inst, case))
    assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
    simA = simulation(picklePath=posFiles[0])
    NfwdModes = len(simA.rsltFwd[0]['aodMode'][:,0])
    simA.conerganceFilter(χthresh=χthresh, forceχ2Calc=forceχ2Calc, verbose=True)
    lIndL = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλLidar))
    fineModeInd, fineModeIndBck = findFineModes(simA)
    if 'lidar' in inst.lower(): simB = simA
    extRMSE = simA.analyzeSimProfile(wvlnthInd=lIndL, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, fwdSim=simB)[0]
    extTrue = np.zeros(len(extRMSE['βext']))
    for i in range(NfwdModes):
#         βprof = norm2absExtProf(simA.rsltFwd[0]['βext'][i,:], simA.rsltFwd[0]['range'][i,:], simA.rsltFwd[0]['aodMode'][i,lIndL])
        βprof = 0
        extTrue = extTrue + βprof   
    # if k==0: axQ.plot(1e6*extTrue, simA.rsltFwd[0]['range'][0,:]/1e3, color=0.5*np.ones(3), linewidth=2)
    barOffset = 50*(k-len(instruments)/2)
    vertLevs = (simB.rsltFwd[0]['range'][0,:]+barOffset)/1e3
    lnSty = 'None'
    axQ.errorbar(1e6*extTrue[:-1], vertLevs[:-1], xerr=1e6*extRMSE['βext'][:-1], linestyle=lnSty, elinewidth=3)
lgHnd = axQ.legend(tuple(instruments))
lgHnd.draggable()
axQ.set_xlabel('Retrieved Extinction RMSE ($Mm^{-1}Sr^{-1}$)')
axQ.set_ylabel('Height (km)')
axQ.set_ylim([-0.25,simB.rsltFwd[0]['range'][0,1]/1e3+0.25])
axQ.set_xlim([-150,150])
axQ.set_title('%s' % (case))





