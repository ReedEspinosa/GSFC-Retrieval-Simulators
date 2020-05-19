#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This script will plot the Lidar profile and polarimeter I, Q, U fits 
It will produce unexpected behavoir if len(rsltFwd)>1 (always uses the zeroth index of rsltFwd)
"""
import numpy as np
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
from simulateRetrieval import simulation
from miscFunctions import matplotlibX11
matplotlibX11()
import matplotlib.pyplot as plt

simRsltFile = '/Users/wrespino/Synced/Working/SIM15_pre613SeminarApr2020/CONCASE4MODEV06_n0_Lidar0600+polar0700_case06d_sza30_phi0_tFct1.00_V1.pkl'
trgtλLidar = 0.532 # μm, note if this lands on a wavelengths without profiles no lidar data will be plotted
trgtλPolar = 0.550 # μm, if this lands on a wavelengths without I, Q or U no polarimeter data will be plotted

# --END INPUT SETTINGS--

simA = simulation(picklePath=simRsltFile)
simA.conerganceFilter(χthresh=19.0, verbose=True)
lIndL = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλLidar))
lIndP = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλPolar))
alphVal = 1/np.sqrt(len(simA.rsltBck))
color1 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 1, 0]])
# LIDAR Prep
measTypesL = [x for x in ['VExt', 'VBS', 'LS'] if 'fit_'+x in simA.rsltFwd[0] and not np.isnan(simA.rsltFwd[0]['fit_'+x][:,lIndL]).any()]
LIDARpresent = False if len(measTypesL)==0 else True
if LIDARpresent:
    print('Lidar data found at %5.3f μm' % simA.rsltFwd[0]['lambda'][lIndL])
    rngVar = 'RangeLidar'
    profExtNm = 'βext'
    βfun = lambda i,l,d: d['aodMode'][i,l]*d[profExtNm][i,:]/np.mean(d[profExtNm][i,:])
    assert not np.isnan(simA.rsltFwd[0]['fit_'+measTypesL[0]][0,lIndL]), 'Nans found in LIDAR data at this wavelength! Is the value of lIndL valid?'
    figL, axL = plt.subplots(1,len(measTypesL)+1,figsize=(12,6))
# Polar Prep
if 'fit_QoI' in simA.rsltBck[0]:
    measTypesP = ['I', 'QoI', 'UoI']
    POLARpresent = True
elif 'fit_QoI' in simA.rsltBck[0]: 
    measTypesP = ['I', 'Q', 'U']
    POLARpresent = True
elif 'fit_I' in simA.rsltBck[0]: 
    measTypesP = ['I']
    POLARpresent = True
else:
    POLARpresent = False
if POLARpresent:
    print('Polarimeter data found at %5.3f μm' % simA.rsltFwd[0]['lambda'][lIndP])
    [x for x in measTypesP if 'fit_'+x in simA.rsltFwd[0]]
    θfun = lambda l,d: [θ if φ<180 else -θ for θ,φ in zip(d['vis'][:,l], d['fis'][:,l])]
    assert not np.isnan(simA.rsltBck[0]['fit_'+measTypesP[0]][0,lIndP]), 'Nans found in Polarimeter data at this wavelength! Is the value of lIndP valid?'
    figP, axP = plt.subplots(1,len(measTypesP),figsize=(12,6))
    if not type(axP)==np.ndarray: axP=[axP]
# Plot LIDAR and Polar measurements and fits
NfwdModes = simA.rsltFwd[0]['aodMode'].shape[0]
NbckModes = simA.rsltBck[0]['aodMode'].shape[0]
frstPass = True
for rb in simA.rsltBck:
    if LIDARpresent:
        for i in range(NbckModes): # Lidar extinction profile
            axL[0].plot(βfun(i,lIndL,rb), rb['range'][i,:]/1e3, color=color1[i], alpha=alphVal)
        frstPass = False
        for i,mt in enumerate(measTypesL): # Lidar retrieval meas & fit
            axL[i+1].plot(1e6*rb['meas_'+mt][:,lIndL], rb[rngVar][:,lIndL]/1e3, color=color1[0], alpha=alphVal)
            axL[i+1].plot(1e6*rb['fit_'+mt][:,lIndL], rb[rngVar][:,lIndL]/1e3, color=color1[1], alpha=alphVal)
    if POLARpresent:
        for i,mt in enumerate(measTypesP): # Polarimeter retrieval meas & fit
            axP[i].plot(θfun(lIndP,rb), rb['meas_'+mt][:,lIndP], color=color1[0], alpha=alphVal)
            axP[i].plot(θfun(lIndP,rb), rb['fit_'+mt][:,lIndP], color=color1[1], alpha=alphVal)
if LIDARpresent:
    mdHnd = []
    lgTxt = []
    for i in range(NfwdModes):
        mdHnd.append(axL[0].plot(βfun(i,lIndL,simA.rsltFwd[0]), simA.rsltFwd[0]['range'][i,:]/1e3, 'o-', color=color1[i]/2))
        lgTxt.append('Mode %d' % i)
    for i,mt in enumerate(measTypesL): # Lidar fwd fit
        axL[i+1].plot(1e6*simA.rsltFwd[0]['fit_'+mt][:,lIndL], simA.rsltFwd[0][rngVar][:,lIndL]/1e3, 'ko-')
        axL[i+1].legend(['Measured', 'Retrieved']) # there are many lines but the first two should be these
    axL[i+1].set_xlim([0,2*1e6*simA.rsltFwd[0]['fit_'+mt][:,lIndL].max()])
if POLARpresent:
    for i,mt in enumerate(measTypesP): # Polarimeter fwd fit
        if 'fit_'+mt not in simA.rsltFwd[0] and 'oI' in mt: # fwd calculation performed with aboslute Q and U
            fwdData = simA.rsltFwd[0]['fit_'+mt[0]][:,lIndP]/simA.rsltFwd[0]['fit_I'][:,lIndP]
        else:
            fwdData = simA.rsltFwd[0]['fit_'+mt][:,lIndP]
        axP[i].plot(θfun(lIndP,simA.rsltFwd[0]), fwdData, 'ko-')
        axP[i].legend(['Measured', 'Retrieved']) # there are many lines but the first two should be these
        axP[i].set_xlabel('viewing zenith (°)')
        axP[i].set_title(mt.replace('o','/'))
fn = os.path.splitext(simRsltFile)[0].split('/')[-1]
if LIDARpresent: # touch up LIDAR plots
    axL[0].legend(list(map(list, zip(*mdHnd)))[0], lgTxt)
    axL[0].set_ylabel('Altitude (km)')
    axL[0].set_xlabel('Modal Extinction (A.U.)')
    axL[0].set_xlim([0,2.0])
    for i,mt in enumerate(measTypesL): # loop throuh measurement types to label x-axes
        if mt == 'VExt':
            lblStr = 'Extinction ($Mm^{-1}$)'
        elif mt == 'VBS':
            lblStr = 'Backscatter ($Mm^{-1}Sr^{-1}$)'
        elif mt == 'LS':
            lblStr = 'Attenuated Backscatter ($Mm^{-1}Sr^{-1}$)'
        else:
            lblStr = mt
        axL[i+1].set_xlabel(lblStr)
    ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd[0]['lambda'][lIndL])
    figL.suptitle(ttlTxt)
    figL.tight_layout(rect=[0, 0.03, 1, 0.95])
if POLARpresent: # touch up Polarimeter plots
    axP[0].set_ylabel('Reflectance')
    ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd[0]['lambda'][lIndP])
    figP.suptitle(ttlTxt)
    figP.tight_layout(rect=[0, 0.03, 1, 0.95])

# For X11 on Discover
#plt.ioff()
#plt.draw()
#plt.show(block=False)
#plt.show(block=False)
