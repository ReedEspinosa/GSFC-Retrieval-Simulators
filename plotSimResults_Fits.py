#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will plot the Lidar profile and polarimeter I,Q, U fits """

import numpy as np
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
from simulateRetrieval import simulation
from miscFunctions import matplotlibX11
matplotlibX11()
import matplotlib.pyplot as plt

n=0
simRsltFile = '/Users/wrespino/Synced/Working/SIM15_pre613SeminarApr2020/TEST01_2mode_n%d_Lidar05+modisMisrPolar_case-variableFineLofted+variableCoarseLofted+variableFine+variableCoarse_sza30_phi0_tFct0.10_V0_YAML%d.pkl' % (n,n)
lIndP = 3 # polarimeter λ to plot
lIndL = 2 # LIDAR λ to plot (3,7)

simA = simulation(picklePath=simRsltFile)
if not type(simA.rsltFwd) is dict: simA.rsltFwd = simA.rsltFwd[0] # HACK [VERY BAD] -- remove when we fix this to work with lists 

alphVal = 1/np.sqrt(len(simA.rsltBck))
color1 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 1, 0]])
# LIDAR Prep
measTypesL = [x for x in ['VExt', 'VBS', 'LS'] if 'fit_'+x in simA.rsltFwd]
LIDARpresent = False if len(measTypesL)==0 else True
if LIDARpresent:
    rngVar = 'RangeLidar'
    profExtNm = 'βext'
    βfun = lambda i,l,d: d['aodMode'][i,l]*d[profExtNm][i,:]/np.mean(d[profExtNm][i,:])
    assert not np.isnan(simA.rsltFwd['fit_'+measTypesL[0]][0,lIndL]), 'Nans found in LIDAR data at this wavelength! Is the value of lIndL valid?'
    figL, axL = plt.subplots(1,len(measTypesL)+1,figsize=(12,6))
# Polar Prep
if 'fit_QoI' in simA.rsltBck[0]:
    measTypesP = ['I', 'QoI', 'UoI']
    POLARpresent = True
elif 'fit_QoI' in simA.rsltBck[0]: 
    measTypesP = ['I', 'Q', 'U']
    POLARpresent = True
else:
    POLARpresent = False
if POLARpresent:
    [x for x in measTypesP if 'fit_'+x in simA.rsltFwd]
    θfun = lambda l,d: [θ if φ<180 else -θ for θ,φ in zip(d['vis'][:,l], d['fis'][:,l])]
    assert not np.isnan(simA.rsltBck[0]['fit_'+measTypesP[0]][0,lIndP]), 'Nans found in Polarimeter data at this wavelength! Is the value of lIndP valid?'
figP, axP = plt.subplots(1,len(measTypesP),figsize=(12,6))
# Plot LIDAR and Polar measurements and fits
NfwdModes = simA.rsltFwd['aodMode'].shape[0]
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
        mdHnd.append(axL[0].plot(βfun(i,lIndL,simA.rsltFwd), simA.rsltFwd['range'][i,:]/1e3, 'o-', color=color1[i]/2))
        lgTxt.append('Mode %d' % i)
    for i,mt in enumerate(measTypesL): # Lidar fwd fit
        axL[i+1].plot(1e6*simA.rsltFwd['fit_'+mt][:,lIndL], simA.rsltFwd[rngVar][:,lIndL]/1e3, 'ko-')
        axL[i+1].legend(['Measured', 'Retrieved']) # there are many lines but the first two should be these
    axL[i+1].set_xlim([0,2*1e6*simA.rsltFwd['fit_'+mt][:,lIndL].max()])
if POLARpresent:
    for i,mt in enumerate(measTypesP): # Polarimeter fwd fit
        if 'fit_'+mt not in simA.rsltFwd and 'oI' in mt: # fwd calculation performed with aboslute Q and U
            fwdData = simA.rsltFwd['fit_'+mt[0]][:,lIndP]/simA.rsltFwd['fit_I'][:,lIndP]
        else:
            fwdData = simA.rsltFwd['fit_'+mt][:,lIndP]
        axP[i].plot(θfun(lIndP,simA.rsltFwd), fwdData, 'ko-')
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
    ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd['lambda'][lIndL])
    figL.suptitle(ttlTxt)
    figL.tight_layout(rect=[0, 0.03, 1, 0.95])
if POLARpresent: # touch up Polarimeter plots
    axP[0].set_ylabel('Reflectance')
    ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd['lambda'][lIndP])
    figP.suptitle(ttlTxt)
    figP.tight_layout(rect=[0, 0.03, 1, 0.95])

# For X11 on Discover
#plt.ioff()
#plt.draw()
#plt.show(block=False)
#plt.show(block=False)
