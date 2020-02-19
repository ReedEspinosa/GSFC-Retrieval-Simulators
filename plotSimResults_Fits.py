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

simRsltFile = '/Users/wrespino/synced/Working/SIM14_lidarPolACCP/SIM43V30_2mode_lidar05+polar07_case-case06c_sza30_phi0_tFct1.00_V2.pkl'
#simRsltFile = '/Users/wrespino/Synced/Working/SIM13_lidarTest/SIM43_lidar05+polar07_case-case06cmonomode_sza30_phi0_tFct1.00_V2.pkl'
lIndL = 3 # LIDAR λ to plot (3,7)
lIndP = 4 # polarimeter λ to plot

simA = simulation(picklePath=simRsltFile)
if not type(simA.rsltFwd) is dict: simA.rsltFwd = simA.rsltFwd[0] # HACK [VERY BAD] -- remove when we fix this to work with lists 

alphVal = 1/np.sqrt(len(simA.rsltBck))
color1 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 1, 0]])
# LIDAR Prep
rngVar = 'RangeLidar'
profExtNm = 'βext'
βfun = lambda i,l,d: d['aodMode'][i,l]*d[profExtNm][i,:]/np.mean(d[profExtNm][i,:])
measTypesL = [x for x in ['VExt', 'VBS', 'LS'] if 'fit_'+x in simA.rsltFwd]
assert not np.isnan(simA.rsltFwd['fit_'+measTypesL[0]][0,lIndL]), 'Nans found in LIDAR data at this wavelength! Is the value of lIndL valid?'
figL, axL = plt.subplots(1,len(measTypesL)+1,figsize=(12,6))
# Polar Prep
θfun = lambda l,d: [θ if φ<180 else -θ for θ,φ in zip(d['vis'][:,l], d['fis'][:,l])]
nrmPol = 'fit_QoI' in simA.rsltBck[0]
measTypesP = ['I', 'QoI', 'UoI'] if nrmPol else ['I', 'Q', 'U']
assert not np.isnan(simA.rsltBck[0]['fit_'+measTypesP[0]][0,lIndP]), 'Nans found in Polarimeter data at this wavelength! Is the value of lIndP valid?'
figP, axP = plt.subplots(1,len(measTypesP),figsize=(12,6))
# Plot LIDAR and Polar measurements and fits
NfwdModes = simA.rsltFwd['aodMode'].shape[0]
NbckModes = simA.rsltBck[0]['aodMode'].shape[0]
frstPass = True
for rb in simA.rsltBck:
#    [np.sum(rb[profExtNm][i,:]*rb['range'][i,:]) for i in range(rb['range'].shape[0])]
#    botPrf = 0 if np.sum(rb[profExtNm][0,:]*rb['range'][0,:])>np.sum(rb[profExtNm][1,:]*rb['range'][1,:]) else 1
#    axA[0].plot(βfun(botPrf%2,lInd,rb), rb['range'][botPrf%2,:]/1e3, color=color1, alpha=alphVal)
#    axA[0].plot(βfun((botPrf+1)%2,lInd,rb), rb['range'][(botPrf+1)%2,:]/1e3, color=color2, alpha=alphVal)
    for i in range(NbckModes):
        axL[0].plot(βfun(i,lIndL,rb), rb['range'][i,:]/1e3, color=color1[i], alpha=alphVal)
#        if frstPass: print(βfun(i,lIndL,rb)[rb['range'][i,:]>2200])
    frstPass = False
    for i,mt in enumerate(measTypesL): # Lidar retrieval meas & fit
        axL[i+1].plot(1e6*rb['meas_'+mt][:,lIndL], rb[rngVar][:,lIndL]/1e3, color=color1[0], alpha=alphVal)
        axL[i+1].plot(1e6*rb['fit_'+mt][:,lIndL], rb[rngVar][:,lIndL]/1e3, color=color1[1], alpha=alphVal)
    for i,mt in enumerate(measTypesP): # Polarimeter retrieval meas & fit
        axP[i].plot(θfun(lIndP,rb), rb['meas_'+mt][:,lIndP], color=color1[0], alpha=alphVal)
        axP[i].plot(θfun(lIndP,rb), rb['fit_'+mt][:,lIndP], color=color1[1], alpha=alphVal)
mdHnd = []
lgTxt = []
for i in range(NfwdModes):
    mdHnd.append(axL[0].plot(βfun(i,lIndL,simA.rsltFwd), simA.rsltFwd['range'][i,:]/1e3, 'o-', color=color1[i]/2))
#    mdHnd.append(axL[0].plot(βfun(i,lIndL,simA.rsltFwd), simA.rsltFwd['range'][i,:]/1e3, 'o-', color=color1[1::-1][i]/2))
    lgTxt.append('Mode %d' % i)
for i,mt in enumerate(measTypesL): # Lidar fwd fit
    axL[i+1].plot(1e6*simA.rsltFwd['fit_'+mt][:,lIndL], simA.rsltFwd[rngVar][:,lIndL]/1e3, 'ko-')
    axL[i+1].legend(['Measured', 'Retrieved']) # there are many lines but the first two should be these
    axL[i+1].set_xlim([0,2*1e6*simA.rsltFwd['fit_'+mt][:,lIndL].max()])
for i,mt in enumerate(measTypesP): # Polarimeter fwd fit
    if 'fit_'+mt not in simA.rsltFwd and 'oI' in mt: # fwd calculation performed with aboslute Q and U
        fwdData = simA.rsltFwd['fit_'+mt[0]][:,lIndP]/simA.rsltFwd['fit_I'][:,lIndP]
    else:
        fwdData = simA.rsltFwd['fit_'+mt][:,lIndP]
    axP[i].plot(θfun(lIndP,simA.rsltFwd), fwdData, 'ko-')
    axP[i].legend(['Measured', 'Retrieved']) # there are many lines but the first two should be these
    axP[i].set_xlabel('viewing zenith (°)')
    axP[i].set_title(mt.replace('o','/'))
# touch up LIDAR plots
axL[0].legend(list(map(list, zip(*mdHnd)))[0], lgTxt)
axL[0].set_ylabel('Altitude (km)')
axL[0].set_xlabel('Modal Extinction (A.U.)')
axL[0].set_xlim([0,0.7])
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
fn = os.path.splitext(simRsltFile)[0].split('/')[-1]
ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd['lambda'][lIndL])
figL.suptitle(ttlTxt)
figL.tight_layout(rect=[0, 0.03, 1, 0.95])
# touch up Polarimeter plots
axP[0].set_ylabel('Reflectance')
ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd['lambda'][lIndP])
figP.suptitle(ttlTxt)
figP.tight_layout(rect=[0, 0.03, 1, 0.95])

# For X11 on Discover
#plt.ioff()
#plt.draw()
#plt.show(block=False)
#plt.show(block=False)
