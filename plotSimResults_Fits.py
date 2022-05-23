#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This script will plot the Lidar profile and polarimeter I, Q, U fits 
It will produce unexpected behavoir if len(rsltFwd)>1 (always uses the zeroth index of rsltFwd)
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

# simRsltFile can have glob style wildcards
simRsltFile = '/Users/aputhukkudy/Working_Data/ACCDAM/2022/Campex_Simulations/Mar2022/Flight#1/Spherical/Linear/16bins/SimulationTest_16Bins_AOD_0p1_550nm.pkl'
# nn = int(sys.argv[1])
# mm = int(sys.argv[2])
# simRsltFile = '/Users/wrespino/Synced/Working/SIM_OSSE_Test/ss450-g5nr.leV30.GRASP.YAML*-n%dpixStrt%d.polar07*.random.20060801_0000z.pkl' % (nn,mm*28)
trgtλLidar = 0.532 # μm, note if this lands on a wavelengths without profiles no lidar data will be plotted
trgtλPolar = 0.87 # μm, if this lands on a wavelengths without I, Q or U no polarimeter data will be plotted
extErrPlot = True
χthresh = 5
minSaved = 40
fineModesBck = [0]

# --END INPUT SETTINGS--
posFiles = glob(simRsltFile)
assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
simA = simulation(picklePath=posFiles[0])
simA.conerganceFilter(χthresh=χthresh, minSaved=minSaved, verbose=True, forceχ2Calc=False)
# simA.rsltFwd = simA.rsltFwd[0:40:9]
# simA.rsltBck = simA.rsltBck[0:40:9]
lIndL = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλLidar))
lIndP = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλPolar))
alphVal = 1/np.sqrt(len(simA.rsltBck))
alphVal = 0.3
color1 = plt.get_cmap("rainbow_r")
color2 = plt.get_cmap("Dark2")


measTypesL = [x for x in ['VExt', 'VBS', 'LS'] if 'fit_'+x in simA.rsltFwd[0] and not np.isnan(simA.rsltFwd[0]['fit_'+x][:,lIndL]).any()]
LIDARpresent = False if len(measTypesL)==0 else True
if LIDARpresent:
    print('Lidar data found at %5.3f μm' % simA.rsltFwd[0]['lambda'][lIndL])
    assert not np.isnan(simA.rsltFwd[0]['fit_'+measTypesL[0]][0,lIndL]), 'Nans found in LIDAR data at this wavelength! Is the value of lIndL valid?'
    figL, axL = plt.subplots(1,len(measTypesL)+1, figsize=(7.5,5), gridspec_kw={'width_ratios': [1.7]+[1 for _ in measTypesL]})
# Polar Prep
if 'fit_QoI' in simA.rsltBck[0]:
    measTypesP = ['I', 'QoI', 'UoI']
    POLARpresent = True
elif 'fit_Q' in simA.rsltBck[0]:
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
    figP, axP = plt.subplots(1,len(measTypesP),figsize=(12,5))
    if not type(axP)==np.ndarray: axP=[axP]
# Plot LIDAR and Polar measurements and fits
NfwdModes = simA.rsltFwd[0]['aodMode'].shape[0] if 'aodMode' in simA.rsltFwd[0] else 0
NbckModes = simA.rsltBck[0]['aodMode'].shape[0]

lidarRangeLow=16 # %
lidarRangeHigh=84 # %
for rb in simA.rsltBck:
    if LIDARpresent:
        mdHnd = []; lgTxt = []
        for i in range(NbckModes): # Lidar extinction profile
            extFits = [norm2absExtProf(rb['βext'][i,:], rb['range'][i,:], rb['aodMode'][i,lIndL]) for rb in simA.rsltBck]
            extFitsLow = np.percentile(extFits, lidarRangeLow, axis=0)
            extFitsHigh = np.percentile(extFits, lidarRangeHigh, axis=0)
            axL[0].fill_betweenx(simA.rsltBck[0]['range'][i,:]/1e3, 1e6*extFitsLow, 1e6*extFitsHigh, color=color1(i), edgecolor='none', alpha=0.34)
    #     for j,rb in enumerate(simA.rsltBck[0:3]): # temp added for testing, REMOVE
        for i,mt in enumerate(measTypesL): # Lidar retrieval meas & fit
            measMeas = []
            measFits = []
            for rb in simA.rsltBck: # temp added for testing, REMOVE
    #             axL[i+1].plot(1e6*rb['meas_'+mt][:,lIndL], rb['RangeLidar'][:,lIndL]/1e3, color=color2[0], alpha=alphVal)
    #             axL[i+1].plot(, rb['RangeLidar'][:,lIndL]/1e3, color=color2[1], alpha=alphVal)
                measMeas.append(1e6*rb['meas_'+mt][:,lIndL])
                measFits.append(1e6*rb['fit_'+mt][:,lIndL])            
            axL[i+1].plot(np.mean(measMeas,axis=0), rb['RangeLidar'][:,lIndL]/1e3, color='k', linewidth=2, alpha=1, zorder=20) #truth (really measurement mean, may not be accurate for small N)
            extFitsLow = np.percentile(measMeas, lidarRangeLow, axis=0)
            extFitsHigh = np.percentile(measMeas, lidarRangeHigh, axis=0)
            axL[i+1].fill_betweenx(simA.rsltBck[0]['range'][i,:]/1e3, extFitsLow, extFitsHigh, color=color2(7), edgecolor='none', alpha=0.3, zorder=10) # measurement
            extFitsLow = np.percentile(measFits, lidarRangeLow, axis=0)
            extFitsHigh = np.percentile(measFits, lidarRangeHigh, axis=0)
            axL[i+1].fill_betweenx(simA.rsltBck[0]['range'][i,:]/1e3, extFitsLow, extFitsHigh, color=color2(7), edgecolor='none', alpha=0.6, zorder=15) # fits
    if POLARpresent:
        for i,mt in enumerate(measTypesP): # Polarimeter retrieval meas & fit
            axP[i].plot(θfun(lIndP,rb), rb['meas_'+mt][:,lIndP], color=color2(0), alpha=alphVal)
            axP[i].plot(θfun(lIndP,rb), rb['fit_'+mt][:,lIndP], color=color1(1), alpha=alphVal)
    if LIDARpresent:
        for i in range(NfwdModes):
            βprof = norm2absExtProf(simA.rsltFwd[0]['βext'][i,:], simA.rsltFwd[0]['range'][i,:], simA.rsltFwd[0]['aodMode'][i,lIndL])
            mdHnd.append(axL[0].plot(1e6*βprof, simA.rsltFwd[0]['range'][i,:]/1e3, '-', color=color1(i), linewidth=2))
            lgTxt.append('Mode %d' % (i+1))
    #         axL[0].plot([], [], 'o-', color=color1[i]/2) # ???
        for i,mt in enumerate(measTypesL): # Lidar fwd fit
            if len(simA.rsltFwd)==1: axL[i+1].plot(1e6*simA.rsltFwd[0]['fit_'+mt][:,lIndL], simA.rsltFwd[0]['RangeLidar'][:,lIndL]/1e3, 'ko-')
            leg = axL[i+1].legend(['Truth', 'Retrieved', 'Measured'])
            leg.set_draggable(True)
            axL[i+1].set_xlim([0,1.03*np.percentile([1e6*rb['meas_'+mt][:,lIndL] for rb in simA.rsltBck], lidarRangeHigh, axis=0).max()])
            axL[i+1].set_yticks([])

    if POLARpresent:
        for i,mt in enumerate(measTypesP): # Polarimeter fwd fit
            if 'fit_'+mt not in simA.rsltFwd[0] and 'oI' in mt: # fwd calculation performed with aboslute Q and U
                fwdData = simA.rsltFwd[0]['fit_'+mt[0]][:,lIndP]/simA.rsltFwd[0]['fit_I'][:,lIndP]
            else:
                fwdData = simA.rsltFwd[0]['fit_'+mt][:,lIndP]
            # axP[i].plot(θfun(lIndP,simA.rsltFwd[0]), fwdData, 'ko-')
            leg = axP[i].legend(['Truth', 'Retrieved', 'Measured']) # there are many lines but the first two should be these
            leg.set_draggable(True)
            axP[i].set_xlabel('viewing zenith (°)')
            axP[i].set_title(mt.replace('o','/'))
    fn = os.path.splitext(posFiles[0])[0].split('/')[-1]

    if LIDARpresent: # touch up LIDAR plots
        lgTxt = ['Smoke (Fine)', 'Smoke (Coarse)', 'Marine (Fine)', 'Marine (Coarse)']
        leg = axL[0].legend(list(map(list, zip(*mdHnd)))[0], lgTxt)
        leg.set_draggable(True)
        axL[0].set_ylabel('Altitude (km)')
        axL[0].set_xlabel('Mode resolved extinction ($Mm^{-1}$)')
        rngBins = simA.rsltBck[0]['RangeLidar'][:,lIndL]/1e3
        for ax in axL: ax.set_ylim([rngBins[-1], rngBins[0]])
        axL[0].set_xlim([0,75])
        if -np.diff(simA.rsltBck[0]['RangeLidar'][:,lIndL])[0] > -2.1*np.diff(simA.rsltBck[0]['RangeLidar'][:,lIndL])[-1]: # probably log-spaced range bins
            for ax in axL: ax.set_yscale('log')
        for i,mt in enumerate(measTypesL): # loop throuh measurement types to label x-axes
            if mt == 'VExt':
                lblStr = 'Total extinction ($Mm^{-1}$)'
            elif mt == 'VBS':
                lblStr = 'Backscatter ($Mm^{-1}Sr^{-1}$)'
            elif mt == 'LS':
                lblStr = 'Attenuated Backscatter ($Mm^{-1}Sr^{-1}$)'
            else:
                lblStr = mt
            axL[i+1].set_xlabel(lblStr)
    #     ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd[0]['lambda'][lIndL])
    #     figL.suptitle(ttlTxt)
    #     figL.tight_layout(rect=[0, 0.03, 1, 0.95])
        figL.tight_layout()
    if POLARpresent: # touch up Polarimeter plots
        axP[0].set_ylabel('Reflectance')
        ttlTxt = '%s [%5.3f μm]' % (fn, simA.rsltFwd[0]['lambda'][lIndP])
        figP.suptitle(ttlTxt)
        figP.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.ion()
plt.show()

print(simA.analyzeSim(lIndP, modeCut=0.5)[0])
cstV = np.mean([rb['costVal'] for rb in simA.rsltBck])
print('Total AOD: %f | Cost Value: %f' % (simA.rsltFwd[0]['aod'][lIndP], cstV))

# For X11 on Discover
# plt.ioff()
# plt.draw()
# plt.show(block=False)
# plt.show(block=False)
