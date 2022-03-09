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
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
from simulateRetrieval import simulation
from miscFunctions import matplotlibX11, norm2absExtProf
matplotlibX11()
import matplotlib.pyplot as plt
# import seaborn as sns

# simRsltFile can have glob style wildcards
simRsltFile = '/Users/aputhukkudy/Working_Data/ACCDAM/2022/Campex_Simulations/Mar2022/Flight#1/Spherical/Log/SimulationTestFlight#1_Level#1_AOD_2p0_550nm.pkl'
# nn = int(sys.argv[1])
# mm = int(sys.argv[2])
# simRsltFile = '/Users/wrespino/Synced/Working/SIM_OSSE_Test/ss450-g5nr.leV30.GRASP.YAML*-n%dpixStrt%d.polar07*.random.20060801_0000z.pkl' % (nn,mm*28)
trgtλLidar = 0.532 # μm, note if this lands on a wavelengths without profiles no lidar data will be plotted
trgtλPolar = 0.55 # μm, if this lands on a wavelengths without I, Q or U no polarimeter data will be plotted
extErrPlot = True
χthresh = 2.5
minSaved = 40
fineModesBck = [0]

lidarRangeLow=16 # %
lidarRangeHigh=84 # %
# lidarRangeLow=5 # %
# lidarRangeHigh=95 # %


SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12.5


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# --END INPUT SETTINGS--
posFiles = glob(simRsltFile)
assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
simA = simulation(picklePath=posFiles[0])
simA.conerganceFilter(χthresh=χthresh, minSaved=minSaved, verbose=True, forceχ2Calc=True)
# simA.rsltFwd = simA.rsltFwd[0:40:9]
# simA.rsltBck = simA.rsltBck[0:40:9]
lIndL = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλLidar))
lIndP = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλPolar))
alphVal = 0.55
color1 = plt.cm.RdBu([0.25,0.1,0.75,0.9])/1.15 # extinction retrieved
color2 = plt.get_cmap("Dark2") # lidar measurement fits
# color3 = plt.get_cmap("Dark2") # polarimeter measurement fits
# color3 = plt.cm.Dark2([0,0.125])/1.4 # polarimeter measurement fits
color3 = plt.cm.Set1([0,0.125])/1.7 # polarimeter measurement fits


measTypesL = [x for x in ['VExt', 'VBS', 'LS'] if 'fit_'+x in simA.rsltFwd[0] and not np.isnan(simA.rsltFwd[0]['fit_'+x][:,lIndL]).any()]
print('Lidar data found at %5.3f μm' % simA.rsltFwd[0]['lambda'][lIndL])
# assert not np.isnan(simA.rsltFwd[0]['fit_'+measTypesL[0]][0,lIndL]), 'Nans found in LIDAR data at this wavelength! Is the value of lIndL valid?'

# Polar Prep
if 'fit_QoI' in simA.rsltBck[0]:
    measTypesP = ['I', 'QoI', 'UoI']
elif 'fit_Q' in simA.rsltBck[0]:
    measTypesP = ['I', 'Q', 'U']
elif 'fit_I' in simA.rsltBck[0]: 
    measTypesP = ['I']

NmeasTypes = len(measTypesL) + 2
fig, axs = plt.subplots(2, NmeasTypes, figsize=(11.5,5), gridspec_kw={'width_ratios': [1.7]+[1.05 for _ in range(NmeasTypes-2)]+[1.3]})
gs = axs[0, 0].get_gridspec()
for axRow in axs: 
    for ax in axRow[0:-1]: ax.remove()
axL = []
for i in range(NmeasTypes-1):
    axL.append(fig.add_subplot(gs[0:, i]))
axP = np.r_[axs[0, -1], axs[1, -1]]

print('Polarimeter data found at %5.3f μm' % simA.rsltFwd[0]['lambda'][lIndP])
[x for x in measTypesP if 'fit_'+x in simA.rsltFwd[0]]
θfun = lambda l,d: [θ if φ<180 else -θ for θ,φ in zip(d['vis'][:,l], d['fis'][:,l])]
assert not np.isnan(simA.rsltBck[0]['fit_'+measTypesP[0]][0,lIndP]), 'Nans found in Polarimeter data at this wavelength! Is the value of lIndP valid?'
if not type(axP)==np.ndarray: axP=[axP]
# Plot LIDAR and Polar measurements and fits
NfwdModes = simA.rsltFwd[0]['aodMode'].shape[0] if 'aodMode' in simA.rsltFwd[0] else 0
NbckModes = simA.rsltBck[0]['aodMode'].shape[0]

mdHnd = []; lgTxt = []
for i in range(NbckModes): # Lidar extinction profile
    extFits = [norm2absExtProf(rb['βext'][i,:], rb['range'][i,:], rb['aodMode'][i,lIndL]) for rb in simA.rsltBck]
    extFitsLow = np.percentile(extFits, lidarRangeLow, axis=0)
    extFitsHigh = np.percentile(extFits, lidarRangeHigh, axis=0)
    axL[0].fill_betweenx(simA.rsltBck[0]['range'][i,:]/1e3, 1e6*extFitsLow, 1e6*extFitsHigh, color=color1[i], edgecolor='none', alpha=0.34)
#     for j,rb in enumerate(simA.rsltBck[0:3]): # temp added for testing, REMOVE
for i,mt in enumerate(measTypesL): # Lidar retrieval meas & fit
    measMeas = []
    measFits = []
    for rb in simA.rsltBck: # temp added for testing, REMOVE
#             axL[i+1].plot(1e6*rb['meas_'+mt][:,lIndL], rb['RangeLidar'][:,lIndL]/1e3, color=color2[0], alpha=alphVal)
#             axL[i+1].plot(, rb['RangeLidar'][:,lIndL]/1e3, color=color2[1], alpha=alphVal)
        measMeas.append(1e6*rb['meas_'+mt][:,lIndL])
        measFits.append(1e6*rb['fit_'+mt][:,lIndL])            
    axL[i+1].plot(np.mean(measMeas,axis=0), rb['RangeLidar'][:,lIndL]/1e3, color='k', linewidth=1, alpha=1, zorder=20) #truth (really measurement mean, may not be accurate for small N)
    extFitsLow = np.percentile(measMeas, lidarRangeLow, axis=0)
    extFitsHigh = np.percentile(measMeas, lidarRangeHigh, axis=0)
    axL[i+1].fill_betweenx(simA.rsltBck[0]['range'][i,:]/1e3, extFitsLow, extFitsHigh, color=color2(7), edgecolor='none', alpha=0.3, zorder=10) # measurement
    extFitsLow = np.percentile(measFits, lidarRangeLow, axis=0)
    extFitsHigh = np.percentile(measFits, lidarRangeHigh, axis=0)
    axL[i+1].fill_betweenx(simA.rsltBck[0]['range'][i,:]/1e3, extFitsLow, extFitsHigh, color=color2(7), edgecolor='none', alpha=0.6, zorder=15) # fits

if len(measTypesP)==3:
    for i,rb in enumerate(simA.rsltBck):
        simA.rsltBck[i]['meas_DoLP'] = np.sqrt(np.sum([rb['meas_'+mt]**2 for mt in measTypesP[1:3]], axis=0))
        simA.rsltBck[i]['fit_DoLP'] = np.sqrt(np.sum([rb['fit_'+mt]**2 for mt in measTypesP[1:3]], axis=0))
        if 'oI' not in measTypesP[1]: # above is using absolute Q and U, we need to divide by I [sqrt(Q^2+U^2)/I]
            simA.rsltBck[i]['meas_DoLP'] = simA.rsltBck[i]['meas_DoLP']/rb['meas_'+measTypesP[0]]
            simA.rsltBck[i]['fit_DoLP'] = simA.rsltBck[i]['fit_DoLP']/rb['fit_'+measTypesP[0]]
    measTypesP = ['I', 'DoLP']
    
for i,mt in enumerate(measTypesP): # Polarimeter retrieval meas & fit
    for rb in simA.rsltBck[[36,40,48]]:
        axP[i].plot(θfun(lIndP,rb), rb['meas_'+mt][:,lIndP], color=color3[0], alpha=alphVal)
        axP[i].plot(θfun(lIndP,rb), rb['fit_'+mt][:,lIndP], color=color3[1], alpha=alphVal)
        if i==0: print('i=%3d, Phi=%5.3f, SZA=%5.3f' % (i,rb['fis'][0,0], rb['sza'][0,0]))
        

for i in range(NfwdModes):
    βprof = norm2absExtProf(simA.rsltFwd[0]['βext'][i,:], simA.rsltFwd[0]['range'][i,:], simA.rsltFwd[0]['aodMode'][i,lIndL])
    mdHnd.append(axL[0].plot(1e6*βprof, simA.rsltFwd[0]['range'][i,:]/1e3, '-', color=color1[i], linewidth=2))
    lgTxt.append('Mode %d' % (i+1))
#         axL[0].plot([], [], 'o-', color=color1[i]/2) # ???
for i,mt in enumerate(measTypesL): # Lidar fwd fit
    if len(simA.rsltFwd)==1: axL[i+1].plot(1e6*simA.rsltFwd[0]['fit_'+mt][:,lIndL], simA.rsltFwd[0]['RangeLidar'][:,lIndL]/1e3, 'ko-')
    if i==1: 
        leg = axL[i+1].legend(['Truth', 'Measurement', 'GRASP Fit'])
        leg.set_draggable(True)
    upBnd = np.percentile([1e6*rb['meas_'+mt][:,lIndL] for rb in simA.rsltBck], lidarRangeHigh, axis=0).max()
    tickStep = 30 if upBnd>10 else 1
    upBnd = np.ceil(upBnd/tickStep)*tickStep
    if i==1: upBnd=3.25 # HACK!
    axL[i+1].set_xlim([0, upBnd])
    axL[i+1].set_xticks(np.r_[0:(1.0001*upBnd):tickStep])    
    axL[i+1].set_yticks([])

lgTxt = ['Smoke (F)', 'Smoke (C)', 'Marine (F)', 'Marine (C)']
leg = axL[0].legend(list(map(list, zip(*mdHnd)))[0], lgTxt)
leg.set_draggable(True)
axL[0].set_ylabel('Altitude (km)')
axL[0].set_xlabel('Mode Resolved Extinction ($Mm^{-1}$)')
rngBins = simA.rsltBck[0]['RangeLidar'][:,lIndL]/1e3
for ax in axL: ax.set_ylim([rngBins[-1], rngBins[0]])
axL[0].set_xlim([0,75])
axL[0].set_xticks([0,15,30,45,60,75])
if -np.diff(simA.rsltBck[0]['RangeLidar'][:,lIndL])[0] > -2.1*np.diff(simA.rsltBck[0]['RangeLidar'][:,lIndL])[-1]: # probably log-spaced range bins
    for ax in axL: ax.set_yscale('log')
for i,mt in enumerate(measTypesL): # loop throuh measurement types to label x-axes
    if mt == 'VExt':
        lblStr = 'Total Extinction ($Mm^{-1}$)'
    elif mt == 'VBS':
        lblStr = 'Backscatter ($Mm^{-1}Sr^{-1}$)'
    elif mt == 'LS':
        lblStr = 'Attenuated Backscatter ($Mm^{-1}Sr^{-1}$)'
    else:
        lblStr = mt
    axL[i+1].set_xlabel(lblStr)


axP[0].set_ylabel('Intensity')
# axP[0].set_yticks([0.05,0.1,0.15,1.2,1.5])
axP[0].set_yticks([0.05,0.1,0.15])
axP[1].set_yticks([0.0,0.2,0.4,0.6])
axP[0].set_xticks([])
axP[1].set_xticks([-60,-30,0,30,60])
axP[1].set_ylabel('DoLP')
leg = axP[0].legend(['Measurement', 'GRASP Fit']) # there are many lines but the first two should be these
leg.set_draggable(True)
axP[1].set_xlabel('Viewing Zenith (°)')
fn = os.path.splitext(posFiles[0])[0].split('/')[-1]

fig.tight_layout()

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
