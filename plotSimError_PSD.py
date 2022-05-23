#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation # This should ensure GSFC-GRASP-Python-Interface is in the path

# *****
# TODO: Does the assumption of all modes having the same radii work with Anin's output?!? #
# *****

# pklDataPath = '/Users/wrespino/Synced/Working/OSSE_Test_Run/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl' # None to skip reloading of data
pklDataPath = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults_MERGED/DRS_V01_polar07_caseAll_tFct1.00_orbSS_multiAngles_nAll_nAngALL.pkl' # None to skip reloading of data
# pklDataPath = '/Users/wrespino/Synced/ACCDAM_RemoteSensingObservability/AninsSimulations/Apr2022_5mode/MERGED_CAMP2ExmegaALL_2modes_AOD_ALL_550nmALL.pkl'
# pklDataPath = '/Users/wrespino/Synced/ACCDAM_RemoteSensingObservability/AninsSimulations/Apr2022_5mode/MERGED_CAMP2ExmegaALL_2modes_AOD_ALL_550nmALL.pkl'
pklDataPath = None # None to skip reloading of data
# plotSaveDir = '/Users/wrespino/Synced/AOS/PLRA/Figures_AODF_bugFixApr11'
plotSaveDir = '/Users/wrespino/Synced/Presentations/UMBC_Earthday_2022/figs'
surf2plot = 'ocean' # land, ocean or both

colors = np.array([[0.83921569, 0.15294118, 0.15686275, 1.        ],  # red  - number
                   [0.12156863, 0.46666667, 0.70588235, 1.        ],  # blue - surface
                   [0.58039216, 0.40392157, 0.74117647, 1.        ],  # purp - volume
                   [0.00803922, 0.00803922, 0.00803922, 0.65      ]]) # grey - relative

inst = 'polar07'
orb = 'Alt800'
simType = 'CanCase'
# simType = 'G5NR' 
version = 'PM25_UMBC_EDS2022-%s-%s-%s-VXX' % (inst, orb, simType) # for PDF file names

if pklDataPath is not None:
    simBase = simulation(picklePath=pklDataPath)
    print('Loaded from %s - %d' % (pklDataPath, len(simBase.rsltBck)))
print('--')

# NOTE: This check only looks at first pixel; if radii grid differs with pixel this will not catch the error
assert np.all(simBase.rsltFwd[0]['r'][0]==simBase.rsltFwd[0]['r']), 'PSD of all Fwd modes must be specfified at the same radii!'
assert np.all(simBase.rsltBck[0]['r'][0]==simBase.rsltBck[0]['r']), 'PSD of all Bck modes must be specfified at the same radii!'

if 'land_prct' not in simBase.rsltFwd[0] or surf2plot=='both':
    keepInd = range(len(simBase.rsltFwd))
    if not surf2plot=='both': print('NO land_prct KEY! Including all surface types...')
else:
    lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
    keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1

# apply convergence filter
# simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 95)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))


calc_dV = lambda rf: rf['dVdlnr'].sum(axis=0)/rf['r'][0]
calc_dV2dA = lambda dv, r: 3*dv/r
calc_dA2dN = lambda da, r: da/(r**2)/4/np.pi

rAll = simBase.rsltFwd[0]['r'][0] # HACK: Again, we assume all pixels, fwd/bck and all modes are defined at the same radii
Nr = len(rAll); Npix = len(simBase.rsltFwd)
dXdrT = np.empty((3, Npix, Nr)) # dXdrT[j,:,:] j=0,1,2 -> number, surface, volume, respectively
dXdrR = np.empty((3, Npix, Nr)) # dXdrR[j,:,:] j=0,1,2 -> number, surface, volume, respectively
for i, (rf,rb) in enumerate(zip(simBase.rsltFwd, simBase.rsltBck)):
    # if modes have different radii we should put them on common grid here
    dXdrT[2,i,:] = calc_dV(rf)
    dXdrR[2,i,:] = calc_dV(rb)
    dXdrT[1,i,:] = calc_dV2dA(dXdrT[2,i,:], rf['r'][0])
    dXdrR[1,i,:] = calc_dV2dA(dXdrR[2,i,:], rb['r'][0]) 
    dXdrT[0,i,:] = calc_dA2dN(dXdrT[1,i,:], rf['r'][0])
    dXdrR[0,i,:] = calc_dA2dN(dXdrR[1,i,:], rb['r'][0])

figM, axM = plt.subplots(1,1, figsize=(8,5)) # MARE figure 
figM.subplots_adjust(right=0.65)

def prettyAxis(hnd, color, label):
    tkw = dict(size=4, width=1.5)
    hnd.set_ylabel(label)
    hnd.yaxis.label.set_color(color)
    with warnings.catch_warnings(): # bug (kinda) in numpy; see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
        warnings.simplefilter(action='ignore', category=FutureWarning)
        hnd.tick_params(axis='y', colors=color, **tkw)
MARE = np.mean(np.abs(dXdrR[0,:,:] - dXdrT[0,:,:])/dXdrT[0,:,:], axis=0)*100
pltHnd, = axM.plot(rAll, MARE, color=colors[-1])
prettyAxis(axM, pltHnd.get_color(), 'Mean Absolute Relative Error (%)')
twinAxs = []
labels = ['Number RMSE ($\# \cdot μm^{-2} \cdot μm^{-1}$)',
          'Surface  RMSE ($μm^{2} \cdot μm^{-2} \cdot μm^{-1}$)',
          'Volume  RMSE ($μm^{3} \cdot μm^{-2} \cdot μm^{-1}$)']
for i in range(3):
    twinAxs.append(axM.twinx())
    if i>0: twinAxs[-1].spines.right.set_position(("axes", 0.96+0.22*i))
    RMSE = np.sqrt(np.mean((dXdrR[i,:,:] - dXdrT[i,:,:])**2, axis=0))
    pltHnd, = twinAxs[-1].plot(rAll, RMSE, ':', color=colors[i], linewidth=2)
    prettyAxis(twinAxs[-1], pltHnd.get_color(), labels[i])
axM.set_xlabel('radius (μm)')
axM.set_xscale('log')

# fn = 'CDFPlots_AODgt%05.3f_%s.pdf' % (aodMin, version)
# figC.savefig(os.path.join(plotSaveDir, fn))
# print('CDF Plot saved as %s' % fn)

plt.ion()
plt.show()
