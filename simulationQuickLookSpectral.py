#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script is a modified verison of the simulationQuickLook.py, with 
the abiltity to plot 2D density and histograms of errors for retrieved parameters.
Specifically designed to accomodate the simulation retreieval study based on
CAMP2Ex measurments

Created on Wed Jul 13 14:18:11 2022

@author: aputhukkudy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Import the librarires
# =============================================================================
import os
from pprint import pprint
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'});mpl.rcParams.update({'ytick.right': 'True'});mpl.rcParams.update({'xtick.top': 'True'});plt.rcParams["font.family"] = "Latin Modern Math"; plt.rcParams["mathtext.fontset"] = "cm"; plt.rcParams["figure.dpi"]=330
from scipy import interpolate
from scipy.stats import norm, gaussian_kde, ncx2, moyal
from simulateRetrieval import simulation
from glob import glob

# =============================================================================
# Local definitions
# =============================================================================
def violinPlotEdits(parts, facecolor=False):
    # change the color of the violin plot
    if facecolor:
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_alpha(0.5)
        
    # color of mean
    for pc in ['cmeans']:
        vp = parts[pc]
        vp.set_edgecolor('k')
        vp.set_linewidth(1)
def axisLabelsLimits(axs, bandStr, limits, 
                     xlabel=r'Spectral bands ($\mu$m)', ylabel='Error in AOD'):
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_ylim(limits)
    axs.set_xticklabels(bandStr)
    axs.yaxis.grid(True)
# =============================================================================
# Initiation
# =============================================================================
# Wavelength index for plotting
waveInd = 2
# Wavelength index for AE calculation
waveInd2 = 4

# Define the string pattern for the file name
fnPtrnList = []
#fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_*z.pkl'
# fnPtrn = 'megaharp01_CAMP2Ex_2modes_AOD_*_550nm_addCoarse__campex_flight#*_layer#00.pkl'
fnPtrn = 'Camp2ex_AOD_*_550nm_*_campex_tria_flight#*_layer#00.pkl'
# fnPtrn = 'Camp2ex_AOD_*_550nm_SZA_30*_PHI_*_campex_flight#*_layer#00.pkl'
# fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_1000z.pkl'

# Location/dir where the pkl files are
inDirPath = '/home/aputhukkudy/ACCDAM/2022/Campex_Simulations/Dec2022/16/fullGeometry/withCoarseMode/darkOcean/2modes/megaharp01/'

# more tags and specifiations for the scatter plot
surf2plot = 'both' # land, ocean or both
aodMin = 0.2 # does not apply to first AOD plot
aodMax = 2
nMode = 0 # Select which layer or mode to plot
fnTag = 'AllCases'
xlabel = 'Simulated Truth'
MS = 2
FS = 10
LW121 = 1
pointAlpha = 0.20
clrText = '#FF6347' #[0,0.7,0.7]
nBins = 200 # no. of bins for histogram
nBins2 = 200 # no. of bins for 2D density plot
lightSave = True # Omit PM elements and extinction profiles from MERGED files to save space

# Define the path of the new merged pkl file
saveFN = 'MERGED_'+fnPtrn.replace('*','ALL')
savePATH = os.path.join(inDirPath,saveFN)

# =============================================================================
# Load data
# =============================================================================

# Define the path of the new merged pkl file
loadPATH = os.path.join(inDirPath,fnPtrn)
simBase = simulation(picklePath=loadPATH)


# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][waveInd])
pprint(simBase.analyzeSim(waveInd)[0])

# =============================================================================
# Filter data
# =============================================================================

# lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
# TODO: this has to be generalized so that it can be ran without crashing the code
# FIXME: at the moment, this hard coded to be ocean pixels
lp = np.array([0 for rf in simBase.rsltFwd])
keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1

# apply convergence filter
simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 95)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])
keepIndAll = keepInd

# variable to color point by in all subplots
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]])
clrVarAll = clrVar
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

# =============================================================================
# %% Plotting
# =============================================================================

# define the fig and axes handles
# Figure for spectral error plots
fig, ax = plt.subplots(3,1, figsize=(6,4))

# wavelengths
bands = simBase.rsltFwd[0]['lambda']
bandStr = ['']+[str(i) for i in bands]

# loop through bands to calculate the error and bias in individual variables
aodLst = []
for i in range(len(bands)):
    # =============================================================================
    # AOD
    # =============================================================================
    true = np.asarray([rf['aod'][i] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['aod'][i] for rf in simBase.rsltBck])[keepInd]
    
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    aodLst.append((rtrv-true).T)
    
    parts = ax[0].violinplot((rtrv-true).T, [i+1], points= 300,
                             showmeans=True, showmedians=True,
                             showextrema=False, quantiles=[0.1, 0.9],
                             bw_method='silverman')
    violinPlotEdits(parts)
    ax[0].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax[0].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
# add title and axis labels

axisLabelsLimits(ax[0], bandStr, limits=[-0.15, 0.15] )
# violinPlotEdits(parts)
    
aodCMLst = []
for i in range(len(bands)):
    # =============================================================================
    # AOD
    # =============================================================================
    true = np.asarray([rf['aodMode'][:,i] for rf in simBase.rsltFwd])[keepInd][:,4]
    rtrv = np.asarray([rf['aodMode'][:,i] for rf in simBase.rsltBck])[keepInd][:,1]
    
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    aodCMLst.append((rtrv-true).T)
    parts = ax[1].violinplot((rtrv-true).T, [i+1], points= 300,
                             showmeans=True, showmedians=True,
                             showextrema=False,
                             bw_method='silverman')
    violinPlotEdits(parts)
    ax[1].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax[1].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
# add title and axis labels
axisLabelsLimits(ax[1], bandStr, limits=[-0.05, 0.05],
                 ylabel='Error in \n coarse mode\n AOD')

aaodLst = []
for i in range(len(bands)):
    # =============================================================================
    # AAOD
    # =============================================================================
    true = np.asarray([(1-rf['ssa'][i])*rf['aod'][i] for rf in simBase.rsltFwd])[keepIndAll]
    rtrv = np.asarray([(1-rb['ssa'][i])*rb['aod'][i] for rb in simBase.rsltBck])[keepIndAll]
    
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    aaodLst.append((rtrv-true).T)
    parts = ax[2].violinplot((rtrv-true).T, [i+1], points= 300,
                             showmeans=True, showmedians=True,
                             showextrema=False,
                             bw_method='silverman')
    violinPlotEdits(parts)
    ax[2].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax[2].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
# add title and axis labels
axisLabelsLimits(ax[2], bandStr, limits=[-0.015, 0.015],
                 ylabel='Error in \n AAOD')


# %%Plotting n,k
# =============================================================================

# define the fig and axes handles
# Figure for spectral error plots
fig2, ax2 = plt.subplots(4,1, figsize=(6,6))

# loop through bands to calculate the error and bias in individual variables
kLst = []
kCMLst = []

for i in range(len(bands)):
    aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
    # true = np.asarray([rf['k'][i] for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([aodWght(rf['k'][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltBck])[keepInd]
    # Modifying the true value based on the NDIM
    # if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
    # and the coarse mode 'sea salt' have different value. So based on the dimension of the var
    # We can distinguish each run type and generalize the code
    tempInd = 0
    for nMode_ in [0,4]:        
        rtrv = np.asarray([rf['k'][:,i] for rf in simBase.rsltBck])[keepInd][:,tempInd]
        
        if nMode_ == 0:
            # true = np.asarray([aodWght(rf['k'][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltFwd])[keepInd]
            true = np.asarray([rf['k'][:,i] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
            Rcoef = np.corrcoef(true, rtrv)[0,1]
            RMSE = np.sqrt(np.median((true - rtrv)**2))
            bias = np.mean((rtrv-true))
            
            kLst.append((rtrv-true).T)
            parts = ax2[0].violinplot((rtrv-true).T, [i+1], points= 300,
                                    showmeans=True, showmedians=True,
                                    showextrema=False,
                                    bw_method='silverman')
            violinPlotEdits(parts)
            ax2[0].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
            ax2[0].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

        else:
            true = np.asarray([rf['k'][:,i] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
            Rcoef = np.corrcoef(true, rtrv)[0,1]
            RMSE = np.sqrt(np.median((true - rtrv)**2))
            bias = np.mean((rtrv-true))
            kCMLst.append((rtrv-true).T)
            parts = ax2[1].violinplot((rtrv-true).T, [i+1], points= 300,
                                    showmeans=True, showmedians=True,
                                    showextrema=False,
                                    bw_method='silverman')
            violinPlotEdits(parts)
            ax2[1].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
            ax2[1].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        tempInd += 1
        

axisLabelsLimits(ax2[0], bandStr, limits=[-0.01, 0.01],
                 ylabel=r'Error in k$_{fine}$')
axisLabelsLimits(ax2[1], bandStr, limits=[-0.001, 0.001],
                 ylabel=r'Error in k$_{coarse}$')

ssaLst=[]
for i in range(len(bands)):
    true = np.asarray([rf['ssa'][i] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['ssa'][i] for rf in simBase.rsltBck])[keepInd]
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    ssaLst.append((rtrv-true).T)
    parts = ax2[2].violinplot((rtrv-true).T, [i+1], points= 300,
                            showmeans=True, showmedians=True,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax2[2].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax2[2].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
axisLabelsLimits(ax2[2], bandStr, limits=[-0.1, 0.1],
                 ylabel=r'Error in SSA$_{total}$') 

ssaCMLst=[]
for i in range(len(bands)):
    true = np.asarray([rf['ssaMode'][:,i] for rf in simBase.rsltFwd])[keepInd][:,4]
    rtrv = np.asarray([rf['ssaMode'][:,i] for rf in simBase.rsltBck])[keepInd][:,1]
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    ssaCMLst.append((rtrv-true).T)
    parts = ax2[3].violinplot((rtrv-true).T, [i+1], points= 300,
                            showmeans=True, showmedians=True,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax2[3].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax2[3].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
axisLabelsLimits(ax2[3], bandStr, limits=[-0.05, 0.05],
                 ylabel=r'Error in SSA$_{coarse}$')


# %%
# define the fig and axes handles
# Figure for spectral error plots
fig3, ax3 = plt.subplots(4,1, figsize=(6,6))

# loop through bands to calculate the error and bias in individual variables
nLst = []
nCMLst = []

for i in range(len(bands)):
    # Modifying the true value based on the NDIM
    # if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
    # and the coarse mode 'sea salt' have different value. So based on the dimension of the var
    # We can distinguish each run type and generalize the code
    tempInd = 0
    for nMode_ in [0,4]:
        rtrv = np.asarray([rf['n'][:,i] for rf in simBase.rsltBck])[keepInd][:,tempInd]
        if nMode_ == 0:
            true = np.asarray([rf['n'][:,i] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
            # true = np.asarray([aodWght(rf['n'][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltFwd])[keepInd]
            Rcoef = np.corrcoef(true, rtrv)[0,1]
            RMSE = np.sqrt(np.median((true - rtrv)**2))
            bias = np.mean((rtrv-true))
            
            nLst.append((rtrv-true).T)
            parts = ax3[0].violinplot((rtrv-true).T, [i+1], points= 300,
                                    showmeans=True, showmedians=True,
                                    showextrema=False, 
                                    bw_method='silverman')
            violinPlotEdits(parts)
            ax3[0].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
            ax3[0].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

        else:
            true = np.asarray([rf['n'][:,i] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
            Rcoef = np.corrcoef(true, rtrv)[0,1]
            RMSE = np.sqrt(np.median((true - rtrv)**2))
            bias = np.mean((rtrv-true))
            
            nLst.append((rtrv-true).T)
            parts = ax3[1].violinplot((rtrv-true).T, [i+1], points= 300,
                                    showmeans=True, showmedians=True,
                                    showextrema=False,
                                    bw_method='silverman')
            violinPlotEdits(parts)
            ax3[1].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
            ax3[1].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        tempInd += 1
axisLabelsLimits(ax3[0], bandStr, limits=[-0.2, 0.2],
                 ylabel=r'Error in n$_{fine}$')
axisLabelsLimits(ax3[1], bandStr, limits=[-0.2, 0.2],
                 ylabel=r'Error in n$_{coarse}$')

ssaFMLst=[]
for i in range(len(bands)):
    true = np.asarray([aodWght(rf['ssaMode'][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['ssaMode'][:,i] for rf in simBase.rsltBck])[keepInd][:,0]
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    ssaFMLst.append((rtrv-true).T)
    parts = ax3[2].violinplot((rtrv-true).T, [i+1], points= 300,
                            showmeans=True, showmedians=True,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax3[2].text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax3[2].text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
axisLabelsLimits(ax3[2], bandStr, limits=[-0.1, 0.1],
                 ylabel=r'Error in SSA$_{fine}$')
#%% Spectrally independent

fig4, ax4 = plt.subplots(1,1, figsize=(3,3))

volConcFineLst = []
volConcCoarseLst = []
tempInd = 0
for nMode_ in [0,4]:
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,tempInd]
    if nMode_ == 0:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]*4
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcFineLst.append((rtrv-true).T)
        parts = ax4.violinplot((rtrv-true).T, [1], points= 300,
                                showmeans=True, showmedians=True,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax4.text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        ax4.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    else:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcCoarseLst.append((rtrv-true).T)
        parts = ax4.violinplot((rtrv-true).T, [2], points= 300,
                                showmeans=True, showmedians=True,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax4.text(2.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        ax4.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    tempInd += 1
axisLabelsLimits(ax4, ['', 'Fine mode\n vol. conc.', '', 'Coarse mode\n vol. conc.'], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error', xlabel='')
#%% Effective raidius
fig5, ax5 = plt.subplots(1,2, figsize=(5,3))

subMicronEffRadLst = []
aboveMicronEffRadLst = []

# Fine mode rEff
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]

Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))

subMicronEffRadLst.append((rtrv-true).T)
parts = ax5[0].violinplot((rtrv-true).T, [1], points= 300,
                        showmeans=True, showmedians=True,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax5[0].text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
ax5[0].text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
# Coarse mode rEff
true = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltBck])[keepInd]

Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))

aboveMicronEffRadLst.append((rtrv-true).T)
parts = ax5[1].violinplot((rtrv-true).T, [1], points= 300,
                        showmeans=True, showmedians=True,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax5[1].text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
ax5[1].text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
axisLabelsLimits(ax5[0], ['', '', r'Submicron r$_{eff}$', ], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error ($\mu$m)', xlabel='')
axisLabelsLimits(ax5[1], ['', '', r'above micron r$_{eff}$'], limits=[-2, 2],
                 ylabel='', xlabel='')
#%% Spectrally independent SF

fig6, ax6 = plt.subplots(1,1, figsize=(3,3))

fineSFLst = []
coarseSFLst = []

true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[keepInd]
# Modifying the true value based on the NDIM
# if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
# and the coarse mode 'sea salt' have different value. So based on the dimension of the var
# We can distinguish each run type and generalize the code
if true.ndim >1:
    true = np.asarray([rf['sph']for rf in simBase.rsltFwd])[keepInd][:,nMode]
    rtrv = np.asarray([rf['sph']for rf in simBase.rsltBck])[keepInd][:,0]
    
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))

fineSFLst.append((rtrv-true).T)
parts = ax6.violinplot((rtrv-true).T, [1], points= 300,
                        showmeans=True, showmedians=True,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax6.text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
ax6.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

# coarse mode SF
true = np.asarray([rf['sph']for rf in simBase.rsltFwd])[keepInd][:,4]
rtrv = np.asarray([rf['sph']for rf in simBase.rsltBck])[keepInd][:,1]

Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))

coarseSFLst.append((rtrv-true).T)
parts = ax6.violinplot((rtrv-true).T, [2], points= 300,
                        showmeans=True, showmedians=True,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax6.text(2.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
ax6.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
axisLabelsLimits(ax6, ['', 'Fine mode\n SF', '', 'Coarse mode\n SF.'], limits=[-50, 50],
                 ylabel=r'Absolute error (%)', xlabel='')

#%% AOD weighted vol. conc.
fig7, ax7 = plt.subplots(1,1, figsize=(3,3))

volConcFineAODLst = []
volConcCoarseAODLst = []
tempInd = 0
for nMode_ in [0,4]:
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,tempInd]
    if nMode_ == 0:
        true = np.asarray([aodWght(rf['vol'], rf['aodMode'][:,waveInd]) for rf in simBase.rsltFwd])[keepInd]
        # true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]*4
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcFineLst.append((rtrv-true).T)
        parts = ax7.violinplot((rtrv-true).T, [1], points= 300,
                                showmeans=True, showmedians=True,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax7.text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        ax7.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    else:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcCoarseLst.append((rtrv-true).T)
        parts = ax7.violinplot((rtrv-true).T, [2], points= 300,
                                showmeans=True, showmedians=True,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax7.text(1+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        ax7.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    tempInd += 1
axisLabelsLimits(ax7, ['', 'Fine mode\n vol. conc.', '', 'Coarse mode\n vol. conc.'], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error', xlabel='')

#%% Surface
fig8, ax8 = plt.subplots(1,1, figsize=(6,3))

wtrSrfLst = []
for i in range(len(bands)):
    true = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltFwd])
    rtrv = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltBck])
    
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    wtrSrfLst.append((rtrv-true).T)
    parts = ax8.violinplot((rtrv-true).T, [i+1], points= 300,
                            showmeans=True, showmedians=True,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax8.text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    ax8.text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
axisLabelsLimits(ax8, bandStr, limits=[-0.01, 0.01],
                 ylabel=r'Error in L$_{water}$')
    
# %%
# =============================================================================
# Save the figure
# =============================================================================
figSavePath = saveFN.replace('.pkl',('_%s_%s_spectral.png' % (surf2plot, fnTag)))
print('Saving figure to: %s' % (os.path.join(inDirPath,figSavePath)))
ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (saveFN, simBase.rsltFwd[0]['lambda'][waveInd], surf2plot, aodMin)

fig.suptitle(ttlStr.replace('MERGED_',''))

fig2.suptitle(ttlStr.replace('MERGED_',''))
fig.savefig(inDirPath + figSavePath, dpi=330)
fig2.savefig(inDirPath + figSavePath.replace('MERGED_','k_'), dpi=330)
fig3.savefig(inDirPath + figSavePath.replace('MERGED_','n_'), dpi=330)
fig4.savefig(inDirPath + figSavePath.replace('MERGED_','vol_'), dpi=330)
fig5.savefig(inDirPath + figSavePath.replace('MERGED_','rEff_'), dpi=330)
fig6.savefig(inDirPath + figSavePath.replace('MERGED_','SF_'), dpi=330)
fig7.savefig(inDirPath + figSavePath.replace('MERGED_','aodWeightVol_'), dpi=330)
fig8.savefig(inDirPath + figSavePath.replace('MERGED_','L_'), dpi=330)
print('Saving figure to: %s' % (os.path.join(inDirPath,figSavePath.replace('MERGED_','Hist_'))))
# plt.show()
# %%
