#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script is a modified version of the simulationQuickLook.py, with 
the ability to plot 2D density and histograms of errors for retrieved parameters.
Specifically designed to accommodate the simulation retrieval study based on
CAMP2Ex measurements

Created on Wed Jul 13 14:18:11 2022

@author: anin puthukkudy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Import the librarires
# =============================================================================
# automatic reload option for IPython
%load_ext autoreload
%autoreload 2

import os
from glob import glob
from pprint import pprint
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'});mpl.rcParams.update({'ytick.right': 'True'});mpl.rcParams.update({'xtick.top': 'True'});plt.rcParams["font.family"] = "Latin Modern Math"; plt.rcParams["mathtext.fontset"] = "cm"; plt.rcParams["figure.dpi"]=330
from scipy import interpolate
from scipy.stats import norm, gaussian_kde, ncx2, moyal
from simulateRetrieval import simulation
import seaborn as sns
from matplotlib.colors import LogNorm
import pickle

# =============================================================================
# Local definitions
# =============================================================================
def violinPlotEdits(parts, facecolor=False):
    '''
    Violin plot edits
    
    Parameters
    ----------
    parts : dict
        Dictionary of violin plot elements
    facecolor : bool, optional
        Change the color of the violin plot. The default is False.
    Returns
    -------
    None
    
    '''
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
    '''
    axisLabelsLimits edits the axis labels and limits of the plot
    
    Parameters
    ----------
    axs : matplotlib.axes._subplots.AxesSubplot
        Axes object
    bandStr : list
        List of strings of spectral bands
    limits : list
        List of limits for the y-axis
    xlabel : str, optional
        Label for the x-axis. The default is r'Spectral bands ($\mu$m)'.
    ylabel : str, optional
        Label for the y-axis. The default is 'Error in AOD'.
    
    Returns
    -------
    None
    '''
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_ylim(limits)
    axs.set_xticklabels(bandStr)
    axs.yaxis.grid(True)
    
def scatter_density(x, y, cmap='jet', alpha=0.5, sizeM=1, cBar=False,
                    xmin=None, xmax=None, ymin=None, ymax=None,
                    xAxis = None, yAxis=None):
    """
    Create a scatter plot with 2D density overlay.

    Parameters:
    ----------
        x (array-like): X values.
        y (array-like): Y values.
        cmap (str): Colormap (default: 'jet').
        alpha (float): Alpha value for scatter points (default: 0.5).
        sizeM (float): Size of the marker (default: 1).
        cBar (bool): Add colorbar (default: False).
        xmin (float): Minimum x value (default: None).
        xmax (float): Maximum x value (default: None).
        ymin (float): Minimum y value (default: None).
        ymax (float): Maximum y value (default: None).
        xAxis (str): X-axis label (default: None).
        yAxis (str): Y-axis label (default: None).

    Returns:
    -------
        None
    """
    # Calculate the density of the data
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Clip the density values between 0 and 1
    density = np.clip(density, 0, 1)

    # Normalize the density values to range [0, 1]
    density_norm = (density - np.min(density)) / (np.max(density) - np.min(density))
    
    # Create a scatter plot with density overlay
    fig, ax = plt.subplots(figsize=(3,3))
    im = ax.scatter(x, y, c=density_norm,
                cmap=cmap, alpha=alpha, s=sizeM)
    if xAxis is not None:      
        ax.set_xlabel(xAxis)
    if yAxis is not None:
        ax.set_ylabel(yAxis)
  
    # Set the x and y limits
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if cBar:
        plt.colorbar(im)
    plt.show()
    del im

def makesweetgraph(x=None, y=None, cmap='jet', ylab=None, xlab=None,
                   bins=100, sets=None, figsize=(3,3), snsbins=100):
    '''
    This function creates a 2D histogram with marginal histograms
    
    Parameters
    ----------
    x : array-like
        x-axis values
    y : array-like
        y-axis values
    cmap : str, optional
        Colormap. The default is 'jet'.
    ylab : str, optional
        y-axis label. The default is None.
    xlab : str, optional
        x-axis label. The default is None.
    bins : int, optional
        Number of bins for the 2D histogram. The default is 100.
    sets : list, optional
        List of strings for the legend. The default is None.
    figsize : tuple, optional
        Figure size. The default is (3,3).
    snsbins : int, optional
        Number of bins for the marginal histograms. The default is 100.
    
    Returns
    -------
    None
    
    '''
    ax1 = sns.jointplot(x=x, y=y,marginal_kws=dict(bins=snsbins))
    ax1.fig.set_size_inches(figsize[0], figsize[1])
    ax1.ax_joint.cla()
    plt.sca(ax1.ax_joint)
    
    # create the 2D histogram
    plt.hist2d(x,y,bins,norm=LogNorm(),cmap=cmap)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    cbar_ax = ax1.fig.add_axes([1, 0.1, .03, .7])
    
    # add the colorbar
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label(r'$\log_{10}$ density of points')

aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
    
def plotSpectralVars(statDict_, specVar, bands, bandStr,
                     simBase, keepInd,
                     ax, Lst, limits=[-0.1, 0.1],
                     diffMode=False, modeIdx=0,
                     yStr=None, aodWgt=False, 
                     spclVar=None):
    '''
    Plots the spectral variables for the forward and backward simulations
    Then plots the difference between the two and calculates the RMSE, bias, etc.
    Then plot error distribution using violin plots
    
    Parameters
    ----------
    statDict_ : dict
        Dictionary containing the statistics for the spectral variables
    specVar : str
        Spectral variable to plot
    bands : list
        List of spectral bands
    bandStr : list
        List of spectral bands as strings
    simBase : object
        Object containing the forward and backward simulations
    ax : object
        Axis object
    Lst : list
        List of differences
    limits : list, optional
        Limits for the y-axis. The default is [-0.1, 0.1].
    diffMode : bool, optional
        If True, then plot the difference between the forward and backward simulations for fine and coarse modes. The default is False.
    modeIdx : int, optional
        Index of the mode to plot. The default is 0. for fine mode.
        1 is for coarse mode.
    aodWgt : bool, optional
        If True, then plot the spectral variables weighted by AOD. The default is False.
    spclVar : str, optional
        Special variable to calculate stats. The default is None.
    
    Returns
    -------
    None

    '''
    
    #FIXME: This is a hack to get the correct band for the forward and backward simulations of fine and coarse mode
    if modeIdx == 0: trueMode=0
    if modeIdx == 1: trueMode=4
    
    # initiate the class for storing statistics
    if spclVar is None:
        spclVar = specVar
        
    sc = statsCalc(statDict_, spclVar)
    
    # Plot the spectral variables for the forward and backward simulations
    # loop through the spectral bands
    
    for i in range(len(bands)):
        if diffMode:
            #FIXME: This is a hack to get the correct band for the forward and backward simulations of fine and coarse mode
            if aodWgt:             
                true = np.asarray([aodWght(rf[specVar][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltFwd])[keepInd]
            else:
                true = np.asarray([rf[specVar][:,i] for rf in simBase.rsltFwd])[keepInd][:,trueMode]
            rtrv = np.asarray([rf[specVar][:,i] for rf in simBase.rsltBck])[keepInd][:,modeIdx]
        else:
            if aodWgt: 
                true = np.asarray([(1-rf[specVar][i])*rf['aod'][i] for rf in simBase.rsltFwd])[keepInd]
                rtrv = np.asarray([(1-rb[specVar][i])*rb['aod'][i] for rb in simBase.rsltBck])[keepInd]
            else:  
                true = np.asarray([rf[specVar][i] for rf in simBase.rsltFwd])[keepInd]
                rtrv = np.asarray([rf[specVar][i] for rf in simBase.rsltBck])[keepInd]
        
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        sc.calc(statDict_, spclVar, true, rtrv)
        
        Lst.append((rtrv-true).T)
        
        parts = ax.violinplot((rtrv-true).T, [i+1], points= 300,
                                showmeans=True, showmedians=False,
                                showextrema=False, quantiles=[0.1, 0.9],
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax.text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        # ax.text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    
    # Change the y axis labels
    if yStr is None:
        yStr = specVar
        
    axisLabelsLimits(ax, bandStr, limits=limits,
                     ylabel='Error in \n %s' %yStr) 

class statsCalc():
    '''
    Calculates the statistics for the given variable
    
    The parameters inclues R, RMSE, bias, and the 10th and 90th percentile of the error distribution
    
    '''
    def __init__(self, dct, varStr):
        dct[varStr] = {}
        dct[varStr]['R'] = []
        dct[varStr]['RMSE'] = []
        dct[varStr]['bias'] = []
        dct[varStr]['10th'] = []
        dct[varStr]['90th'] = []
        
    def calc(self, dct, varStr, true, rtrv):
        dct[varStr]['R'].append(np.corrcoef(true, rtrv)[0,1])
        dct[varStr]['RMSE'].append(np.sqrt(np.median((true - rtrv)**2)))
        dct[varStr]['bias'].append(np.mean((rtrv-true)))
        dct[varStr]['10th'].append(np.percentile((rtrv-true), 10))
        dct[varStr]['90th'].append(np.percentile((rtrv-true), 90))
    

# =============================================================================
# %% Initiation
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

# define a dict to store the statisctics
statsDict_ = {}

# define the fig and axes handles
# Figure for spectral error plots
fig, ax = plt.subplots(3,1, figsize=(6,4))

# wavelengths
bands = simBase.rsltFwd[0]['lambda']
bandStr = ['']+[str(i) for i in bands]

# loop through bands to calculate the error and bias in individual variables
aodLst = []
plotSpectralVars(statsDict_, 'aod', bands, bandStr, simBase, keepInd, ax[0], aodLst, yStr='AOD')

aodCMLst = []
plotSpectralVars(statsDict_, 'aodMode', bands, bandStr, simBase, keepInd, ax[1], aodCMLst,
                 diffMode=True,
                 modeIdx=1, yStr='AOD CM', spclVar='aodCM')

aaodLst = []
plotSpectralVars(statsDict_, 'ssa', bands, bandStr, simBase, keepIndAll, ax[2],
                 aaodLst, yStr='AAOD',
                 aodWgt=True, limits=[-0.025,0.025], spclVar='aaod')

# =============================================================================
# %%Plotting n,k
# =============================================================================

# define the fig and axes handles
# Figure for spectral error plots
fig2, ax2 = plt.subplots(4,1, figsize=(6,6))

# loop through bands to calculate the error and bias in individual variables
ssaLst = []
plotSpectralVars(statsDict_, 'ssa', bands, bandStr, simBase, 
                 keepInd, ax2[0], ssaLst, yStr='SSA',
                 )

ssaCMLst = []
plotSpectralVars(statsDict_, 'ssaMode', bands, bandStr, simBase, keepInd,
                 ax2[1], ssaCMLst, diffMode=True,
                 modeIdx=1, yStr='SSA CM', spclVar='ssaCM')
kFMLst = []
plotSpectralVars(statsDict_, 'k', bands, bandStr, simBase, keepInd,
                 ax2[2], kFMLst, diffMode=True,
                 modeIdx=0, yStr='k FM', limits=[-0.01,0.01], spclVar='kFM')
kCMLst = []
plotSpectralVars(statsDict_, 'k', bands, bandStr, simBase, keepInd,
                 ax2[3], kCMLst, diffMode=True,
                 modeIdx=1, yStr='k CM', limits=[-0.002,0.002], spclVar='kCM')

# %%
# define the fig and axes handles
# Figure for spectral error plots
fig3, ax3 = plt.subplots(3,1, figsize=(6,4))

nFMLst = []
plotSpectralVars(statsDict_, 'n', bands, bandStr, simBase, keepInd,
                 ax3[2], kFMLst, diffMode=True,
                 modeIdx=0, yStr='n FM', limits=[-0.1,0.1], spclVar='nFM')

nCMLst = []
plotSpectralVars(statsDict_, 'n', bands, bandStr, simBase,
                 keepInd, ax3[1], kFMLst, diffMode=True,
                 modeIdx=1, yStr='n CM', limits=[-0.1,0.1], spclVar='nCM')

ssaFMLst = []
plotSpectralVars(statsDict_, 'ssaMode', bands, bandStr, simBase,
                 keepInd, ax3[0], ssaCMLst, diffMode=True,
                 modeIdx=0, yStr='SSA FM', spclVar='ssaFM')

#%% Spectrally independent

fig4, ax4 = plt.subplots(1,1, figsize=(3,2))

volConcFineLst = []
volConcCoarseLst = []
tempInd = 0
for nMode_ in [0,4]:
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,tempInd]
    if nMode_ == 0:
        spclVar = 'volFine'
        sc = statsCalc(statsDict_, spclVar)
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]*4
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcFineLst.append((rtrv-true).T)
        parts = ax4.violinplot((rtrv-true).T, [1], points= 300,
                                showmeans=True, showmedians=False,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax4.text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        # ax4.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        sc.calc(statsDict_, spclVar, true, rtrv)
    else:
        spclVar = 'volCoarse'
        sc = statsCalc(statsDict_, spclVar)
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcCoarseLst.append((rtrv-true).T)
        parts = ax4.violinplot((rtrv-true).T, [2], points= 300,
                                showmeans=True, showmedians=False,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax4.text(2.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        # ax4.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        sc.calc(statsDict_, spclVar, true, rtrv)
    tempInd += 1
axisLabelsLimits(ax4, ['', 'Fine mode\n vol. conc.', '', 'Coarse mode\n vol. conc.'], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error', xlabel='')

#%% Effective raidius
fig5, ax5 = plt.subplots(1,2, figsize=(3,2))

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
                        showmeans=True, showmedians=False,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax5[0].text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
# ax5[0].text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

spclVar = 'rEffFine'
sc = statsCalc(statsDict_, spclVar)
sc.calc(statsDict_, spclVar, true, rtrv) 

# Coarse mode rEff
true = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltBck])[keepInd]

Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))

aboveMicronEffRadLst.append((rtrv-true).T)
parts = ax5[1].violinplot((rtrv-true).T, [1], points= 300,
                        showmeans=True, showmedians=False,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax5[1].text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
# ax5[1].text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

spclVar = 'rEffCoarse'
sc = statsCalc(statsDict_, spclVar)
sc.calc(statsDict_, spclVar, true, rtrv) 

axisLabelsLimits(ax5[0], ['', '', r'Submicron r$_{eff}$', ], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error ($\mu$m)', xlabel='')
axisLabelsLimits(ax5[1], ['', '', r'above micron r$_{eff}$'], limits=[-2, 2],
                 ylabel='', xlabel='')

#%% Spectrally independent SF
fig6, ax6 = plt.subplots(1,1, figsize=(3,2))

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
                        showmeans=True, showmedians=False,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax6.text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
# ax6.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

spclVar = 'sphFine'
sc = statsCalc(statsDict_, spclVar)
sc.calc(statsDict_, spclVar, true, rtrv) 

# coarse mode SF
true = np.asarray([rf['sph']for rf in simBase.rsltFwd])[keepInd][:,4]
rtrv = np.asarray([rf['sph']for rf in simBase.rsltBck])[keepInd][:,1]

Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))

coarseSFLst.append((rtrv-true).T)
parts = ax6.violinplot((rtrv-true).T, [2], points= 300,
                        showmeans=True, showmedians=False,
                        showextrema=False,
                        bw_method='silverman')
violinPlotEdits(parts)
ax6.text(2.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
# ax6.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

spclVar = 'sphCoarse'
sc = statsCalc(statsDict_, spclVar)
sc.calc(statsDict_, spclVar, true, rtrv) 

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
                                showmeans=True, showmedians=False,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax7.text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        # ax7.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        spclVar = 'volFineAOD'
        sc = statsCalc(statsDict_, spclVar)
        sc.calc(statsDict_, spclVar, true, rtrv) 
    else:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        
        volConcCoarseLst.append((rtrv-true).T)
        parts = ax7.violinplot((rtrv-true).T, [2], points= 300,
                                showmeans=True, showmedians=False,
                                showextrema=False,
                                bw_method='silverman')
        violinPlotEdits(parts)
        ax7.text(1+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
        # ax7.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        
        spclVar = 'volFineCoarse'
        sc = statsCalc(statsDict_, spclVar)
        sc.calc(statsDict_, spclVar, true, rtrv)
        
    tempInd += 1
axisLabelsLimits(ax7, ['', 'Fine mode\n vol. conc.', '', 'Coarse mode\n vol. conc.'], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error', xlabel='')

#%% Surface
fig8, ax8 = plt.subplots(1,1, figsize=(6,3))

wtrSrfLst = []
spclVar = 'wtrSurf'
sc = statsCalc(statsDict_, spclVar)
for i in range(len(bands)):
    true = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltFwd])
    rtrv = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltBck])
    
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    wtrSrfLst.append((rtrv-true).T)
    parts = ax8.violinplot((rtrv-true).T, [i+1], points= 300,
                            showmeans=True, showmedians=False,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax8.text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    # ax8.text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    
    sc.calc(statsDict_, spclVar, true, rtrv)
axisLabelsLimits(ax8, bandStr, limits=[-0.01, 0.01],
                 ylabel=r'Error in L$_{water}$')

# FMF distribution plot using simulated data and retrieved data
true = np.asarray([rf['aodMode'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,4]
rtrv = np.asarray([rf['aodMode'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,1]
trueT = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrvT = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
cmf_rtrv = rtrv/rtrvT
cmf = true/trueT

# plot the figure
fig10, ax10 = plt.subplots(figsize=(3,3), dpi=300)
plt.hist(1-cmf, alpha=0.3, bins=np.linspace(0,1,100), label='true')
plt.hist(1-cmf_rtrv, alpha=0.3, bins=np.linspace(0,1,100), label='rtrv')
plt.legend()
plt.tight_layout()
plt.xlabel('FMF')
plt.ylabel('Frequency')

# =============================================================================
# %% Save the figure
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
fig10.savefig(inDirPath + figSavePath.replace('MERGED_','FMF_'), dpi=330)
print('Saving figure to: %s' % (os.path.join(inDirPath,figSavePath.replace('MERGED_','Hist_'))))
# plt.show()
# %% Save the stats

# open a file, where you ant to store the data
f = open(os.path.join(inDirPath,figSavePath.replace('MERGED_','STATS_').replace('.png','.pkl')), 'wb')

try:
    # write data to the file
    pickle.dump(statsDict_, f)
    print('Saving stats to: %s' % (os.path.join(inDirPath,figSavePath.replace('MERGED_','STATS_').replace('.png','.pkl'))))
except Exception as e:
    print(e)
# close the file
f.close()
    


# %%
