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
from scipy import interpolate
from scipy.stats import norm
from simulateRetrieval import simulation
from glob import glob

# =============================================================================
# Definition to plot the 2D histo gram
# =============================================================================
def density_scatter(x, y, ax=None, fig=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpolate.interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=5, alpha=0.1, **kwargs)

    norm = mpl.colors.Normalize(vmin=np.max([0, np.min(z)]), vmax=np.max(z))
    if fig:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm), ax=ax)
        cbar.ax.set_ylabel('Density')

    return ax

def plotProp(axs, titleStr ='', scale='linear', ylabel=None,
             xlabel=None, stat=True, MinMax=True):
    
    # min max
    if MinMax:
        axs.plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121) # line plot
        axs.set_xlim(minAOD,maxAOD)
        axs.set_ylim(minAOD,maxAOD)
    
    # Title of the plot
    axs.set_title(titleStr)
    
    # x and y label
    if ylabel:
        axs.set_ylabel(ylabel)
    if xlabel:
        axs.set_xlabel(xlabel)
    # scale
    if scale == 'log':
        axs.set_xscale('log')
        axs.set_yscale('log')
    elif scale == 'linear':
        axs.ticklabel_format(axis='both', style='plain', useOffset=False)
        
    # Plot the statistics
    if stat:
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
        tHnd = axs.annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)
        textstr = frmt % (Rcoef, RMSE, bias)
        tHnd = axs.annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                            textcoords='offset points', color=clrText, fontsize=FS)
    return tHnd

def modifiedHist(x, axs, fig=None, titleStr ='', xscale=None, ylabel=None,
             xlabel=None, stat=True, MinMax=False, nBins=20, err=0.05,
             yscale=None):

    # Creating histogram
    N, bins, patches = axs.hist(x, bins = nBins, density=True)
     
    # Setting color
    fracs = ((N**(1 / 2)) / N.max())
    norm_ = mpl.colors.Normalize(fracs.min(), fracs.max())
     
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm_(thisfrac))
        thispatch.set_facecolor(color)
    # min max
    if MinMax:
        axs.set_xlim(-err, err)
    
    # Title of the plot
    axs.set_title(titleStr)
    
    # x and y label
    if ylabel:
        axs.set_ylabel(ylabel)
    if xlabel:
        axs.set_xlabel(xlabel)
    # scale
    if yscale == 'log':
        axs.set_yscale('log')
    if xscale == 'log':
        axs.set_xscale('log')
        
    # Fit a normal distribution to the data:
    # mean and standard deviation
    if stat:
        mu, std = norm.fit(x)
        # Plot the PDF.
        x_ = [abs(np.percentile(x,1)), 
              abs(np.percentile(x,99))]
        
        xmax = np.max(x_)
        xmin = -xmax
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        axs.plot(x, p, 'C03', linewidth=1.5)
        
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        frmt = 'RMS=%5.3f\nbias=%5.3f'
        textstr = frmt % (RMSE, bias)
        tHnd = axs.annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top',
                            xycoords='axes fraction',
                            textcoords='offset points', color=clrText,
                            fontsize=FS)
    
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
fnPtrn = 'megaharp01_2_2modes_AOD_*_550nm*_addCoarse__campex_flight#*_layer#00.pkl'
# fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_1000z.pkl'

# Location/dir where the pkl files are
inDirPath = '/Users/aputhukkudy/Working_Data/ACCDAM/2022/Campex_Simulations/'

# more tags and specifiations for the scatter plot
surf2plot = 'ocean' # land, ocean or both
aodMin = 0.2 # does not apply to first AOD plot
nMode = 0 # Select which layer or mode to plot
fnTag = 'AllCases'
xlabel = 'Simulated Truth'
MS = 2
FS = 10
LW121 = 1
pointAlpha = 0.10
clrText = [0.5,0,0.0]
nBins = 200 # no. of bins for histogram

# define the fig and axes handles
# Figure for 2D density plots
fig, ax = plt.subplots(3,5, figsize=(15,9))
plt.locator_params(nbins=3)
# Figure for histograms
fig_hist, ax_hist = plt.subplots(3,5, figsize=(15,9))
plt.locator_params(nbins=3)
lightSave = True # Omit PM elements and extinction profiles from MERGED files to save space

# Define the path of the new merged pkl file
saveFN = 'MERGED_'+fnPtrn.replace('*','ALL')
savePATH = os.path.join(inDirPath,saveFN)

# If already exists, load the file
if os.path.exists(savePATH):
    simBase = simulation(picklePath=savePATH)
    print('Loading from %s - %d' % (saveFN, len(simBase.rsltBck)))
else:
    files = glob(os.path.join(inDirPath, fnPtrn))
    assert len(files)>0, 'No files found!'
    simBase = simulation()
    simBase.rsltFwd = np.empty(0, dtype=dict)
    simBase.rsltBck = np.empty(0, dtype=dict)
    print('Building %s - Nfiles=%d' % (saveFN, len(files)))
    for file in files: # loop over all available nAng
        simA = simulation(picklePath=file)
        if lightSave:
            for pmStr in ['angle', 'p11','p12','p22','p33','p34','p44','range','βext']:
                [rb.pop(pmStr, None) for rb in simA.rsltBck]
        NrsltBck = len(simA.rsltBck)
        print('%s - %d' % (file, NrsltBck))
        Nrepeats = 1 if NrsltBck==len(simA.rsltFwd) else NrsltBck
        for _ in range(Nrepeats): simBase.rsltFwd = np.r_[simBase.rsltFwd, simA.rsltFwd]
        simBase.rsltBck = np.r_[simBase.rsltBck, simA.rsltBck]
    simBase.saveSim(savePATH)
    print('Saving to %s - %d' % (saveFN, len(simBase.rsltBck)))
print('--')

# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][waveInd])
pprint(simBase.analyzeSim(waveInd)[0])

# lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
# TODO: this has to be generalized so that it can be ran without crashing the code
# at the moment, this hard coded to be ocean pixels
lp = np.array([0 for rf in simBase.rsltFwd])
keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1

# apply convergence filter
# simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 90)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])
keepIndAll = keepInd

# variable to color point by in all subplots
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]])
clrVarAll = clrVar
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

# =============================================================================
# Plotting
# =============================================================================

# =============================================================================
# AOD
# =============================================================================
true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]

minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05

density_scatter(true, rtrv, ax=ax[0,0])
plotProp(ax[0,0], titleStr='AOD', scale='log', ylabel='Retrieved')

# histogram
modifiedHist((rtrv-true), axs=ax_hist[0,0], titleStr='AOD', ylabel= 'Density',
             fig=fig_hist, nBins=nBins)

# =============================================================================
# # AAOD
# =============================================================================
true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.9
maxAOD = 0.15

density_scatter(true, rtrv, ax=ax[0,2])
plotProp(ax[0,2], titleStr='AAOD')

# histogram
modifiedHist((rtrv-true), axs=ax_hist[0,2], titleStr='AAOD',
             fig=fig_hist, nBins=nBins)

# apply AOD min after we plot AOD
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>=aodMin for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) 


# apply Reff min
# simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
simBase._addReffMode(1.0, True) # reframe with cut at 1 micron diameter
#keepInd = np.logical_and(keepInd, [rf['rEffMode']>=2.0 for rf in simBase.rsltBck])
#print('%d/%d fit surface type %s and aod≥%4.2f AND retrieved Reff>2.0μm' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
#clrVar = np.sqrt([rb['rEff']/rf['rEff']-1 for rb,rf in zip(simBase.rsltBck[keepInd], simBase.rsltFwd[keepInd])])

# =============================================================================
# # ANGSTROM
# =============================================================================
aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
true = -np.log(aod1/aod2)/logLamdRatio

aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
rtrv = -np.log(aod1/aod2)/logLamdRatio

minAOD = np.percentile(true,1) # BUG: Why is Angstrom >50 in at least one OSSE cases?
maxAOD = np.percentile(true,99)
density_scatter(true, rtrv, ax=ax[0,1])
plotProp(ax[0,1], titleStr='Angstrom Exponent')

# histogram
modifiedHist((rtrv-true), axs=ax_hist[0,1], titleStr='Angstrom Exponent',
             fig=fig_hist, nBins=nBins, yscale='log')

# =============================================================================
# # k
# =============================================================================
aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
true = np.asarray([rf['k'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['k'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
# Modifying the true value based on the NDIM
# if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
# and the coarse mode 'sea salt' have different value. So based on the dimension of the var
# We can distinguish each run type and generalize the code
tempInd = 0
for nMode_ in [0,4]:
    true = np.asarray([rf['k'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
    rtrv = np.asarray([rf['k'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,tempInd]
    
    if nMode_ == 0:
        minAOD = 0.0005
        maxAOD = np.max(true)*1.15
        density_scatter(true, rtrv, ax=ax[1,3])
        plotProp(ax[1,3], titleStr=r'k$_{fine}$', scale='log')
        ax[1,3].set_xticks([0.001, 0.01])
        ax[1,3].set_yticks([0.001, 0.01])
        # histogram
        modifiedHist((rtrv-true), axs=ax_hist[1,3], titleStr=r'k$_{fine}$',
                     fig=fig_hist, nBins=nBins)
    else:
        minAOD = 0.0001
        maxAOD = 0.001
        density_scatter(true, rtrv, ax=ax[1,4])
        plotProp(ax[1,4], titleStr=r'k$_{coarse}$', scale='log')
        ax[1,4].set_xticks([0.0001, 0.001])
        ax[1,4].set_yticks([0.0001, 0.001])
        # histogram
        modifiedHist((rtrv-true), axs=ax_hist[1,4], titleStr=r'k$_{coarse}$',
                     fig=fig_hist, nBins=nBins)
    tempInd += 1

# =============================================================================
# # FMF (vol)
# =============================================================================
def fmfCalc(r,dvdlnr):
    cutRadius = 0.5
    fInd = r<=cutRadius
    logr = np.log(r)
    return np.trapz(dvdlnr[fInd],logr[fInd])/np.trapz(dvdlnr,logr)
try:
    true = np.asarray([fmfCalc(rf['r'], rf['dVdlnr']) for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([fmfCalc(rb['r'][0,:], rb['dVdlnr'].sum(axis=0)) for rb in simBase.rsltBck])[keepInd]
    minAOD = 0.01
    maxAOD = 1.0
    ax[0,4].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[0,4].set_title('Volume FMF')
    ax[0,4].set_xscale('log')
    ax[0,4].set_yscale('log')
    ax[0,4].set_xlim(minAOD,maxAOD)
    ax[0,4].set_ylim(minAOD,maxAOD)
    ax[0,4].set_xticks([minAOD, maxAOD])
    ax[0,4].set_yticks([minAOD, maxAOD])
    ax[0,4].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[0,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[0,4].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)
    
    # g
    true = np.asarray([rf['g'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['g'][waveInd] for rf in simBase.rsltBck])[keepInd]
    minAOD = np.min(true)*0.95
    maxAOD = np.max(true)*1.05
    ax[1,0].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[1,0].set_title('g')
    # ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel('Retrieved')
    ax[1,0].set_xlim(minAOD,maxAOD)
    ax[1,0].set_ylim(minAOD,maxAOD)
    ax[1,0].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[1,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[1,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)
except Exception as err:
    print('Error in plotting FMF: \n error: %s' %err)
    
    # try plotting bland altman
    true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepIndAll]
    rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepIndAll]
    rtrv = true - rtrv
    minAOD = np.min(true)*0.9
    maxAOD = np.max(true)*1.1
    ax[1,0].plot([minAOD,maxAOD], [0,0], 'k', linewidth=LW121)
    ax[1,0].set_title('difference in AOD')
    ax[1,0].set_ylabel('true-retrieved')
    ax[1,0].set_xlim(minAOD,maxAOD)
    ax[1,0].set_ylim(-maxAOD/10,maxAOD/10)

    ax[1,0].set_xscale('log')
    density_scatter(true, rtrv, ax=ax[1,0])

    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    tHnd = ax[1,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)   
    
# =============================================================================
#     # SSA
# =============================================================================
    true = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltBck])[keepInd]
    minAOD = np.min(true)*0.95
    maxAOD = 1

    ax[2,0].set_xlim(minAOD,maxAOD)
    ax[2,0].set_ylim(minAOD,maxAOD)
    
    density_scatter(true, rtrv, ax=ax[2,0])
    plotProp(ax[2,0], titleStr=r'SSA', scale='linear', ylabel='Retrieved',
             xlabel=xlabel)
    # histogram
    modifiedHist((rtrv-true), axs=ax_hist[1,0], titleStr='SSA',
                 fig=fig_hist, nBins=nBins, ylabel= 'Density', 
                 xlabel='Retrieved-Simulated')
    

# =============================================================================
# # spherical fraction
# =============================================================================
true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[keepInd]
# Modifying the true value based on the NDIM
# if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
# and the coarse mode 'sea salt' have different value. So based on the dimension of the var
# We can distinguish each run type and generalize the code
if true.ndim >1:
    true = np.asarray([rf['sph']for rf in simBase.rsltFwd])[keepInd][:,nMode]
minAOD = 0
maxAOD = 100.1

ax[0,3].set_xticks(np.arange(minAOD, maxAOD, 25))
ax[0,3].set_yticks(np.arange(minAOD, maxAOD, 25))

density_scatter(true, rtrv, ax=ax[0,3])
plotProp(ax[0,3], titleStr='spherical vol. frac.')

# histogram
modifiedHist((rtrv-true), axs=ax_hist[0,3], titleStr='spherical vol. frac.',
             fig=fig_hist, nBins=nBins)

# =============================================================================
# # rEff
# =============================================================================
#simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.8
maxAOD = np.max(true)*1.2

density_scatter(true, rtrv, ax=ax[0,4])
plotProp(ax[0,4], titleStr='Submicron r_eff', scale=None)
# histogram
modifiedHist((rtrv-true), axs=ax_hist[0,4], titleStr='Submicron r_eff',
             fig=fig_hist, nBins=nBins)

# Corase mode rEff
true = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.8
maxAOD = np.max(true)*1.2

density_scatter(true, rtrv, ax=ax[2,1])
plotProp(ax[2,1], titleStr='above micron r_eff', scale=None, xlabel=xlabel,
         MinMax=False)
# histogram
modifiedHist((rtrv-true), axs=ax_hist[2,1], titleStr='above micron r_eff',
             fig=fig_hist, nBins=nBins, xlabel='Retrieved-Simulated')
# =============================================================================
# # n
# =============================================================================
true = np.asarray([rf['n'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([aodWght(rf['n'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
# Modifying the true value based on the NDIM
# if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
# and the coarse mode 'sea salt' have different value. So based on the dimension of the var
# We can distinguish each run type and generalize the code
tempInd = 0
for nMode_ in [0,4]:
    true = np.asarray([rf['n'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
    rtrv = np.asarray([rf['n'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,tempInd]
    minAOD = np.min(true)*0.95
    maxAOD = np.max(true)*1.05
    if nMode_ == 0:
        density_scatter(true, rtrv, ax=ax[1,1])
        plotProp(ax[1,1], titleStr=r'n$_{fine}$', scale=None)
        # histogram
        modifiedHist((rtrv-true), axs=ax_hist[1,1], titleStr=r'n$_{fine}$',
                     fig=fig_hist, nBins=nBins)
    else:
        density_scatter(true, rtrv, ax=ax[1,2])
        plotProp(ax[1,2], titleStr=r'n$_{coarse}$', scale=None)
        # histogram
        modifiedHist((rtrv-true), axs=ax_hist[1,2], titleStr=r'n$_{fine}$',
                     fig=fig_hist, nBins=nBins)
    tempInd += 1

# =============================================================================
# # %% intensity
# =============================================================================
intensity = False
if intensity:
    true = np.sum([rb['meas_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
    rtrv = np.sum([rb['fit_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
    minAOD = np.min(true)*0.95
    maxAOD = np.max(true)*1.05

    density_scatter(true, rtrv, ax=ax[1,4], xlabel=xlabel)
    plotProp(ax[2,4], titleStr='sum(intensity)', scale='linear')

else:
    true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd]
    tempInd = 0
    for nMode_ in [0,4]:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
        rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,tempInd]
        minAOD = np.min(rtrv)*0.90
        maxAOD = np.max(rtrv)*1.10
        if nMode_ == 0:
            true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]*4
            density_scatter(true*8, rtrv, ax=ax[2,3])
            plotProp(ax[2,3], titleStr='Fine mode \n Vol concentration',
                     scale='linear', xlabel=xlabel, MinMax=True)
            # histogram
            modifiedHist((rtrv-true), axs=ax_hist[2,3],
                         titleStr='Fine mode \n Vol concentration',
                         fig=fig_hist, nBins=nBins,
                         xlabel='Retrieved-Simulated')
        else:
            density_scatter(true*8, rtrv, ax=ax[2,2])
            plotProp(ax[2,2], titleStr='Coarse mode \n Vol concentration',
                     scale='linear', xlabel=xlabel, MinMax=True)
            # histogram
            modifiedHist((rtrv-true), axs=ax_hist[2,2],
                         titleStr='Fine mode \n Vol concentration',
                         fig=fig_hist, nBins=nBins,
                         xlabel='Retrieved-Simulated')
        tempInd += 1
# =============================================================================
# Empt axs
# =============================================================================
ax[2,4].axis('off')
ax_hist[2,0].axis('off')
ax_hist[2,4].axis('off')
# =============================================================================
# Save the figure
# =============================================================================
figSavePath = saveFN.replace('.pkl',('_%s_%s_%04dnm.png' % (surf2plot, fnTag, simBase.rsltFwd[0]['lambda'][waveInd]*1000)))
print('Saving figure to: %s' % figSavePath)
ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (saveFN, simBase.rsltFwd[0]['lambda'][waveInd], surf2plot, aodMin)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.tight_layout()
fig.suptitle(ttlStr.replace('MERGED_',''))
fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig_hist.tight_layout()
fig_hist.suptitle(ttlStr.replace('MERGED_','Hist_'))
fig.savefig(inDirPath + figSavePath, dpi=330)
fig_hist.savefig(inDirPath + figSavePath.replace('MERGED_','Hist_'), dpi=330)
print('Saving figure to: %s' % figSavePath.replace('MERGED_','Hist_'))
# plt.show()

