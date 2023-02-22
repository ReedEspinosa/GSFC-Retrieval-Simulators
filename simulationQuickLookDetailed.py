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
mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'});mpl.rcParams.update({'ytick.right': 'True'});mpl.rcParams.update({'xtick.top': 'True'});plt.rcParams["font.family"] = "Latin Modern Math"; plt.rcParams["mathtext.fontset"] = "cm"
from scipy import interpolate
from scipy.stats import norm, gaussian_kde, ncx2, moyal
from simulateRetrieval import simulation
from glob import glob
try:
    import mpl_scatter_density # adds projection='scatter_density'
    mpl_scatter = True
except:
    print('mpl_scatter_density library not available')
    mpl_scatter = False
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# Initiation
# =============================================================================


# coarseModeFwd = 4 # ANIN
coarseModeFwd = 4

# Define the string pattern for the file name
fnPtrnList = []
#fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_*z.pkl'
# fnPtrn = 'megaharp01_CAMP2Ex_2modes_AOD_*_550nm_addCoarse__campex_flight#*_layer#00.pkl'
fnPtrn = 'Camp2ex_AOD_*_550nm_*_campex_tria_flight#*_layer#00.pkl'
fnPtrn = 'DRS_V01_Lidar050+polar07_case08a1_tFct1.00_orbSS_multiAngles_n30_nAngALL.pkl'
# fnPtrn = 'Camp2ex_AOD_*_550nm_SZA_30*_PHI_*_campex_flight#*_layer#00.pkl'
# fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_1000z.pkl'

# Location/dir where the pkl files are
inDirPath = '/home/aputhukkudy/ACCDAM/2022/Campex_Simulations/Dec2022/04/fullGeometry/withCoarseMode/ocean/2modes/megaharp01/'
inDirPath = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults/'

# pklPathPatern = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults/DRS_V01_Lidar050_case08m1_tFct1.00_orbSS_multiAngles_n204_nAngALL.pkl'


# Anin's CAMP2Ex Settings
# waveInd = 2 # Wavelength index for plotting
# waveInd2 = 4 # Wavelength index for AE calculation
# fineFwdInd = 0 # index in forward data to use for fine mode plots 
# fineBckInd = 0 # index in backward data to use for fine mode plots
# crsFwdInd = 4 # index in forward data to use for coarse mode plots
# crsBckInd = 1 # index in backward data to use for coarse mode plots
# fineFwdScale = 4 # hack for CAMP2Ex data where fine mode is spread over 4 fwd modes 

# Reed's ABI Settings
waveInd = 2 # Wavelength index for plotting
waveInd2 = 4 # Wavelength index for AE calculation
fineFwdInd = 0 # index in forward data to use for fine mode plots 
fineBckInd = 0 # index in backward data to use for fine mode plots
crsFwdInd = 3 # index in forward data to use for coarse mode plots
crsBckInd = 1 # index in backward data to use for coarse mode plots
fineFwdScale = 1 # should be unity when fwd/back modes pair one-to-one


# more tags and specifiations for the scatter plot
surf2plot = 'both' # land, ocean or both
aodMin = 0.02 # does not apply to first AOD plot
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
nBins2 = 50 # no. of bins for 2D density plot


# =============================================================================
# Definition to plot the 2D histogram
# =============================================================================

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-4, '#440053'),
    (0.2, '#404388'),
    (0.35, '#2a788e'),
    (0.4, '#21a784'),
    (0.45, '#78d151'),
    (1, '#fde624'),
], N=512)

def using_mpl_scatter_density(x, y, ax, fig=None):
    density = ax.scatter_density(x, y,
                                 cmap=white_viridis, dpi=60)
    # ax.colorbar(density, label='Number of points per pixel')
    if fig:
        fig.colorbar(density, label='Frequency')
    
def density_scatter(x, y, ax=None, fig=None, sort=True, bins=20,
                    mplscatter=True, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    if not mplscatter:
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpolate.interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                    data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
        # Calculate the point density
        # xy = np.vstack([x_e, y_e])
        # z = gaussian_kde(xy)(xy)

        # To be sure to plot all data
        # z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, s=5, alpha=0.1, **kwargs)

        norm = mpl.colors.Normalize(vmin=np.max([0, np.min(z)]), vmax=np.max(z))
        if fig:
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm), ax=ax)
            cbar.ax.set_ylabel('Density')
    else:
        # use mpl_scatter_density library    
        if fig:
            using_mpl_scatter_density( x, y, ax, fig)
        else:
            using_mpl_scatter_density( x, y, ax)
    return ax

def plotProp(axs, titleStr ='', scale='linear', ylabel=None,
             xlabel=None, stat=True, MinMax=True, moreDigits=False):
    
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
        if not moreDigits:
            frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
        else:
            frmt = 'R=%5.3f\nRMS=%5.4f\nbias=%5.4f'
        tHnd = axs.annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)
        textstr = frmt % (Rcoef, RMSE, bias)
        tHnd = axs.annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                            textcoords='offset points', color=clrText, fontsize=FS)
    return tHnd

def modifiedHist(x, axs, fig=None, titleStr ='', xscale=None, ylabel=None,
             xlabel=None, stat=True, MinMax=False, nBins=20, err=0.05,
             yscale=None, fit='norm'):

    # Creating histogram
    N, bins, patches = axs.hist(x, bins = nBins, density=False)
     
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
        
        # Plot the PDF.
        # x_ = [abs(np.percentile(x,1)), 
        #       abs(np.percentile(x,99))]
        # # x_ = [abs(np.min(x)), 
        # #       abs(np.max(x))]       
        # xmax = np.max(x_)
        # xmin = -xmax
        # x2 = np.linspace(xmin, xmax, 100)
        # if fit=='norm':
        #     mu, std = norm.fit(x)
        #     p = norm.pdf(x2, mu, std)
            
        # elif fit=='ncx2':
        #     mu, std = ncx2.fit(x)
        #     p = ncx2.pdf(x2, mu, std)
        # elif fit=='moyal':
        #     mu, std = moyal.fit(x)
        #     p = moyal.pdf(x2, mu, std)
        
        # axs.plot(x2, p, 'C03', linewidth=1.5)
        
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        frmt = 'RMS=%5.3f\nbias=%5.3f'
        textstr = frmt % (RMSE, bias)
        tHnd = axs.annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top',
                            xycoords='axes fraction',
                            textcoords='offset points', color=clrText,
                            fontsize=FS)
    
    
# =============================================================================
# Loading and filtering data and prepping plots
# =============================================================================

# define the fig and axes handles
# Figure for 2D density plots
# fig, ax = plt.subplots(3,5, figsize=(15,9), subplot_kw={'projection': 'scatter_density'})
fig, ax = plt.subplots(3,5, figsize=(15,9))
plt.locator_params(nbins=3)
# Figure for histograms
fig_hist, ax_hist = plt.subplots(3,5, figsize=(15,9))
plt.locator_params(nbins=3)
lightSave = True # Omit PM elements and extinction profiles from MERGED files to save space

# Define the path of the new merged pkl file
loadPATH = os.path.join(inDirPath,fnPtrn)
simBase = simulation(picklePath=loadPATH)

# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][waveInd])
pprint(simBase.analyzeSim(waveInd)[0])

# lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
# TODO: this has to be generalized so that it can be ran without crashing the code
# at the moment, this hard coded to be ocean pixels
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
# Plotting
# =============================================================================

# =============================================================================
# AOD
# =============================================================================
true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]

minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
density_scatter(true, rtrv, ax=ax[0,0], bins=logBins, fig=fig, mplscatter=mpl_scatter)
plotProp(ax[0,0], titleStr='AOD', scale='log', ylabel='Retrieved')

# histogram
nBins_ = np.linspace(-0.05,0.05, nBins)
modifiedHist((rtrv-true), axs=ax_hist[0,0], titleStr='AOD', ylabel= 'Frequency',
             fig=fig_hist, nBins=nBins_)

# =============================================================================
# Coarse mode AOD
# =============================================================================
true = np.asarray([rf['aodMode'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,crsFwdInd]
rtrv = np.asarray([rf['aodMode'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,crsBckInd]

minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
density_scatter(true, rtrv, ax=ax[1,0], bins=logBins, mplscatter=mpl_scatter)
plotProp(ax[1,0], titleStr='Coarse mode AOD', scale='log', ylabel='Retrieved')

# histogram
nBins_ = np.linspace(-0.025,0.025, nBins)
modifiedHist((rtrv-true), axs=ax_hist[2,4], titleStr='Coarse mode AOD',
             fig=fig_hist, nBins=nBins_, xlabel='Retrieved-Simulated')
# =============================================================================
# # AAOD – [0,2]
# =============================================================================
true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepIndAll]
rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepIndAll]
minAOD = np.min(true)*0.9
maxAOD = np.max(true)*1.1
logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
density_scatter(true, rtrv, ax=ax[0,2], bins=logBins, mplscatter=mpl_scatter)
plotProp(ax[0,2], titleStr='AAOD', scale='log')

# histogram
nBins_ = np.linspace(-0.05,0.05, nBins)
modifiedHist((rtrv-true), axs=ax_hist[0,2], titleStr='AAOD',
             fig=fig_hist, nBins=nBins_)

# apply AOD min after we plot AOD
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>=aodMin for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]<=aodMax for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≤%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMax))
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

minAOD = np.percentile(rtrv,1) # BUG: Why is Angstrom >50 in at least one OSSE cases?
maxAOD = np.percentile(rtrv,99)
density_scatter(true, rtrv, ax=ax[0,1], mplscatter=mpl_scatter)
plotProp(ax[0,1], titleStr='Angstrom Exponent', MinMax=True)

# histogram
nBins_ = np.linspace(-0.5,0.5, nBins)
modifiedHist((rtrv-true), axs=ax_hist[0,1], titleStr='Angstrom Exponent',
             fig=fig_hist, nBins=nBins_)

# =============================================================================
# # k (fine) – [1,3]
# =============================================================================

# TODO: We shouldn't need this hack to handle single mode cases... they should be treated by runGRASP as Npix x 1 size arrays (ndim==2)
# aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
# true = np.asarray([rf['k'][waveInd] for rf in simBase.rsltFwd])[keepInd]
# rtrv = np.asarray([aodWght(rf['k'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
# Modifying the true value based on the NDIM
# if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
# and the coarse mode 'sea salt' have different value. So based on the dimension of the var
# We can distinguish each run type and generalize the code

true = np.asarray([rf['k'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,fineFwdInd]
rtrv = np.asarray([rf['k'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,fineBckInd]
minAOD = 0.0005
maxAOD = np.max(true)*1.15
logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
density_scatter(true, rtrv, ax=ax[1,3], bins=logBins, mplscatter=mpl_scatter)
plotProp(ax[1,3], titleStr=r'k$_{fine}$', scale='log')
ax[1,3].set_xticks([0.001, 0.01])
ax[1,3].set_yticks([0.001, 0.01])
# histogram
modifiedHist((rtrv-true), axs=ax_hist[1,3], titleStr=r'k$_{fine}$',
             fig=fig_hist, nBins=nBins)


# =============================================================================
# # k (coarse) – [1,4]
# =============================================================================

true = np.asarray([rf['k'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,crsFwdInd]
rtrv = np.asarray([rf['k'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,crsBckInd]
minAOD = 0.0001
maxAOD = 0.001
logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
density_scatter(true, rtrv, ax=ax[1,4], bins=logBins, mplscatter=mpl_scatter)
plotProp(ax[1,4], titleStr=r'k$_{coarse}$', scale='log', moreDigits=True)
ax[1,4].set_xticks([0.0001, 0.001])
ax[1,4].set_yticks([0.0001, 0.001])
# histogram
modifiedHist((rtrv-true), axs=ax_hist[1,4], titleStr=r'k$_{coarse}$',
             fig=fig_hist, nBins=nBins)


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
    print('Error in plotting FMF: \n error: %s \n Retrying with keepIndAll (SHOULD LOOK INTO THIS)' %err)
    
    # try plotting bland altman
    true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepIndAll]
    rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepIndAll]
    rtrv = true - rtrv
    minAOD = np.min(true)*0.9
    maxAOD = np.max(true)*1.1
    ax[2,4].plot([minAOD,maxAOD], [0,0], 'k', linewidth=LW121)
    ax[2,4].set_title('difference in AOD')
    ax[2,4].set_ylabel('true-retrieved')
    ax[2,4].set_xlabel('true')
    ax[2,4].set_xscale('log')
    logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
    density_scatter(true, rtrv, ax=ax[2,4], bins=logBins, mplscatter=mpl_scatter)
    ax[2,4].set_xlim(minAOD,maxAOD)
    ax[2,4].set_ylim(-0.3,0.3)

    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    tHnd = ax[2,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
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

density_scatter(true, rtrv, ax=ax[2,0], mplscatter=mpl_scatter)
plotProp(ax[2,0], titleStr=r'SSA', scale='linear', ylabel='Retrieved', xlabel=xlabel)
# histogram
nBins_ = np.linspace(-0.1,0.1, nBins)
modifiedHist((rtrv-true), axs=ax_hist[1,0], titleStr='SSA',
             fig=fig_hist, nBins=nBins_, ylabel= 'Frequency')
    
# =============================================================================
# # spherical fraction
# =============================================================================

true = np.asarray([rf['sph']for rf in simBase.rsltFwd])[keepInd][:,fineFwdInd]
rtrv = np.asarray([rb['sph']for rb in simBase.rsltBck])[keepInd][:,fineBckInd]
minAOD = 0
maxAOD = 100.1

ax[0,3].set_xticks(np.arange(minAOD, maxAOD, 25))
ax[0,3].set_yticks(np.arange(minAOD, maxAOD, 25))

density_scatter(true, rtrv, ax=ax[0,3], mplscatter=mpl_scatter)
plotProp(ax[0,3], titleStr='fine mode spherical vol. frac.')

# histogram
modifiedHist((rtrv-true), axs=ax_hist[0,3], titleStr='Fine mode SF',
             fig=fig_hist, nBins=nBins)

# coarse mode SF
true = np.asarray([rf['sph']for rf in simBase.rsltFwd])[keepInd][:,crsFwdInd]
rtrv = np.asarray([rb['sph']for rb in simBase.rsltBck])[keepInd][:,crsBckInd]
modifiedHist((rtrv-true), axs=ax_hist[0,4], titleStr='coarse mode SF',
             fig=fig_hist, nBins=nBins)

# =============================================================================
# # rEff
# =============================================================================
#simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(rtrv)*0.8
maxAOD = np.max(rtrv)*1.2

density_scatter(true, rtrv, ax=ax[0,4], mplscatter=mpl_scatter)
plotProp(ax[0,4], titleStr=r'Submicron r$_{eff}$', scale=None)
# histogram
modifiedHist((rtrv-true), axs=ax_hist[2,0], titleStr=r'Submicron r$_{eff}$',
             fig=fig_hist, nBins=nBins, xlabel='Retrieved-Simulated', ylabel='Frequency')

# Corase mode rEff
true = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(rtrv)*0.8
maxAOD = np.max(rtrv)*1.2

density_scatter(true, rtrv, ax=ax[2,1], mplscatter=mpl_scatter)
plotProp(ax[2,1], titleStr=r'above micron r$_{eff}$', scale=None, xlabel=xlabel)
# histogram
nBins_ = np.linspace(-0.5,0.5, nBins)
modifiedHist((rtrv-true), axs=ax_hist[2,1], titleStr=r'above micron r$_{eff}$',
             fig=fig_hist, nBins=nBins_, xlabel='Retrieved-Simulated')

# =============================================================================
# # n (fine)
# =============================================================================


true = np.asarray([rf['n'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,fineFwdInd]
rtrv = np.asarray([rf['n'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,fineBckInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
density_scatter(true, rtrv, ax=ax[1,1], mplscatter=mpl_scatter)
plotProp(ax[1,1], titleStr=r'n$_{fine}$', scale=None)
# histogram
modifiedHist((rtrv-true), axs=ax_hist[1,1], titleStr=r'n$_{fine}$',
             fig=fig_hist, nBins=nBins)


# =============================================================================
# # n (coarse)
# =============================================================================

true = np.asarray([rf['n'][:,waveInd] for rf in simBase.rsltFwd])[keepInd][:,crsFwdInd]
rtrv = np.asarray([rf['n'][:,waveInd] for rf in simBase.rsltBck])[keepInd][:,crsBckInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
density_scatter(true, rtrv, ax=ax[1,2], mplscatter=mpl_scatter)
plotProp(ax[1,2], titleStr=r'n$_{coarse}$', scale=None)
# histogram
modifiedHist((rtrv-true), axs=ax_hist[1,2], titleStr=r'n$_{coarse}$',
             fig=fig_hist, nBins=nBins)


# =============================================================================
# # %% intensity (if on, overwrites k)
# =============================================================================
intensity = False
if intensity:
    true = np.sum([rb['meas_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
    rtrv = np.sum([rb['fit_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
    minAOD = np.min(true)*0.95
    maxAOD = np.max(true)*1.05

    density_scatter(true, rtrv, ax=ax[1,4], xlabel=xlabel, mplscatter=mpl_scatter)
    plotProp(ax[2,4], titleStr='sum(intensity)', scale='linear')



# =============================================================================
# # volume conc (fine)
# =============================================================================

true = fineFwdScale*np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,fineFwdInd]
rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,fineBckInd]
minAOD = np.min(rtrv)*0.90
maxAOD = np.max(rtrv)*1.10

density_scatter(true, rtrv, ax=ax[2,3], mplscatter=mpl_scatter)
plotProp(ax[2,3], titleStr='Fine mode \n Vol concentration',
         scale='linear', xlabel=xlabel, MinMax=True)
# histogram
nBins_ = np.linspace(-0.10,0.10, nBins)
modifiedHist((rtrv-true), axs=ax_hist[2,3],
             titleStr='Fine mode \n Vol concentration',
             fig=fig_hist, nBins=nBins_,
             xlabel='Retrieved-Simulated')


# =============================================================================
# # volume conc (coarse)
# =============================================================================

true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,crsFwdInd]
rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,crsBckInd]
minAOD = np.min(rtrv)*0.90
maxAOD = np.max(rtrv)*1.10

density_scatter(true, rtrv, ax=ax[2,2], mplscatter=mpl_scatter)
plotProp(ax[2,2], titleStr='Coarse mode \n Vol concentration',
         scale='linear', xlabel=xlabel, MinMax=True)
# histogram
nBins_ = np.linspace(-0.25,0.25, nBins)
modifiedHist((rtrv-true), axs=ax_hist[2,2],
             titleStr='Coarse mode \n Vol concentration',
             fig=fig_hist, nBins=nBins_,
             xlabel='Retrieved-Simulated')
                         

# =============================================================================
# Save the figure
# =============================================================================
saveFN = os.path.basename(loadPATH)
figSavePath = saveFN.replace('.pkl',('_%s_%s_%04dnm.png' % (surf2plot, fnTag, simBase.rsltFwd[0]['lambda'][waveInd]*1000)))
print('Saving figure to: %s' % (os.path.join(inDirPath,figSavePath)))
ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (saveFN, simBase.rsltFwd[0]['lambda'][waveInd], surf2plot, aodMin)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.tight_layout()
fig.suptitle(ttlStr.replace('MERGED_',''))
fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig_hist.tight_layout()
fig_hist.suptitle(ttlStr.replace('MERGED_','Hist_'))
fig.savefig(inDirPath + figSavePath, dpi=330)
fig_hist.savefig(inDirPath + figSavePath.replace('MERGED_','Hist_'), dpi=330)
print('Saving figure to: %s' % (os.path.join(inDirPath,figSavePath.replace('MERGED_','Hist_'))))
# plt.show()

