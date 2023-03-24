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
mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'}); mpl.rcParams.update({'ytick.right': 'True'}); mpl.rcParams.update({'xtick.top': 'True'}); plt.rcParams["font.family"] = "Latin Modern Math"; plt.rcParams["mathtext.fontset"] = "cm"
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
# Initiation and User Provided Settings
# =============================================================================

### Reed's ABI/testing Settings###
waveInd = 2 # Wavelength index for plotting
waveInd2 = 4 # Wavelength index for AE calculation
fineFwdInd = 0 # index in forward data to use for fine mode plots 
fineBckInd = 0 # index in backward data to use for fine mode plots
crsFwdInd = 3 # index in forward data to use for coarse mode plots
crsBckInd = 1 # index in backward data to use for coarse mode plots
fineFwdScale = 1 # should be unity when fwd/back modes pair one-to-one
inDirPath = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults/' # Location/dir where the pkl files are
fnPtrn = 'DRS_V01_Lidar050+polar07_case08a1_tFct1.00_orbSS_multiAngles_n30_nAngALL.pkl'


### Anin's CAMP2Ex Settings ###
# Location/dir where the pkl files are
# inDirPath = '/home/aputhukkudy/ACCDAM/2022/Campex_Simulations/Dec2022/04/fullGeometry/withCoarseMode/ocean/2modes/megaharp01/'
# fnPtrn = 'megaharp01_CAMP2Ex_2modes_AOD_*_550nm_addCoarse__campex_flight#*_layer#00.pkl'
# fnPtrn = 'Camp2ex_AOD_*_550nm_*_campex_tria_flight#*_layer#00.pkl'
# fnPtrn = 'Camp2ex_AOD_*_550nm_SZA_30*_PHI_*_campex_flight#*_layer#00.pkl'
# waveInd = 2 # Wavelength index for plotting
# waveInd2 = 4 # Wavelength index for AE calculation
# fineFwdInd = 0 # index in forward data to use for fine mode plots 
# fineBckInd = 0 # index in backward data to use for fine mode plots
# crsFwdInd = 4 # index in forward data to use for coarse mode plots
# crsBckInd = 1 # index in backward data to use for coarse mode plots
# fineFwdScale = 4 # hack for CAMP2Ex data where fine mode is spread over 4 fwd modes 



# more tags and specifiations for the scatter plot
surf2plot = 'both' # land, ocean or both
aodMin = 0.02 # does not apply to first AOD plot
aodMax = 2 # Pixels with AOD above this will be filtered from plots of intensive parameters
fnTag = 'AllCases'
xlabel = 'Simulated Truth'
FS = 10 # Plot font size
LW121 = 1 # line width of the one-to-one line
clrText = '#FF6347' # color of statistics text
nBins = 200 # no. of bins for histogram of differences plots
nBins2 = 50 # no. of bins for 2D density plot

"""
# TODO:
<DONE> 1 - Make figure number of subplots automatic
<DONE> 2 - Link axis control to vars2plot
<DONE> 3  - Can function definitions below be cleaned up?
<DONE> 4  - Can some common code under each vars elif statement be refactored into common function?
<DONE> 5  - Are all the inputs directly above here needed?
6  - Test against open version in other terminal tab... that ship has sailed, but we can look at old commits
<DONE> 7  - How does Anin have runs without land_prct in either fwd or bck?
8  - Printing of N pixels passing convergence filter is confusing; should just happen once
9  - Why is this so slow?
<DONE> 10 - The definition of histogram bins nBins_ are hardcoded, and sometimes are outside the range of data...
11 – There should be an option in genPlots for forced [max, min] (e.g., [0,100] for SPH)
"""

vars2plot = [
    'aod',
    'aod_f',
    'angstrom',
    'aaod',
    'sph_f',
    'fmf',
    'reff_sub_um',
    'sph_f',
    'sph_c',
    'aod_c',
    'g',
    'n_f',
    'n_c',
    'k_f',
    'k_c',
    'intensity',
    'ssa',
    'reff_abv_um',
    'vol_c',
    'vol_f',
    'blandAltman',
]

nRows = int(np.sqrt(len(vars2plot)))
nCols = int(np.ceil(len(vars2plot)/nRows))
axesInd = [[i,j] for i in range(nRows) for j in range(nCols)]


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

def fmfCalc(r,dvdlnr):
    assert np.all(r[0]==r[-1]), 'First and last mode defined with different radii!' # This is not perfect, but likely to catch non-standardized PSDs
    if r.ndim==2: r=r[0] # We hope all modes defined over same radii (partial check above)
    dvdlnr = dvdlnr.sum(axis=0)  # Loading checks are in place in runGRASP.py to guarantee 2D arrays of absolute dvdlnr
    cutRadius = 0.5
    fInd = r<=cutRadius
    logr = np.log(r)
    return np.trapz(dvdlnr[fInd],logr[fInd])/np.trapz(dvdlnr,logr)
    
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
        ax.scatter_density(x, y, cmap=white_viridis, dpi=60)
        density = using_mpl_scatter_density( x, y, ax, figFnd)
        if fig: fig.colorbar(density, label='Density')
    return ax


def plotProp(true, rtrv, axs, titleStr ='', scale='linear', xlabel=False, ylabel=False, MinMax=None, stat=True, moreDigits=False):
    """
    Formatting and statistics for scatter plots
    """
    # min max
    if MinMax is not None:
        axs.plot(MinMax, MinMax, 'k', linewidth=LW121) # line plot
        axs.set_xlim(MinMax[0],MinMax[1])
        axs.set_ylim(MinMax[0],MinMax[1])
    # Title of the plot
    axs.set_title(titleStr)
    # x and y label
    if xlabel: axs.set_xlabel('Truth')
    if ylabel: axs.set_ylabel('Retrieved')
    # scale
    axs.set_xscale(scale)
    axs.set_yscale(scale)
    if scale=='linear': axs.ticklabel_format(axis='both', style='plain', useOffset=False)
        
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


def modifiedHist(x, axs, titleStr='', xlabel=False, ylabel=False, nBins=20, stat=True):
    # Creating histogram
    N, bins, patches = axs.hist(x, bins=nBins, density=False)
     
    # Setting colors
    assert not np.isclose(N.max(), 0), 'N.max() was zero... No values exist within the provided bins.' 
    fracs = ((N**(1 / 2)) / N.max())
    norm_ = mpl.colors.Normalize(fracs.min(), fracs.max())
     
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm_(thisfrac))
        thispatch.set_facecolor(color)
    # Title of the plot
    axs.set_title(titleStr)    
    # x and y label
    if xlabel: axs.set_xlabel('Retrieved-Simulated')
    if ylabel: axs.set_ylabel('Frequency')

    # mean and standard deviation
    if stat:        
        RMSE = np.sqrt(np.median(diff**2))
        bias = np.mean(diff)
        frmt = 'RMS=%5.3f\nbias=%5.3f'
        textstr = frmt % (RMSE, bias)
        tHnd = axs.annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top',
                            xycoords='axes fraction',
                            textcoords='offset points', color=clrText,
                            fontsize=FS)
    
    
def genPlots(true, rtrv, axScat, axHist, varName, xlabel, ylabel, scale='linear', moreDig=False, stats=False):
    minVal = np.percentile(true, 1)
    maxVal = np.percentile(true, 99)
    logMin = min(0.0001, np.log10(minVal))
    logBins = np.logspace(start=logMin, stop=np.log10(maxVal), num=nBins2)
    density_scatter(true, rtrv, ax=axScat, mplscatter=mpl_scatter)
    plotProp(true, rtrv, axScat, varName, scale, xlabel, ylabel, [minVal,maxVal], moreDigits=moreDig, stat=stats)
    
    if not stats: return # If stats do not make sense, a histogram probably will not either (e.g., for Bland Altman)
    # histogram
    diff = rtrv-true
    maxDiff = np.percentile(np.abs(diff), 99)
    nBins_ = np.linspace(-maxDiff,maxDiff, nBins)
    modifiedHist(diff, axHist, varName, xlabel, ylabel, nBins_)


# =============================================================================
# Loading and filtering data and prepping plots
# =============================================================================
# Figure for 2D density plots
# fig, ax = plt.subplots(3,5, figsize=(15,9), subplot_kw={'projection': 'scatter_density'})
fig, ax = plt.subplots(nRows, nCols, figsize=(15,9))
plt.locator_params(nbins=3)
# Figure for histograms
fig_hist, ax_hist = plt.subplots(nRows, nCols, figsize=(15,9))
plt.locator_params(nbins=3)

# Define the path of the new merged pkl file
loadPATH = os.path.join(inDirPath,fnPtrn)
simBase = simulation(picklePath=loadPATH)

# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][waveInd])
pprint(simBase.analyzeSim(waveInd)[0])

if 'land_prct' in simBase.rsltBck[0]:
    lp = np.array([rb['land_prct'] for rb in simBase.rsltBck])
    keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1
else:
    print('WARNING: land_prct key not found, using all pixels!')
    keepInd = np.ones(len(simBase.rsltBck), dtype='int32')

# apply convergence filter
simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 95)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])
keepIndAll = keepInd
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

# apply AOD min after we plot AOD
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>=aodMin for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]<=aodMax for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≤%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMax))

# apply Reff min
simBase._addReffMode(1.0, True) # reframe with cut at 1 micron diameter
#keepInd = np.logical_and(keepInd, [rf['rEffMode']>=2.0 for rf in simBase.rsltBck])
#print('%d/%d fit surface type %s and aod≥%4.2f AND retrieved Reff>2.0μm' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))


# =============================================================================
# Plotting
# =============================================================================

for var,axInd in zip(vars2plot, axesInd[0:len(vars2plot)]):
    axScat = ax[tuple(axInd)]
    axHist = ax_hist[tuple(axInd)]
    xlabel = axInd[0]==(nRows-1)
    ylabel = axInd[1]==0
    if var=='aod':     # AOD
        true = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltFwd])[keepIndAll]
        rtrv = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltBck])[keepIndAll]
        genPlots(true, rtrv, axScat, axHist, 'AOD', xlabel, ylabel, scale='log')
    elif var=='aod_f':    # Fine mode AOD
        true = np.asarray([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltFwd])[keepIndAll][:,fineFwdInd]
        rtrv = np.asarray([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltBck])[keepIndAll][:,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine mode AOD', xlabel, ylabel, scale='log')        
    elif var=='aod_c':    # Coarse mode AOD
        true = np.asarray([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltFwd])[keepIndAll][:,crsFwdInd]
        rtrv = np.asarray([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltBck])[keepIndAll][:,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Coarse mode AOD', xlabel, ylabel, scale='log')
    elif var=='aaod':     # # AAOD
        true = np.asarray([(1-rslt['ssa'][waveInd])*rslt['aod'][waveInd] for rslt in simBase.rsltFwd])[keepIndAll]
        rtrv = np.asarray([(1-rslt['ssa'][waveInd])*rslt['aod'][waveInd] for rslt in simBase.rsltBck])[keepIndAll]
        genPlots(true, rtrv, axScat, axHist, 'Absorbing AOD', xlabel, ylabel, scale='log')
    elif var=='angstrom':     # # ANGSTROM
        aod1 = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltFwd])[keepInd]
        aod2 = np.asarray([rslt['aod'][waveInd2] for rslt in simBase.rsltFwd])[keepInd]
        logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
        true = -np.log(aod1/aod2)/logLamdRatio
        aod1 = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltBck])[keepInd]
        aod2 = np.asarray([rslt['aod'][waveInd2] for rslt in simBase.rsltBck])[keepInd]
        rtrv = -np.log(aod1/aod2)/logLamdRatio
        genPlots(true, rtrv, axScat, axHist, 'Angstrom Exponent', xlabel, ylabel)
    elif var=='k_f':     # # k (fine)
        true = np.asarray([rslt['k'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd][:,fineFwdInd]
        rtrv = np.asarray([rslt['k'][:,waveInd] for rslt in simBase.rsltBck])[keepInd][:,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, r'k$_{fine}$', xlabel, ylabel, scale='log', moreDigits=True)
    elif var=='k_c':    # # k (coarse)
        true = np.asarray([rslt['k'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd][:,crsFwdInd]
        rtrv = np.asarray([rslt['k'][:,waveInd] for rslt in simBase.rsltBck])[keepInd][:,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, r'k$_{coarse}$', xlabel, ylabel, scale='log', moreDigits=True)
    elif var=='fmf':    # # FMF (vol)
        true = np.asarray([fmfCalc(rslt['r'], rslt['dVdlnr']) for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([fmfCalc(rslt['r'], rslt['dVdlnr']) for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine Mode Fraction', xlabel, ylabel)
    elif var=='g':    # # Asymmetry Parameter
        true = np.asarray([rslt['g'][waveInd] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['g'][waveInd] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, 'g', xlabel, ylabel)
    elif var=='blandAltman':     # # Bland Altman of AOD
        true = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltFwd])[keepIndAll]
        rtrv = true - np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltBck])[keepIndAll]
        genPlots(true, rtrv, axScat, axHist, 'Difference in AOD', xlabel, ylabel, scale='log', stats=False)
        minAOD = np.min(true)*0.9
        maxAOD = np.max(true)*1.1
        axNow.plot([minAOD,maxAOD], [0,0], 'k', linewidth=LW121)
        axNow.set_ylabel('true-retrieved')
        axNow.set_xlabel('true')
        logBins = np.logspace(start=np.log10(minAOD), stop=np.log10(maxAOD), num=nBins2)
        density_scatter(true, rtrv, ax=axNow, bins=logBins, mplscatter=mpl_scatter)
        axNow.set_xlim(minAOD,maxAOD)
        yRng = np.percentile(np.abs(rtrv), 99)
        axNow.set_ylim(-yRng, yRng)
    elif var=='ssa':     # # Single Scattering Albedo
        true = np.asarray([rslt['ssa'][waveInd] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['ssa'][waveInd] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, 'SSA', xlabel, ylabel)
    elif var=='sph_f':    # # spherical fraction (fine)
        true = np.asarray([rslt['sph']for rslt in simBase.rsltFwd])[keepInd][:,fineFwdInd]
        rtrv = np.asarray([rslt['sph']for rslt in simBase.rsltBck])[keepInd][:,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine Mode SPH', xlabel, ylabel)
    elif var=='sph_c':     # # spherical fraction (coarse)
        true = np.asarray([rslt['sph']for rslt in simBase.rsltFwd])[keepInd][:,crsFwdInd]
        rtrv = np.asarray([rslt['sph']for rslt in simBase.rsltBck])[keepInd][:,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Coarse Mode SPH', xlabel, ylabel)
    elif var=='reff_sub_um':     # # rEff (sub micron)
        true = np.asarray([rslt['rEffMode'][0] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['rEffMode'][0] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, r'Submicron r$_{eff}$', xlabel, ylabel)
    elif var=='reff_abv_um':     # # rEff (super micron)
        true = np.asarray([rslt['rEffMode'][1] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['rEffMode'][1] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, r'above micron r$_{eff}$', xlabel, ylabel)
    elif var=='n_f':     # # n (fine)
        true = np.asarray([rslt['n'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd][:,fineFwdInd]
        rtrv = np.asarray([rslt['n'][:,waveInd] for rslt in simBase.rsltBck])[keepInd][:,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, r'n$_{fine}$', xlabel, ylabel)
    elif var=='n_c':     # # n (coarse)
        true = np.asarray([rslt['n'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd][:,crsFwdInd]
        rtrv = np.asarray([rslt['n'][:,waveInd] for rslt in simBase.rsltBck])[keepInd][:,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, r'n$_{coarse}$', xlabel, ylabel)
    elif var=='intensity':     # # %% intensity
        true = np.sum([rslt['meas_I'][:,waveInd] for rslt in simBase.rsltBck[keepInd]], axis=1)
        rtrv = np.sum([rslt['fit_I'][:,waveInd] for rslt in simBase.rsltBck[keepInd]], axis=1)
        genPlots(true, rtrv, axScat, axHist, 'sum(intensity)', xlabel, ylabel, stats=False)
    elif var=='vol_f':     # # volume conc (fine)
        true = fineFwdScale*np.asarray([rslt['vol'] for rslt in simBase.rsltFwd])[keepInd][:,fineFwdInd]
        rtrv = np.asarray([rslt['vol'] for rslt in simBase.rsltBck])[keepInd][:,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine Mode Volume', xlabel, ylabel)
    elif var=='vol_c':     # # volume conc (coarse)
        true = np.asarray([rslt['vol'] for rslt in simBase.rsltFwd])[keepInd][:,crsFwdInd]
        rtrv = np.asarray([rslt['vol'] for rslt in simBase.rsltBck])[keepInd][:,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Coarse Mode Volume', xlabel, ylabel)
    else:
        assert False, 'Variable name: %s was not recognized!' % var                         


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

