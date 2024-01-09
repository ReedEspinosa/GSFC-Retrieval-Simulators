#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This script is used to quickly look at the results of a retrieval simulation, like that
output from the runSimulation.py script. It will plot the results of the simulation for a
given wavelength across all variables listed in the vars2plot Dictionary. 
'''

# =============================================================================
# Import the librarires
# =============================================================================
from os import path
import warnings
from pprint import pprint
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy import interpolate
from scipy.stats import norm, gaussian_kde, ncx2, moyal
from simulateRetrieval import simulation

# Modified version of mpl_scatter_density library to allow for custom colorbar
try:
    import mpl_scatter_density # adds projection='scatter_density'
    mpl_scatter = True
except:
    print('mpl_scatter_density library not available')
    mpl_scatter = False

# =============================================================================
# Initiation and User Provided Settings
# =============================================================================

### Reed's ABI Settings###
# waveInd = 2 # Wavelength index for plotting
# waveInd2 = 4 # Wavelength index for AE calculation
# fineFwdInd = 0 # index in forward data to use for fine mode plots 
# fineBckInd = 0 # index in backward data to use for fine mode plots
# crsFwdInd = 0 # index in forward data to use for coarse mode plots
# crsBckInd = 1 # index in backward data to use for coarse mode plots
# fineFwdScale = 1 # should be unity when fwd/back modes pair one-to-one
# pubQuality = True # If True, we use publication quality figures
# # filePathPtrn = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults/DRS_V01_Lidar050+polar07_case08a1_tFct1.00_orbSS_multiAngles_n30_nAngALL.pkl'
# filePathPtrn = '/Users/wrespino/Synced/Working/ABI_initialTests/Test_threeSites_Ocean_V02.pkl'


### Reed's PLRA Validation Settings###
waveInd = 2 # Wavelength index for plotting
waveInd2 = 4 # Wavelength index for AE calculation
fineFwdInd = 0 # index in forward data to use for fine mode plots 
fineBckInd = 0 # index in backward data to use for fine mode plots
crsFwdInd = 1 # index in forward data to use for coarse mode plots
crsBckInd = 1 # index in backward data to use for coarse mode plots
fineFwdScale = 1 # should be unity when fwd/back modes pair one-to-one
pubQuality = True # If True, we use publication quality figures
filePathPtrn = '/Users/wrespino/Synced/Working/NoahsAngleDependentError_Simulations/V1_Noah/Run-30_polarAOSnoah_case08*_tFctrandLogNrm*_n*_nAng0.pkl'
# filePathPtrn = '/Users/wrespino/Synced/AOS/Phase-A/PLRA_RequirementsAndTraceability/GSFC_ValidationSimulationsData/V0/Run-02_polarAOS_case08a_tFct0.000_n0_nAng0.pkl'

### Anin's CAMP2Ex Settings ###
# Location/dir where the pkl files are
# filePathPtrn = '/home/aputhukkudy/ACCDAM/2022/Campex_Simulations/Dec2022/04/fullGeometry/withCoarseMode/ocean/2modes/megaharp01/megaharp01_CAMP2Ex_2modes_AOD_*_550nm_addCoarse__campex_flight#*_layer#00.pkl'
# filePathPtrn = '/home/aputhukkudy/ACCDAM/2022/Campex_Simulations/Dec2022/04/fullGeometry/withCoarseMode/ocean/2modes/megaharp01/Camp2ex_AOD_*_550nm_*_campex_tria_flight#*_layer#00.pkl'
# filePathPtrn = '/home/aputhukkudy/ACCDAM/2022/Campex_Simulations/Dec2022/04/fullGeometry/withCoarseMode/ocean/2modes/megaharp01/Camp2ex_AOD_*_550nm_SZA_30*_PHI_*_campex_flight#*_layer#00.pkl'
# waveInd = 2 # Wavelength index for plotting
# waveInd2 = 4 # Wavelength index for AE calculation
# fineFwdInd = 0 # index in forward data to use for fine mode plots 
# fineBckInd = 0 # index in backward data to use for fine mode plots
# crsFwdInd = 4 # index in forward data to use for coarse mode plots
# crsBckInd = 1 # index in backward data to use for coarse mode plots
# fineFwdScale = 4 # hack for CAMP2Ex data where fine mode is spread over 4 fwd modes
# pubQuality = True # If True, we use publication quality figures

# more tags and specifiations for the scatter plot
surf2plot = 'both' # land, ocean or both
aodMin = 0.1 # does not apply to first AOD plot
aodMax = 5 # Pixels with AOD above this will be filtered from plots of intensive parameters
fnTag = 'AllCases'
xlabel = 'Simulated Truth'
FS = 10 # Plot font size
LW121 = 1 # line width of the one-to-one line
clrText = '#FF6347' # color of statistics text
nBins = 200 # no. of bins for histogram of differences plots
nBins2 = 50 # no. of bins for 2D density plot
showOverallStats = True # print RMSE of many GVs to terminal (may be slow for large Npixels)
recalcChi = False # If True, we recalculate chi^2 using difference between fwd and bck fit variables

# The variables to plot; will automatically remove variables for which rsltDict is missing
# vars2plot = { # Format is variable_name_in_this_script:main_relevant_rsltsDict_variable_key
#     'aod':'aod',
#     'aod_c':'aodMode',
#     'aod_f':'aodMode',
#     'angstrom':'aod',
#     'aaod':'ssa',
#     'fmf':'dVdlnr',
#     'sph_f':'sph',
#     'sph_c':'sph',
#     'g':'g',
#     'n_f':'n',
#     'n_c':'n',
#     'k_f':'k',
#     'k_c':'k',
#     'intensity':'meas_I',
#     'ssa':'ssa',
#     'reff_sub_um':'rEffMode',
#     'reff_abv_um':'rEffMode',
#     'vol_c':'vol',
#     'vol_f':'vol',
#     'blandAltman':'aod',
# }

vars2plot = { # Format is variable_name_in_this_script:main_relevant_rsltsDict_variable_key
    'aod':'aod',
    'angstrom':'aod',
    'aaod':'ssa',
    'g':'g',
    'intensity':'meas_I',
    'ssa':'ssa',
    'reff_sub_um':'rEffMode',
    'reff_abv_um':'rEffMode',
}


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

forceReff = True #Force rEff claculations of fine/coarse in both fwd and back
modeCuttOff = 0.5 # Separation between fine and coarse mode in μm of radius 



# =============================================================================
# Function Definitions for Plots
# =============================================================================

def fmfCalc(r, dvdlnr):
    '''Calculate fmf from r and dvdlnr
    
    Parameters
    ----------
    r : array-like, shape (N,)
        Radii in microns
    dvdlnr : array-like, shape (N,)
        dV/dlnr in m^-3 um^-1
    
    Returns
    -------
    fineModeFraction : float
        Fine mode fraction
    '''
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
    
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data
    ax : matplotlib.axes.Axes, optional
        Axes in which to draw the plot, otherwise use the current Axes.
    fig : matplotlib.figure.Figure, optional
        Figure in which to draw the plot, otherwise use the current Figure.
    sort : bool, optional
        Whether to sort the points by density, so that the densest points are
        plotted last
    bins : int, optional
        Number of bins in the 2d histogram
    mplscatter : bool, optional
        Whether to use plt.scatter or ax.scatter to draw the points
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot.
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

def plotProp(true, rtrv, axs, titleStr ='', scale='linear', xlabel=False, ylabel=False,
             MinMax=None, stat=True, moreDigits=False):
    """
    Formatting and statistics for scatter plots
    
    Parameters
    ----------
    true : array-like, shape (n, )
        True values
    rtrv : array-like, shape (n, )
        Retrieved values
    axs : matplotlib.axes.Axes
        Axes object with the plot.
    titleStr : str, optional
        Title of the plot
    scale : str, optional
        Scale of the plot, 'linear' or 'log'
    xlabel : bool, optional
    ylabel : bool, optional
    MinMax : tuple, optional
        Minimum and maximum values of the plot
    stat : bool, optional
        Whether to plot the statistics
    moreDigits : bool, optional
        Whether to use more digits in the statistics
        
    Returns
    -------
    None
    """
    # min max
    if MinMax is not None:
        assert MinMax[0]<MinMax[1], 'Plot minimum value was not less than maximum!'
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
        axs.text(0.6,0.1, 'N=%d' % len(true), transform=axs.transAxes, color=clrText, fontsize=FS) 
        Rcoef = np.nan if np.isclose(true[0], true).all() else np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        if not moreDigits:
            frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
        else:
            frmt = 'R=%5.3f\nRMS=%5.4f\nbias=%5.4f'
        textstr = frmt % (Rcoef, RMSE, bias)
        axs.text(0.07,0.65, textstr, transform=axs.transAxes, color=clrText, fontsize=FS)


def modifiedHist(x, axs, titleStr='', xlabel=False, ylabel=False, nBins=20, stat=True):
    '''Modified histogram plot
    
    Parameters
    ----------
    x : array-like, shape (n, )
        Input data
    axs : matplotlib.axes.Axes
        Axes in which to draw the plot
    titleStr : str, optional
        Title of the plot
    xlabel : bool, optional
    ylabel : bool, optional
    nBins : int, optional
        Number of bins in the histogram
    stat : bool, optional
        Whether to plot the statistics
        
    Returns
    -------
    none
    '''
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
        RMSE = np.sqrt(np.median(x**2))
        bias = np.mean(x)
        frmt = 'RMS=%5.3f\nbias=%5.3f'
        textstr = frmt % (RMSE, bias)
        tHnd = axs.annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top',
                            xycoords='axes fraction',
                            textcoords='offset points', color=clrText,
                            fontsize=FS)

    

def genPlots(true, rtrv, axScat, axHist, varName, xlabel, ylabel, scale='linear', moreDigits=False, stats=True, MinMax=None):
    '''Generate scatter and histogram plots
    
    Parameters
    ----------
    true : array-like, shape (n, )
        True values
    rtrv : array-like, shape (n, )
        Retrieved values
    axScat : matplotlib.axes.Axes
        Axes object with the scatter plot
    varName : str
        Name of the variable
    xlabel : bool
    ylabel : bool
    scale : str, optional
        Scale of the plot, 'linear' or 'log'
    moreDigits : bool, optional
        Whether to use more digits in the statistics
    stats : bool, optional
        Whether to plot the statistics
    MinMax : tuple, optional
        Minimum and maximum values of the plot
    
    Returns
    -------
    None
    '''
    nonNan = np.logical_and(~np.isnan(true), ~np.isnan(rtrv))
    if not nonNan.all():
        true = true[nonNan]
        rtrv = rtrv[nonNan]
        if len(true)<=1: 
            print('%s not plotted – all but ≤1 pixels were NAN! (At least two pixels needed to plot.)' % varName)
            return
        print('Removing %d NAN pixels from %s plots...' % (sum(~nonNan), varName))
    if scale=='log' and (np.any(true<=0) or np.any(rtrv<=0)):
        warnings.warn('Log scale set for %s with true and/or rtrv values that are less than or equal to zero!' % varName)
    if MinMax is None:
        minVal = np.percentile(np.r_[true, rtrv], 1)
        maxVal = np.percentile(np.r_[true, rtrv], 99)
        MinMax = [minVal,maxVal]
    density_scatter(true, rtrv, ax=axScat, mplscatter=mpl_scatter)
    plotProp(true, rtrv, axScat, varName, scale, xlabel, ylabel, MinMax, moreDigits=moreDigits, stat=stats)
    
    if not stats: return # If stats do not make sense, a histogram probably will not either (e.g., for Bland Altman)
    # histogram
    diff = rtrv-true
    maxDiff = np.percentile(np.abs(diff), 99)
    nBins_ = np.linspace(-maxDiff,maxDiff, nBins)
    modifiedHist(diff, axHist, varName, xlabel, ylabel, nBins_)


# =============================================================================
# Loading and filtering data and prepping plots
# =============================================================================

nRows = int(np.sqrt(len(vars2plot)))
nCols = int(np.ceil(len(vars2plot)/nRows))
axesInd = [[i,j] for i in range(nRows) for j in range(nCols)]

# Set the matplotlib parameters to use publication quality figures
if pubQuality:
    mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'})
    mpl.rcParams.update({'ytick.right': 'True'}); mpl.rcParams.update({'xtick.top': 'True'})
    plt.rcParams["font.family"] = "Latin Modern Math"; plt.rcParams["mathtext.fontset"] = "cm"

# Figure for 2D density plots
fig, ax = plt.subplots(nRows, nCols, figsize=(15,9))
plt.locator_params(nbins=3)

# Figure for histograms
fig_hist, ax_hist = plt.subplots(nRows, nCols, figsize=(15,9))
plt.locator_params(nbins=3)

# Define the path of the new merged pkl file
simBase = simulation(picklePath=filePathPtrn)
simBase.rsltFwd = np.asarray(simBase.rsltFwd)
simBase.rsltBck = np.asarray(simBase.rsltBck)
# # HACK: makes Test_threeSites_Ocean_V02.pkl work; just for testing
# for 
#     rs['meas_I_ocean'] = rs.pop('meas_ocean_I') # this is format populateFromRslt() can handle; should have just started with it in convert_YingxiFile2_to_rsltsPkl.py


# print general stats to console
fwdLambda = simBase.rsltFwd[0]['lambda'][waveInd]
bckLambda = simBase.rsltBck[0]['lambda'][waveInd]
print('Showing results for λ_fwd = %5.3f μm.' % bckLambda)

# check if the forward and backward lambdas are close
if not np.isclose(fwdLambda, bckLambda, atol=0.001):
    warnings.warn('\nThe values of lambda for the forward (%.3f μm) and backward (%.3f μm) differed by more than 1 nm at waveInd=%d!' % (fwdLambda, bckLambda, waveInd))
    print('Interpolating forward results to back wavelengths... (this should fix wavelength misalignment noted in prior warning)')
    simBase.spectralInterpFwdToBck()
if showOverallStats: 
    print('------ RMSE ------')
    pprint(simBase.analyzeSim(waveInd)[0])
    print('------------------')

# filter out pixels that are not on the surface of interest    
if 'land_prct' in simBase.rsltBck[0]:
    lp = np.array([rb['land_prct'] for rb in simBase.rsltBck])
    keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1
else:
    if surf2plot != 'both':
        warnings.warn('land_prct key not found, using all pixels!')
    keepInd = np.ones(len(simBase.rsltBck), dtype='bool')

# apply convergence filter
simBase.conerganceFilter(forceχ2Calc=recalcChi) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 95)
keepInd = np.logical_and(keepInd, [rb['costVal'] < costThresh for rb in simBase.rsltBck])
keepIndAll = keepInd
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

# apply AOD min after we plot AOD
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>=aodMin for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]<=aodMax for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≤%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMax))

# Calculate modal Reff above and below a micron in diameter
if np.any(['reff' in var.lower() for var in vars2plot.keys()]): 
    simBase._addReffMode(modeCuttOff, Force=forceReff) # reframe with cut at 0.5 micron radius

# Purge variables for which we do not have sufficient data in Fwd/Bck rsltsDicts
for key in list(vars2plot):
    inFwd = vars2plot[key] in simBase.rsltFwd[0].keys() 
    inBck = vars2plot[key] in simBase.rsltBck[0].keys()
    if not (inFwd and inBck):
        print('%s not found in fwd and/or bck rsltsDict. %s will not be plotted.' % (vars2plot[key],key))
        del vars2plot[key]

# =============================================================================
# Plotting
# =============================================================================

for var,axInd in zip(vars2plot.keys(), axesInd[0:len(vars2plot)]):
    axScat = ax[tuple(axInd)]
    axHist = ax_hist[tuple(axInd)]
    xlabel = axInd[0]==(nRows-1)
    ylabel = axInd[1]==0
    
    # based on the variable name, plot the appropriate data
    if var=='aod':     # AOD
        true = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltFwd])[keepIndAll]
        rtrv = np.asarray([rslt['aod'][waveInd] for rslt in simBase.rsltBck])[keepIndAll]
        genPlots(true, rtrv, axScat, axHist, 'AOD', xlabel, ylabel, scale='log')
    elif var=='aod_f':    # Fine mode AOD
        true = np.vstack([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltFwd])[keepIndAll,fineFwdInd]
        rtrv = np.vstack([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltBck])[keepIndAll,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine mode AOD', xlabel, ylabel, scale='log')        
    elif var=='aod_c':    # Coarse mode AOD
        true = np.vstack([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltFwd])[keepIndAll,crsFwdInd]
        rtrv = np.vstack([rslt['aodMode'][:,waveInd] for rslt in simBase.rsltBck])[keepIndAll,crsBckInd]
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
        true = np.vstack([rslt['k'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd,fineFwdInd]
        rtrv = np.vstack([rslt['k'][:,waveInd] for rslt in simBase.rsltBck])[keepInd,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, r'k$_{fine}$', xlabel, ylabel, scale='log', moreDigits=True)
    elif var=='k_c':    # # k (coarse)
        true = np.vstack([rslt['k'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd,crsFwdInd]
        rtrv = np.vstack([rslt['k'][:,waveInd] for rslt in simBase.rsltBck])[keepInd,crsBckInd]
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
        genPlots(true, rtrv, axScat, axHist, 'Difference in AOD', xlabel, ylabel, scale='linear', stats=False)
    elif var=='ssa':     # # Single Scattering Albedo
        true = np.asarray([rslt['ssa'][waveInd] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['ssa'][waveInd] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, 'SSA', xlabel, ylabel)
    elif var=='sph_f':    # # spherical fraction (fine)
        true = np.vstack([rslt['sph'] for rslt in simBase.rsltFwd])[keepInd,fineFwdInd]
        rtrv = np.vstack([rslt['sph'] for rslt in simBase.rsltBck])[keepInd,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine Mode SPH', xlabel, ylabel, MinMax=[0,100])
    elif var=='sph_c':     # # spherical fraction (coarse)
        true = np.vstack([rslt['sph'] for rslt in simBase.rsltFwd])[keepInd,crsFwdInd]
        rtrv = np.vstack([rslt['sph'] for rslt in simBase.rsltBck])[keepInd,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Coarse Mode SPH', xlabel, ylabel, MinMax=[0,100])
    elif var=='reff_sub_um':     # # rEff (sub micron)
        true = np.asarray([rslt['rEffMode'][0] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['rEffMode'][0] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, r'Submicron r$_{eff}$', xlabel, ylabel)
    elif var=='reff_abv_um':     # # rEff (super micron)
        true = np.asarray([rslt['rEffMode'][1] for rslt in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rslt['rEffMode'][1] for rslt in simBase.rsltBck])[keepInd]
        genPlots(true, rtrv, axScat, axHist, r'above micron r$_{eff}$', xlabel, ylabel)
    elif var=='n_f':     # # n (fine)
        true = np.vstack([rslt['n'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd,fineFwdInd]
        rtrv = np.vstack([rslt['n'][:,waveInd] for rslt in simBase.rsltBck])[keepInd,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, r'n$_{fine}$', xlabel, ylabel)
    elif var=='n_c':     # # n (coarse)
        true = np.vstack([rslt['n'][:,waveInd] for rslt in simBase.rsltFwd])[keepInd,crsFwdInd]
        rtrv = np.vstack([rslt['n'][:,waveInd] for rslt in simBase.rsltBck])[keepInd,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, r'n$_{coarse}$', xlabel, ylabel)
    elif var=='intensity':     # # %% intensity
        true = np.sum([rslt['meas_I'][:,waveInd] for rslt in simBase.rsltBck[keepInd]], axis=1)
        rtrv = np.sum([rslt['fit_I'][:,waveInd] for rslt in simBase.rsltBck[keepInd]], axis=1)
        genPlots(true, rtrv, axScat, axHist, 'sum(intensity)', xlabel, ylabel, stats=False)
    elif var=='vol_f':     # # volume conc (fine)
        true = fineFwdScale*np.asarray([rslt['vol'] for rslt in simBase.rsltFwd])[keepInd,fineFwdInd]
        rtrv = np.asarray([rslt['vol'] for rslt in simBase.rsltBck])[keepInd,fineBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Fine Mode Volume', xlabel, ylabel)
    elif var=='vol_c':     # # volume conc (coarse)
        true = np.vstack([rslt['vol'] for rslt in simBase.rsltFwd])[keepInd,crsFwdInd]
        rtrv = np.vstack([rslt['vol'] for rslt in simBase.rsltBck])[keepInd,crsBckInd]
        genPlots(true, rtrv, axScat, axHist, 'Coarse Mode Volume', xlabel, ylabel)
    else:
        # confirm that the variable name is recognized
        assert False, 'Variable name: %s was not recognized!' % var                         


# =============================================================================
# Final formatting and saving of figures
# =============================================================================

# Add title and adjust layouts
dataSetName = path.basename(filePathPtrn)
ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (dataSetName, bckLambda, surf2plot, aodMin)
ttlStr = ttlStr.replace('MERGED_','')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle(ttlStr)
fig_hist.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_hist.suptitle(ttlStr)

# Save the figures
inDirPath = path.dirname(filePathPtrn)
figSaveName = dataSetName.replace('.pkl',('_%s_%s_%04dnm_ScatterPlot.png' % (surf2plot, fnTag, bckLambda*1000)))
print('Saving scatter plot figure to: %s' % path.join(inDirPath,figSaveName))
fig.savefig(path.join(inDirPath,figSaveName), dpi=330)
figSavePathHist = path.join(inDirPath,figSaveName.replace('ScatterPlot','HistErrPlot'))
fig_hist.savefig(figSavePathHist, dpi=330)
print('Saving error histogram figure to: %s' % figSavePathHist)
# plt.show()

# =============================================================================