#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script is a modified version of the simulationQuickLook.py, with 
the ability to plot 2D density and histograms of errors for retrieved parameters.
Specifically designed to accommodate the simulation retrieval study based on
CAMP2Ex measurements

Created on Wed Jul 13 14:18:11 2022
Modified: 

@author: Anin Puthukkudy
"""

# -*- coding: utf-8 -*-
# =============================================================================
# Import the libraries
# =============================================================================

import os
from glob import glob
from pprint import pprint
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'ytick.right': 'True',
    'xtick.top': 'True',
    'mathtext.fontset': 'cm',
    'figure.dpi': 330,
    'font.family': 'cmr10',
    'axes.unicode_minus': False
})
from scipy import interpolate
from scipy.stats import norm, gaussian_kde, ncx2, moyal, binned_statistic
from simulateRetrieval import simulation
import seaborn as sns
from matplotlib.colors import LogNorm
import pickle

# automatic reload option for IPython
from IPython import get_ipython
ipython = get_ipython()
if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Local definitions
# =============================================================================

# aod weighted product
aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)

def violinPlotEdits(parts, facecolor=False, verbose=False):
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
    try:
        for pc in ['cmeans']:
            vp = parts[pc]
            vp.set_edgecolor('k')
            vp.set_linewidth(1)
    except Exception as e:
        if verbose:
            print(e)
            print('Error in plotting mean')

    try:
        for pc in ['cmedians']:
            vp = parts[pc]
            vp.set_linewidth(1)
    except Exception as e:
        if verbose:
            print(e)
            print('Error in plotting median')

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

def makesweetgraph(x=None, y=None, cmap='Blues', ylab=None, xlab=r'AOD$_{550\ nm}$',
                   bins=5, figsize=(3,3), snsbins=20,
                   yScale='linear', xScale='log', alpha=0.3, cScale='linear',
                   cMean=False, pLine=False,
                   **kwargs):
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
        Number of bins for the x axis in 2D histogram. 
    figsize : tuple, optional
        Figure size. The default is (3,3).
    snsbins : int, optional
        Number of bins for the marginal histograms.
    yScale : str, optional
        Scale of the y-axis. The default is 'linear'.
    xScale : str, optional
        Scale of the x-axis. The default is 'linear'.
    alpha : float, optional
        Transparency of the 2D histogram. The default is 0.3.
    **kwargs : dict
        Keyword arguments for the function.
    
    Returns
    -------
    None
    
    '''
    # change the values based on the kwargs
    if kwargs:
        for key, value in kwargs.items():
            if key == 'cmap':
                cmap = value
            elif key == 'ylab':
                ylab = value
            elif key == 'xlab':
                xlab = value
            elif key == 'bins':
                bins = value
            elif key == 'figsize':
                figsize = value
            elif key == 'snsbins':
                snsbins = value
            elif key == 'yScale':
                yScale = value
            elif key == 'xScale':
                xScale = value
            elif key == 'alpha':
                alpha = value
            elif key == 'cScale':
                cScale = value
            else:
                print('Invalid keyword argument:')
                print(key)

    # This is where the x abd y bins are defined
    x_ = np.logspace(np.log10(0.01), np.log10(2),bins)
    if xScale == 'linear':
        x_ = np.linspace(0, np.max(x), bins)
    
    # y axis bins
    yMax = np.max([abs(np.percentile(y, 10)), abs(np.percentile(y, 90))])
    y_ = np.linspace(-yMax, yMax, snsbins)
    if yScale == 'log':
        y_1 = np.logspace(-3, np.log10(yMax), int(snsbins/2))
        y_2 = np.flip(-y_1)
        y_ = np.concatenate((y_2, y_1))
        
    # plot the joint distribution
    ax1 = sns.jointplot()# marginal_kws=dict(bins=(x_, y_))
    ax1.ax_marg_x.hist(x, bins=x_, alpha=alpha)
    # ax1.ax_joint.set_xscale('log')
    ax1.ax_marg_y.hist(y, bins=y_, alpha=alpha, orientation="horizontal")
    ax1.fig.set_size_inches(figsize[0], figsize[1])
    ax1.ax_joint.cla()
    plt.sca(ax1.ax_joint)
    
    # create the 2D histogram. There are two ways to do this: 1) using hist2d or 2) using pcolormesh
    # 1) hist2d is the best option because you have more control over the colorbar bins
    
    # hist = np.log10(np.histogram2d(x,y,bins=(x_log, y_))[0]).T1
    # plt.pcolormesh(x_log, y_, hist, norm=LogNorm() ,cmap=cmap)
    if cScale == 'log':
        plt.hist2d(x,y,bins=(x_, y_), cmap=cmap, norm=LogNorm())
    else:
        plt.hist2d(x,y,bins=(x_, y_), cmap=cmap)
    
    # defining the scale of the axes
    if xScale == 'log':
        plt.semilogx()
    if yScale == 'log':
        if xScale == 'log':
            plt.loglog()
        else:
            plt.semilogy()
            
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    cbar_ax = ax1.fig.add_axes([1, 0.1, .03, .7])

    # set limits of y-axis
    ax1.ax_marg_y.set_ylim(-yMax, yMax)
    
    # add the colorbar
    cb = plt.colorbar(cax=cbar_ax)
    if cScale == 'log':
        cb.set_label(r'$\log_{10}$ density of points')
    else:
        cb.set_label('density of points')
    
    # binned statistics
    if cMean or pLine:
        s, edges, _ = binned_statistic(x, y, statistic='mean', bins=x_)
        std_, _, _ = binned_statistic(x, y, statistic=np.std, bins=x_)
        n_, _, _ = binned_statistic(x, y, statistic='count', bins=x_)
        p15, _, _= binned_statistic(x, y, statistic=lambda x: np.percentile(x, 15), bins=x_)
        p85, _, _= binned_statistic(x, y, statistic=lambda x: np.percentile(x, 85), bins=x_) 

    # Bias of each bin in the x-axis
    if cMean: sns.scatterplot(y=s, x=edges[:-1]+np.diff(edges)/2, alpha=0.5, c="crimson", zorder=3, ax=ax1.ax_joint)
    
    # percentile
    try:
        if pLine:
            sns.lineplot(y=p15, x=edges[:-1]+np.diff(edges)/2, color="y", zorder=2, ax=ax1.ax_joint)
            sns.lineplot(y=p85, x=edges[:-1]+np.diff(edges)/2, color="y", zorder=2, ax=ax1.ax_joint)
    except ValueError:
        print('Error in plotting percentile lines')

def plotSpectralVars(statDict_, specVar, rs_,
                     simBase, 
                     ax, Lst, limits=[-0.1, 0.1],
                     diffMode=False, modeIdx=0,
                     yStr=None, aodWgt=False, 
                     spclVar=None,keepInd_=None, xVarLst=None,
                     **kwargs):
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
    rs_ : dict
        Dictionary containing the specifications for the spectral variables
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
    keepInd_ : array-like, optional
        Indices to keep. The default is None.
    
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
    
    if keepInd_ is None:
        keepInd_ = rs_['keepInd']
    
    # Plot the spectral variables for the forward and backward simulations
    # loop through the spectral bands
    
    for i, bnds in enumerate(rs_['bands']):
        if diffMode:
            #FIXME: This is a hack to get the correct band for the forward and backward simulations of fine and coarse mode
            if aodWgt:             
                true = np.asarray([aodWght(rf[specVar][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltFwd])[keepInd_]
            else:
                true = np.asarray([rf[specVar][:,i] for rf in simBase.rsltFwd])[keepInd_][:,trueMode]
            rtrv = np.asarray([rf[specVar][:,i] for rf in simBase.rsltBck])[keepInd_][:,modeIdx]
        else:
            if aodWgt: 
                true = np.asarray([(1-rf[specVar][i])*rf['aod'][i] for rf in simBase.rsltFwd])[keepInd_]
                rtrv = np.asarray([(1-rb[specVar][i])*rb['aod'][i] for rb in simBase.rsltBck])[keepInd_]
            else:  
                true = np.asarray([rf[specVar][i] for rf in simBase.rsltFwd])[keepInd_]
                rtrv = np.asarray([rf[specVar][i] for rf in simBase.rsltBck])[keepInd_]
                
                # if specVar == 'aod':
                #     rs_['trueAOD'] = true
        
        Rcoef = np.corrcoef(true, rtrv)[0,1]
        RMSE = np.sqrt(np.median((true - rtrv)**2))
        bias = np.mean((rtrv-true))
        median = np.median((rtrv-true))
        sc.calc(statDict_, spclVar, true, rtrv)
        
        Lst.append((rtrv-true).T)
        
        parts = ax.violinplot((rtrv-true).T, [i+1], points= 300,
                                showmeans=False, showmedians=True,
                                showextrema=False,#  quantiles=[0.1, 0.9],
                                bw_method=0.5) # 'silverman' or 'scott' read the https://matplotlib.org/stable/gallery/statistics/violinplot.html#sphx-glr-gallery-statistics-violinplot-py for documentation
        violinPlotEdits(parts)
        
        # line in the violin plot for the mean/median
        ax.text(i+1.1,median, '%0.3f' %round(median, 3), fontsize=8)
        # ax.text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
        
        # plot the makesweetgraph
        if i == rs_['waveInd']:
            if xVarLst is None:
                xVarLst = rs_['trueAOD']
            makesweetgraph(xVarLst, rtrv-true, ylab='Error in %s' %spclVar, **kwargs)
            
    # Change the y axis labels
    if yStr is None:
        yStr = specVar
        
    axisLabelsLimits(ax, rs_['bandStr'], limits=limits,
                     ylabel='Error in \n %s' %yStr) 

def handle_plot_error(ax, ax_idx, y_str, x_lab):
    print(f'Error in plotting {y_str}')
    ax[ax_idx].cla()
    while len(ax[ax_idx].lines) > 0:
        ax[ax_idx].lines[0].remove()
    ax[ax_idx].text(0.5,0.5, f'Error in plotting {y_str} vs {x_lab}', fontsize=8)
    ax[ax_idx].axis('off')

def plotStats(statsArr_, ax, title, cmap='viridis',
              fmt=".2f", **kwargs):
    
    # plot the array
    hm = sns.heatmap(statsArr_, annot=True, fmt=fmt, cmap=cmap, ax=ax, **kwargs)
    ax.set_xticklabels(rs_['bands'])
    ax.set_yticklabels(spectralVars)
    
    hm.set_xticklabels(hm.get_xticklabels(), rotation=60)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax.set_title(title)

def calculate_metrics(true, rtrv):
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    return Rcoef, RMSE, bias
   
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
        dct[varStr]['RMSE'].append(np.sqrt(np.mean((true - rtrv)**2)))
        dct[varStr]['bias'].append(np.mean((rtrv-true)))
        dct[varStr]['10th'].append(np.percentile((rtrv-true), 10))
        dct[varStr]['90th'].append(np.percentile((rtrv-true), 90))
    

# =============================================================================
# %% Initiation
# =============================================================================
# define a runSpec_ dictionary to store the run specifications
rs_ = {}

# Wavelength index for plotting
rs_['waveInd'] = 2
# Wavelength index for AE calculation
rs_['waveInd2'] = 4

# CAMP2Ex configuration dictionary
confLst = [ ['00',  'dark'],
            ['01',  'open'],
            ['02',  'dark'],
            ['03',  'open'],
            ['04',  'open']]
i = 0                               # index of the configuration to be used for the case of camp2ex [0-4]

# Define the string pattern for the file name
rs_['fnPtrnList'] = []
rs_['conf#'] = confLst[i][0]        # Options: 00, 01, 02, 03, 04, 0000, 0100, 0200, 0300, 0400  
rs_['psdBinType'] = 'bi'            # 'bi' or 'tria' This for camp2ex
rs_['month'] = 'Feb'                # Camp2Ex configuration
rs_['day'] = '05'                   # Camp2Ex configuration
rs_['year'] = 2024                  # Camp2Ex configuration
rs_['dirTag'] = 'Sim'               # 'Sim' for real simulation, 'Test' for test run. Camp2Ex configuration
rs_['oceanType'] = confLst[i][1]    # 'dark' or 'bright' ocean surface. Camp2Ex configuration
rs_['instrument'] = 'uvswirmap01'   # 'uvswirmap01' or 'uvswirmap0100' Camp2Ex configuration

rs_['fnPtrn'] = 'Camp2ex_OLH_TS*AOD_*_550nm_*_conf#%s_*_campex_%s_*_flight#*_layer#00.pkl' % (rs_['conf#'], rs_['psdBinType'])
# fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_1000z.pkl'

# Location/dir where the pkl files are
rs_['inDirPath'] = '/data/ESI/User/aputhukkudy/ACCDAM/%d/%s/%s/%s/Full/Geometry/CoarseModeFalse/%sOcean/2modes/%s/' % \
                    (rs_['year'], rs_['dirTag'], rs_['month'], rs_['day'], rs_['oceanType'], rs_['instrument']) 

# more tags and specifiations for the scatter plot
rs_['surf2plot']  = 'both'      # land, ocean or both
rs_['aodMin']     = 0.1         # does not apply to first AOD plot
rs_['aodMax']     = 5
rs_['nMode']      = 0           # Select which layer or mode to plot
rs_['fnTag']      = 'AllCases'
rs_['xlabel']     = 'Simulated Truth'
rs_['MS']         = 2
rs_['FS']         = 10
rs_['LW121']      = 1
rs_['pointAlpha'] = 0.20
rs_['clrText']    = '#FF6347'   # [0,0.7,0.7]
rs_['nBins']      = 200         # no. of bins for histogram
rs_['nBins2']     = 200         # no. of bins for 2D density plot
rs_['lightSave']  = True        # Omit PM elements and extinction profiles from MERGED files to save space
rs_['chiMax']     = 3

# Define the path of the new merged pkl file
rs_['saveFN']     = 'MERGED_'+rs_['fnPtrn'].replace('*','ALL')
rs_['savePATH']   = os.path.join(rs_['inDirPath'],rs_['saveFN'])

# Define the index for the fine and coarse mode
# This is hard coded for now based on the CAMP2Ex simulations
rs_['fineModeIndFWD'] = 0
rs_['coarseModeIndFWD'] = 4
rs_['fineModeIndBCK'] = 0
rs_['coarseModeIndBCK'] = 1

# =============================================================================
# Load data
# =============================================================================

# Define the path of the new merged pkl file
loadPATH = os.path.join(rs_['inDirPath'],rs_['saveFN'])
try:
    simBase = simulation(picklePath=loadPATH)
except Exception as e:
    print(e)
    print('Error in loading the simulation object: check the path')
    print(loadPATH)
    raise SystemExit

rs_['saveFN'] = rs_['saveFN'].replace('#', '-')

# print general stats to console
print('Showing results for %5.3f μm' % simBase.rsltFwd[0]['lambda'][rs_['waveInd']])
pprint(simBase.analyzeSim(rs_['waveInd'])[0])

# =============================================================================
# Filter data
# =============================================================================

# lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
# TODO: this has to be generalized so that it can be ran without crashing the code
# FIXME: at the moment, this hard coded to be ocean pixels
lp = np.array([0 for rf in simBase.rsltFwd])
rs_['keepInd'] = lp>99 if rs_['surf2plot']=='land' else lp<1 if rs_['surf2plot']=='ocean' else lp>-1

# apply convergence filter
σx={'I'   :0.030, # relative
    'QoI' :0.005, # absolute
    'UoI' :0.005, # absolute
    'Q'   :0.005, # absolute in terms of Q/I
    'U'   :0.005, # absolute in terms of U/I
    } # absolute
# ours looks more normal, but GRASP's produces slightly lower RMSE
simBase.conerganceFilter(forceχ2Calc=True, σ=σx) 

rs_['costThresh']   = np.percentile(np.array([rb['costVal'] for rb in simBase.rsltBck])[rs_['keepInd']],10)
rs_['costThresh']   = min(rs_['costThresh'], rs_['chiMax'])
rs_['keepIndAll']   = rs_['keepInd']        # saving all indexes before the chi2 filter
rs_['keepInd_']     = np.logical_and(rs_['keepInd'], [rb['costVal']<rs_['costThresh'] for rb in simBase.rsltBck])
maxIter             = 50                # Keep the pixels that are below max iterations
try:
    rs_['keepInd']  = np.logical_and(rs_['keepInd_'], [rb['nIter']<maxIter for rb in simBase.rsltBck])
    print('%d/%d max iterations filter (maxIter = %d)' % (rs_['keepInd'].sum(), rs_['keepInd_'].sum(), maxIter))
except Exception as e:
    print(e)
    print('Error in applying the max iteration filter')
    print('Keeping all pixels')
    rs_['keepInd'] = rs_['keepInd_']
clrVar              = np.sqrt(np.array([rb['costVal'] for rb in simBase.rsltBck])[rs_['keepInd']]) # this is slow!
rs_['clrVar']       = np.asarray([rb['costVal'] for rb in simBase.rsltBck])[rs_['keepInd']]
rs_['clrVarAll']    = rs_['clrVar']
print('%d/%d fit surface type %s and convergence filter' % (rs_['keepInd_'].sum(), len(simBase.rsltBck), rs_['surf2plot']))

# =============================================================================
# AOD Filter
# =============================================================================

# apply AOD min after we plot AOD
rs_['keepInd'] = np.logical_and(rs_['keepInd'], [rf['aod'][rs_['waveInd']]>=rs_['aodMin'] for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (rs_['keepInd'].sum(), len(simBase.rsltBck), rs_['surf2plot'], rs_['aodMin']))
rs_['keepInd'] = np.logical_and(rs_['keepInd'], [rf['aod'][rs_['waveInd']]<rs_['aodMax'] for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (rs_['keepInd'].sum(), len(simBase.rsltBck), rs_['surf2plot'], rs_['aodMax']))
# =============================================================================
# %% Plotting
# =============================================================================

# define a dict to store the statisctics and 
statsDict_ = {}
varDict = {}

# wavelengths
rs_['bands'] = simBase.rsltFwd[0]['lambda']
rs_['bandStr'] = ['']+[str(i) for i in rs_['bands']]

# store the true AOD and AOD CM
rs_['trueAOD'] = np.asarray([rf['aod'][rs_['waveInd']] for rf in simBase.rsltFwd])[rs_['keepInd']]
rs_['trueAODCoarse'] = np.asarray([rf['aodMode'][rs_['coarseModeIndFWD'], 
                                    rs_['waveInd'] ] for rf in simBase.rsltFwd])[rs_['keepInd']]

# define the fig and axes handles
# Figure for spectral error plots
fig, ax = plt.subplots(3,1, figsize=(6,4))

# Define various spectral variables and the parameters for plotting
# parameters = [(param, y_str, diff_mode, mode_idx, spcl_var, limits, x_var_lst, x_lab)]
parameters = [
    ('aod', 'AOD', None, None, 'aod', [-0.1,0.1], None, None),
    ('aodMode', 'AOD CM', True, 1, 'aodCM', [-0.025,0.025], None, None),
    ('ssa', 'AAOD', None, None, 'aaod', [-0.05,0.05], None, r'AOD$_{550\ nm}$')
    
]

# loop throu parameters to plot the spectral variables
for i, (param, y_str, diff_mode, mode_idx, spcl_var, limits, x_var_lst, x_lab) in enumerate(parameters):
    varDict[spcl_var] = []
    aodWgt_ = True if param == 'ssa' else False
    plotSpectralVars(statsDict_, param, rs_, simBase,
                    ax[i], varDict[spcl_var], diffMode=diff_mode,
                    modeIdx=mode_idx, yStr=y_str,
                    limits=limits, spclVar=spcl_var,
                    xVarLst=x_var_lst, xlab=x_lab, aodWgt=aodWgt_)

# =============================================================================
# %%Plotting n,k
# =============================================================================

# define the fig and axes handles
# Figure for spectral error plots
fig2, ax2 = plt.subplots(4,1, figsize=(6,6))

# loop through bands to calculate the error and bias in individual variables
# parameters = [(param, y_str, diff_mode, mode_idx, spcl_var, limits, x_var_lst, x_lab)]
parameters = [
    ('ssa', 'SSA', False, None, 'ssa', None, None, None),
    ('ssaMode', 'SSA CM', True, 1, 'ssaCM', None, None, None),
    ('k', 'k FM', True, 0, 'kFM', [-0.01,0.01], None, None),
    ('k', 'k CM', True, 1, 'kCM', [-0.002,0.002], None, None)
]
for i, (param, y_str, diff_mode, mode_idx, spcl_var, limits, x_var_lst, x_lab) in enumerate(parameters):
    varDict[spcl_var] = []
    plotSpectralVars(statsDict_, param, rs_, simBase,
                    ax2[i], varDict[spcl_var], diffMode=diff_mode,
                    modeIdx=mode_idx, yStr=y_str,
                    limits=limits, spclVar=spcl_var,
                    xVarLst=x_var_lst, xlab=x_lab)

# %%
# define the fig and axes handles
# Figure for spectral error plots
fig3, ax3 = plt.subplots(4,1, figsize=(6,5))

# parameters = [(param, y_str, diff_mode, mode_idx, spcl_var, limits, x_var_lst, x_lab)]
parameters = [
    (3, 'n', 'n FM', True, 0, 'nFM', [-0.1,0.1], None, None),
    (2, 'n', 'n CM', True, 1, 'nCM', [-0.1,0.1], rs_['trueAODCoarse'], r'AOD$_{550\ nm, coarse}$'),
    (1, 'n', 'n CM', True, 1, 'nCM', [-0.1,0.1], None, None),
    (0, 'ssaMode', 'SSA FM', True, 0, 'ssaFM', None, None, None)
]

for i, (ax_idx, param, y_str, diff_mode, mode_idx, spcl_var, limits, x_var_lst, x_lab) in enumerate(parameters):
    # save the results to a dictionary
    varDict[spcl_var] = []
    try:
        plotSpectralVars(statsDict_, param, rs_, simBase,
                        ax3[ax_idx], varDict[spcl_var], diffMode=diff_mode,
                        modeIdx=mode_idx, yStr=y_str,
                        limits=limits, spclVar=spcl_var,
                        xVarLst=x_var_lst, xlab=x_lab)
    except Exception as e:
        print(e)
        handle_plot_error(ax3, ax_idx, y_str, x_lab)
#%% Spectrally independent

fig4, ax4 = plt.subplots(1,1, figsize=(3,2))

tempInd = 0
for nMode_ in [0,4]:
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[rs_['keepInd']][:,tempInd]
    if nMode_ == 0:
        spclVar = 'volFine'
        sc = statsCalc(statsDict_, spclVar)
        if 'olh' in rs_['fnPtrn'].lower():
            true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[rs_['keepInd']][:,nMode_]
        else:
            true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[rs_['keepInd']][:,nMode_]*4
    else:
        spclVar = 'volCoarse'
        sc = statsCalc(statsDict_, spclVar)
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[rs_['keepInd']][:,nMode_]
    
    Rcoef, RMSE, bias = calculate_metrics(true, rtrv)

    varDict[spclVar] = []
    varDict[spclVar].append((rtrv-true).T)

    parts = ax4.violinplot((rtrv-true).T, [tempInd+1], points= 300,
                            showmeans=True, showmedians=False,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax4.text(tempInd+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    # ax4.text(2.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    sc.calc(statsDict_, spclVar, true, rtrv)
    # plot the makesweetgraph
    makesweetgraph(rs_['trueAOD'], (rtrv-true), ylab='Error in %s' %spclVar, xlab=r'AOD$_{550\ nm}$')
    tempInd += 1
axisLabelsLimits(ax4, ['', 'Fine mode\n vol. conc.', '', 'Coarse mode\n vol. conc.'], limits=[-0.2, 0.2],
                 ylabel=r'Absolute error', xlabel='')

#%% Effective raidius
fig5, ax5 = plt.subplots(1,2, figsize=(3,2))
for i in [0,1]:
    # Fine mode rEff
    true = np.asarray([rf['rEffMode'][i] for rf in simBase.rsltFwd])[rs_['keepInd']]
    rtrv = np.asarray([rf['rEffMode'][i] for rf in simBase.rsltBck])[rs_['keepInd']]

    Rcoef, RMSE, bias = calculate_metrics(true, rtrv)
    spclVar = 'rEffFine' if i == 0 else 'rEffCoarse'
    varDict[spclVar] = []
    varDict[spclVar].append((rtrv-true).T)
    
    parts = ax5[i].violinplot((rtrv-true).T, [1], points= 300,
                            showmeans=True, showmedians=False,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax5[i].text(1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    # ax5[i].text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')

    
    sc = statsCalc(statsDict_, spclVar)
    sc.calc(statsDict_, spclVar, true, rtrv) 
    # plot the makesweetgraph
    makesweetgraph(rs_['trueAOD'], (rtrv-true), ylab='Error in %s' %spclVar, xlab=r'AOD$_{550\ nm}$')
    limitsAndLabel = [[[-0.1, 0.1], r'Submicron r$_{eff}$', r'Absolute error ($\mu$m)'], 
                        [[-3, 3], r'Above micron r$_{eff}$', '']]
    axisLabelsLimits(ax5[i], ['', '', limitsAndLabel[i][1]], limits=limitsAndLabel[i][0],
                    ylabel=limitsAndLabel[i][2], xlabel='')

#%% Spectrally independent SF
fig6, ax6 = plt.subplots(1,1, figsize=(3,2))

for i in [0,1]:
    # Fine mode SF
    if i == 0:
        true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[rs_['keepInd']]
    elif i == 1:
        true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[rs_['keepInd']][:,4]
    
    rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[rs_['keepInd']]
    # Modifying the true value based on the NDIM
    # if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
    # and the coarse mode 'sea salt' have different value. So based on the dimension of the var
    # We can distinguish each run type and generalize the code
    if true.ndim > 1:
        true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[rs_['keepInd']][:, rs_['nMode']]
        rtrv = np.asarray([rf['sph'] for rf in simBase.rsltBck])[rs_['keepInd']][:, i]
    elif i == 1:
        true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[rs_['keepInd']][:, 4]
        rtrv = np.asarray([rf['sph'] for rf in simBase.rsltBck])[rs_['keepInd']][:, i]
    Rcoef, RMSE, bias = calculate_metrics(true, rtrv)
    spclVar = 'sphFine' if i == 0 else 'sphCoarse'
    varDict[spclVar] = []
    varDict[spclVar].append((rtrv - true).T)
    parts = ax6.violinplot((rtrv - true).T, [i+1], points=300, showmeans=False, showmedians=True,
                        showextrema=False, bw_method='silverman')
    violinPlotEdits(parts)
    ax6.text(i+1.1, bias, '%0.4f' % round(bias, 4), fontsize=8)
    sc = statsCalc(statsDict_, spclVar)
    sc.calc(statsDict_, spclVar, true, rtrv)
    makesweetgraph(rs_['trueAOD'], (rtrv - true), ylab='Error in %s' % spclVar, xlab=r'AOD$_{550\ nm}$')
axisLabelsLimits(ax6, ['', 'Fine mode\n SF', '', 'Coarse mode\n SF.'], limits=[-50, 50],
                ylabel=r'Absolute error (%)', xlabel='')

#%% AOD weighted vol. conc.
fig7, ax7 = plt.subplots(1,1, figsize=(3,3))

tempInd = 0
for nMode_ in [0,4]:
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[rs_['keepInd']][:,tempInd]
    if nMode_ == 0:
        true = np.asarray([aodWght(rf['vol'], rf['aodMode'][:,rs_['waveInd']]) for rf in simBase.rsltFwd])[rs_['keepInd']]
        # true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[rs_['keepInd']][:,nMode_]*4
        if not 'olh' in rs_['fnPtrn'].lower():
            rtrv = rtrv*4
        spclVar = 'volFineAOD'
    else:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[rs_['keepInd']][:,nMode_]
        spclVar = 'volCoarseAOD'
    Rcoef, RMSE, bias = calculate_metrics(true, rtrv)
    varDict[spclVar] = []
    varDict[spclVar].append((rtrv-true).T)
    parts = ax7.violinplot((rtrv-true).T, [tempInd+1], points= 300,
                            showmeans=True, showmedians=False,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax7.text(tempInd+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    # ax7.text(1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    
    sc = statsCalc(statsDict_, spclVar)
    sc.calc(statsDict_, spclVar, true, rtrv)
    # plot the makesweetgraph
    makesweetgraph(rs_['trueAOD'], (rtrv-true), ylab='Error in %s' %spclVar, xlab=r'AOD$_{550\ nm}$')    
    tempInd += 1
axisLabelsLimits(ax7, ['', 'Fine mode\n vol. conc.', '', 'Coarse mode\n vol. conc.'], limits=[-0.1, 0.1],
                 ylabel=r'Absolute error', xlabel='')

#%% Surface
fig8, ax8 = plt.subplots(1,1, figsize=(6,3))
spclVar = 'wtrSurf'
varDict[spclVar] = []
sc = statsCalc(statsDict_, spclVar)
for i, bnds in enumerate(rs_['bands']):
    true = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltFwd])[rs_['keepInd']]
    rtrv = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltBck])[rs_['keepInd']]
    
    Rcoef, RMSE, bias = calculate_metrics(true, rtrv)
    
    varDict[spclVar].append((rtrv-true).T)
    parts = ax8.violinplot((rtrv-true).T, [i+1], points= 300,
                            showmeans=True, showmedians=False,
                            showextrema=False,
                            bw_method='silverman')
    violinPlotEdits(parts)
    ax8.text(i+1.1,bias, '%0.4f' %round(bias, 4), fontsize=8)
    # ax8.text(i+1.1,RMSE, '%0.4f' %round(RMSE, 4), fontsize=8, color='r')
    
    sc.calc(statsDict_, spclVar, true, rtrv)
    # plot the makesweetgraph
    if i == rs_['waveInd']:
        makesweetgraph(rs_['trueAOD'], (rtrv-true), ylab='Error in %s' %spclVar, xlab=r'AOD$_{550\ nm}$')
axisLabelsLimits(ax8, rs_['bandStr'], limits=[-0.01, 0.01],
                 ylabel=r'Error in L$_{water}$')

# FMF distribution plot using simulated data and retrieved data
true = np.asarray([rf['aodMode'][:,rs_['waveInd']] for rf in simBase.rsltFwd])[rs_['keepInd']][:,4]
rtrv = np.asarray([rf['aodMode'][:,rs_['waveInd']] for rf in simBase.rsltBck])[rs_['keepInd']][:,1]
trueT = np.asarray([rf['aod'][rs_['waveInd']] for rf in simBase.rsltFwd])[rs_['keepInd']]
rtrvT = np.asarray([rf['aod'][rs_['waveInd']] for rf in simBase.rsltBck])[rs_['keepInd']]
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

# plot the difference in FMF as a function of AOD
makesweetgraph(rs_['trueAOD'], (cmf_rtrv-cmf), ylab='Error in FMF', xlab='AOD$_{550\ nm}$')
try:
    makesweetgraph(rs_['trueAODCoarse'], (cmf_rtrv-cmf), ylab='Error in FMF', xlab='AOD$_{550\ nm, Coarse}$')
except Exception as e:
    print(e)
# =============================================================================
# %% Save the figure
# =============================================================================
figSavePath = rs_['saveFN'].replace('.pkl',('_%s_%s_spectral.png' % (rs_['surf2plot'], rs_['fnTag'])))
print('Saving figure to: %s' % (os.path.join(rs_['inDirPath'],figSavePath)))
ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (rs_['saveFN'],
                                                    simBase.rsltFwd[0]['lambda'][rs_['waveInd']],
                                                    rs_['surf2plot'], rs_['aodMin'])
figures = [fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig10]
save_paths = ['', 'k_', 'n_', 'vol_', 'rEff_', 'SF_', 'aodWeightVol_', 'L_', 'FMF_']

for figure, save_path in zip(figures, save_paths):
    if save_path == '':
        figure.suptitle(ttlStr.replace('MERGED_', ''))
    figure.savefig(rs_['inDirPath'] + figSavePath.replace('MERGED_', save_path), dpi=330)
print('Saving figure to: %s' % (os.path.join(rs_['inDirPath'],figSavePath.replace('MERGED_','Hist_'))))
# plt.show()
# %% Save the stats

# open a file, where you ant to store the data
f = open(os.path.join(rs_['inDirPath'],figSavePath.replace('MERGED_','STATS_').replace('.png','.pkl')), 'wb')

try:
    # write data to the file
    pickle.dump(statsDict_, f)
    print('Saving stats to: %s' % (os.path.join(rs_['inDirPath'],figSavePath.replace('MERGED_','STATS_').replace('.png','.pkl'))))
except Exception as e:
    print(e)
# close the file
f.close()
    
# %% Load the stats from the file and plot them in a heatmap

# fig11, ax11 = plt.subplots(3,2, figsize=(15,8))

# define the 3D array
spectralVars = ['aod', 'aodCM', 'aaod', 'ssa', 'ssaCM', 'kFM', 'kCM', 'nFM', 'nCM', 'ssaFM', 'wtrSurf']
nVar  = len(spectralVars)
nWav = len(statsDict_[spectralVars[0]]['R'])
nStats = len(statsDict_[spectralVars[0]])

statsArr = np.zeros((nVar, nWav, nStats))

# fill the array
for i, var in enumerate(spectralVars):
    for j, stat in enumerate(statsDict_[var]):
        try:
            statsArr[i,:,j] = statsDict_[var][stat]
        except Exception as e:
            print(e)
            print('Error in %s' %var)
            print('Error in %s' %stat)
            print('Error in %s' %statsDict_[var][stat])
            print('Error in %s' %statsDict_[var])
        
# plot the array
# hm = sns.heatmap(statsArr[:,:,0], annot=True, fmt=".2f", cmap='viridis', ax=ax11.flatten()[0])
# ax11[0,0].set_xticklabels(bands)
# ax11[0,0].set_yticklabels(spectralVars)

# hm.set_xticklabels(hm.get_xticklabels(), rotation=60)
# hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

# %% Plotting the stats for the spectral dependent variables 
fig_ = {}
ax_ = {} 
for j, stat in enumerate(statsDict_[spectralVars[0]]):
    fig_[j], ax_[j] = plt.subplots(1, 1, figsize=(6,4))
    if 'bias' in stat:
        plotStats(statsArr[:,:,j], ax_[j], stat, cmap='RdBu_r',
                  fmt=".3f", vmin=-0.1, vmax=0.1)
    elif 'RMSE' in stat:
        plotStats(statsArr[:,:,j], ax_[j], stat, cmap='magma',
                  fmt=".3f", vmin=0, vmax=0.05)
    elif 'R' in stat:
        plotStats(statsArr[:,:,j], ax_[j], stat, cmap='YlGn',
                  fmt=".3f", vmin=0, vmax=1)
    elif '90th' in stat:
        plotStats(statsArr[:,:,j], ax_[j], stat, fmt=".3f", cmap='YlGn_r')
    elif '10th' in stat:
        plotStats(statsArr[:,:,j], ax_[j], stat, fmt=".3f", cmap='YlGn')
# %% Plotting the stats for the spectral independent variables

fig12, ax12 = plt.subplots(1,1, figsize=(6,3))

# define the 3D array
spectralVars2 = ['volFine', 'volCoarse', 'rEffFine', 'rEffCoarse', 'sphFine', 'sphCoarse', 'volFineAOD', 'volCoarseAOD']
nVar2  = len(spectralVars2)
nStats = len(statsDict_[spectralVars2[0]])
statKeys = list(statsDict_[spectralVars2[0]].keys())
statsArr2 = np.zeros((nVar2, nStats))

# fill the array
for i, var in enumerate(spectralVars2):
    for j, stat in enumerate(statsDict_[var]):
        statsArr2[i,j] = statsDict_[var][stat][0]
        
# plot the array
hm2 = sns.heatmap(statsArr2[:,:], annot=True, fmt=".3f", vmin= -1, vmax=1, cmap='RdYlGn', ax=ax12)
ax12.set_xticklabels(statKeys)
ax12.set_yticklabels(spectralVars2)
hm2.set_yticklabels(hm2.get_yticklabels(), rotation=0)
hm2.set_xticklabels(hm2.get_xticklabels(), rotation=60)

# %% Plotting the cross correlation matrix
fig13, ax13 = plt.subplots(1,1, figsize=(7,7))

# aod based mask FIXME: this is wrong
# rs_['aodMaskInd'] = [x >= rs_['aodMin'] for x in varDict['aod'][rs_['waveInd']]]

rs_['aodMaskInd'] = np.ones(len(varDict['aod'][rs_['waveInd']]), dtype=bool)

# define the 2D array
spectralVars3 = ['volFine', 'volCoarse', 'rEffFine', 'rEffCoarse', 'sphFine', 'sphCoarse', 'volFineAOD', 'volCoarseAOD']
nVar3  = len(spectralVars3)
statsArr3 = np.zeros((nVar3, nVar3))

# fill the array
for i, var in enumerate(spectralVars3):
    for j, var2 in enumerate(spectralVars3):
        statsArr3[i,j] = calculate_metrics(varDict[var][0][rs_['aodMaskInd']],
                                            varDict[var2][0][rs_['aodMaskInd']])[0]

# plot the array
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(statsArr3, dtype=bool), k=1)
hm3 = sns.heatmap(statsArr3[:,:], mask=mask, annot=True, fmt=".3f", vmin= -1, vmax=1, cmap='RdBu_r', ax=ax13)
ax13.set_xticklabels(spectralVars3)
ax13.set_yticklabels(spectralVars3)
hm3.set_yticklabels(hm3.get_yticklabels(), rotation=0)
hm3.set_xticklabels(hm3.get_xticklabels(), rotation=60)

# %% Plotting the cross correlation matrix
fig14, ax14 = plt.subplots(1,1, figsize=(9,6))

# define the 2D array
spectralVars4 = ['aod', 'aodCM', 'aaod', 'ssa',  'ssaFM', 'ssaCM', 'nFM', 'nCM', 'kFM', 'kCM', 'wtrSurf']
nVar4  = len(spectralVars4)
statsArr4 = np.zeros((nVar4, nVar4))

# fill the array
for i, var in enumerate(spectralVars4):
    for j, var2 in enumerate(spectralVars4):
    
        try:
            statsArr4[i,j] = calculate_metrics(varDict[var][rs_['waveInd']][rs_['aodMaskInd']],
                                                varDict[var2][rs_['waveInd']][rs_['aodMaskInd']])[0]
        except Exception as e:
            print(e)
            print('Error in %s' %var)
            print('Error in %s' %var2)
            raise SyntaxError('Error in calculating the cross correlation matrix')
            


# plot the array
mask = np.triu(np.ones_like(statsArr4, dtype=bool), k=1)
hm4 = sns.heatmap(statsArr4[:,:], mask=mask, annot=True, fmt=".3f", vmin= -1, vmax=1, cmap='RdBu_r', ax=ax14)
ax14.set_xticklabels(spectralVars4)
ax14.set_yticklabels(spectralVars4)
hm4.set_yticklabels(hm4.get_yticklabels(), rotation=0)
hm4.set_xticklabels(hm4.get_xticklabels(), rotation=60)

# %% Plotting the cross correlation matrix with all the variables
fig15, ax15 = plt.subplots(1,1, figsize=(12,8))

# define the 2D array
spectralVars5 = ['aod', 'aodCM', 'aaod', 'ssa', 'ssaFM', 'ssaCM', 'nFM', 'nCM', 'kFM', 'kCM', 'wtrSurf',
                'volFine', 'volCoarse', 'rEffFine', 'rEffCoarse', 'sphFine', 'sphCoarse', 'volFineAOD', 'volCoarseAOD']
nVar5  = len(spectralVars5)
statsArr5 = np.zeros((nVar5, nVar5))

# fill the array
for i, var in enumerate(spectralVars5):
    for j, var2 in enumerate(spectralVars5):
        try:
            if var in ['aod', 'aodCM', 'aaod', 'ssa', 'ssaCM', 'kFM', 'kCM', 'nFM', 'nCM', 'ssaFM', 'wtrSurf']:
                if var2 in ['aod', 'aodCM', 'aaod', 'ssa', 'ssaCM', 'kFM', 'kCM', 'nFM', 'nCM', 'ssaFM', 'wtrSurf']:
                    statsArr5[i,j] = calculate_metrics(varDict[var][rs_['waveInd']][rs_['aodMaskInd']],
                                                        varDict[var2][rs_['waveInd']][rs_['aodMaskInd']])[0]
                else:
                    statsArr5[i,j] = calculate_metrics(varDict[var][rs_['waveInd']][rs_['aodMaskInd']],
                                                        varDict[var2][0][rs_['aodMaskInd']])[0]
            else:
                if var2 in ['volFine', 'volCoarse', 'rEffFine', 'rEffCoarse', 'sphFine', 'sphCoarse', 'volFineAOD', 'volCoarseAOD']:
                    statsArr5[i,j] = calculate_metrics(varDict[var][0][rs_['aodMaskInd']],
                                                        varDict[var2][0][rs_['aodMaskInd']])[0]
                else:
                    statsArr5[i,j] = calculate_metrics(varDict[var][0][rs_['aodMaskInd']],
                                                        varDict[var2][rs_['waveInd']][rs_['aodMaskInd']])[0]
        except Exception as e:
            print(e)
            print('Error in %s' %var)
            print('Error in %s' %var2)

# plot the array
mask = np.triu(np.ones_like(statsArr5, dtype=bool), k=1)
hm5 = sns.heatmap(statsArr5[:,:], mask=mask, annot=True, fmt=".3f", vmin= -1, vmax=1, cmap='RdBu_r', ax=ax15)
ax15.set_xticklabels(spectralVars5)
ax15.set_yticklabels(spectralVars5)
hm5.set_yticklabels(hm5.get_yticklabels(), rotation=0)
hm5.set_xticklabels(hm5.get_xticklabels(), rotation=60)
ax15.set_title(r'Cross correlation matrix for all the variables (Spectral band=%0.3f $\mu$m)' %rs_['bands'][rs_['waveInd']])

# %% save the figure fig_[:], fig12, fig13, fig14, fig15

figSavePath = rs_['saveFN'].replace('.pkl',('_%s_%s_stats.png' % (rs_['surf2plot'], rs_['fnTag'])))
print('Saving figure to: %s' % (os.path.join(rs_['inDirPath'],figSavePath)))

figures = [ fig_, fig12, fig13, fig14, fig15]
save_paths = ['stats_', 'stats2_', 'Corr1_', 'Corr2_', 'CorrAll_']

for figure, save_path in zip(figures, save_paths):
    if figure == fig_:
        print(fig_)
        for i in range(0,5):
            save_path = save_path + str(i)
            figure[i].savefig(rs_['inDirPath'] + figSavePath.replace('MERGED_', save_path), dpi=330)
            print('Saving figure to: %s' % (rs_['inDirPath'] + figSavePath.replace('MERGED_', save_path)))
    else:
        figure.savefig(rs_['inDirPath'] + figSavePath.replace('MERGED_', save_path), dpi=330)
    print('Saving figure to: %s' % (rs_['inDirPath'] + figSavePath.replace('MERGED_', save_path)))


# %%
