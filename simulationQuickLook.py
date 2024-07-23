#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'})
mpl.rcParams.update({'ytick.right': 'True'}); mpl.rcParams.update({'xtick.top': 'True'}); 
plt.rcParams['mathtext.fontset'] = 'cm'; plt.rcParams["figure.dpi"]=330; 
plt.rcParams["font.family"] = "cmr10"; plt.rcParams["axes.formatter.use_mathtext"] = True
from simulateRetrieval import simulation
from glob import glob
from scipy import interpolate
#==============================================================================
# Definitions
#==============================================================================

def density_scatter(x, y, ax=None, fig=None, sort=True, bins=20,
                    mplscatter=False, **kwargs):
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
# =============================================================================
# Main
# =============================================================================

waveInd = 2
waveInd2 = 4
fnPtrnList = []
#fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_*z.pkl'
# fnPtrn = 'megaharp01_CAMP2Ex_2modes_AOD_*_550nm_addCoarse__campex_flight#*_layer#00.pkl'
fnPtrn = 'Camp2ex_OLH_AOD_*_550nm_*_conf#03_*_campex_bi_*_flight#*_layer#00.pkl'
# fnPtrn = 'ss450-g5nr.leV210.GRASP.example.polarimeter07.200608*_1000z.pkl'
inDirPath = '/data/ESI/User/aputhukkudy/ACCDAM/2024/Sim/Jan/23/Full/Geometry/CoarseModeFalse/darkOcean/2modes/uvswirmap01/'

surf2plot = 'both' # land, ocean or both
aodMin = 0.2 # does not apply to first AOD plot
nMode = 0 # Select which layer or mode to plot
fnTag = 'AllCases'
xlabel = 'Simulated Truth'
MS = 5
FS = 10
LW121 = 1
pointAlpha = 0.10
clrText = [0.5,0,0.0]
fig, ax = plt.subplots(2,5, figsize=(15,6))
plt.locator_params(nbins=3)
lightSave = True # Omit PM elements and extinction profiles from MERGED files to save space

saveFN = 'MERGED_'+fnPtrn.replace('*','ALL')
savePATH = os.path.join(inDirPath,saveFN)
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
lp = np.array([0 for rf in simBase.rsltFwd])
keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1

# apply convergence filter
# apply convergence filter
σx={'I'   :0.030, # relative
    'QoI' :0.005, # absolute
    'UoI' :0.005, # absolute
    'Q'   :0.005, # absolute in terms of Q/I
    'U'   :0.005, # absolute in terms of U/I
    } # absolute
simBase.conerganceFilter(forceχ2Calc=True, σ=σx) # ours looks more normal, but GRASP's produces slightly lower RMSE
costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 90)
print('Cost function threshold: %5.3f' % costThresh)
keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])
keepIndAll = keepInd

# variable to color point by in all subplots
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]])
clrVarAll = clrVar
print('%d/%d fit surface type %s and convergence filter' % (keepInd.sum(), len(simBase.rsltBck), surf2plot))

# AOD
true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[0,0].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,0].set_title('AOD')
ax[0,0].set_ylabel('Retrieved')
ax[0,0].set_xlim(minAOD,maxAOD)
ax[0,0].set_ylim(minAOD,maxAOD)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
# ax[0,0].scatter(true, rtrv, '.',  c=clrVar, markersize=MS, alpha=pointAlpha)
im = ax[0,0].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
plt.colorbar(im)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[0,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# AAOD
true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
# minAOD = 0.735
maxAOD = 0.15
ax[0,2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
# ax[0,2].set_title('Co-Albedo (1-SSA)')
ax[0,2].set_title('AAOD')
# ax[0,2].set_xticks(np.arange(0.75, 1.01, 0.05))
# ax[0,2].set_yticks(np.arange(0.75, 1.01, 0.05))
ax[0,2].set_xlim(minAOD,maxAOD)
ax[0,2].set_ylim(minAOD,maxAOD)
ax[0,2].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
tHnd = ax[0,2].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)


# apply AOD min after we plot AOD
keepInd = np.logical_and(keepInd, [rf['aod'][waveInd]>=aodMin for rf in simBase.rsltFwd])
print('%d/%d fit surface type %s and aod≥%4.2f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
# clrVar = np.sqrt([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) # this is slow!
clrVar = np.asarray([rb['costVal'] for rb in simBase.rsltBck[keepInd]]) 


# apply Reff min
# simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
simBase._addReffMode(0.7, True) # reframe with cut at 1 micron diameter
#keepInd = np.logical_and(keepInd, [rf['rEffMode']>=2.0 for rf in simBase.rsltBck])
#print('%d/%d fit surface type %s and aod≥%4.2f AND retrieved Reff>2.0μm' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, aodMin))
#clrVar = np.sqrt([rb['rEff']/rf['rEff']-1 for rb,rf in zip(simBase.rsltBck[keepInd], simBase.rsltFwd[keepInd])])

# ANGSTROM
aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
true = -np.log(aod1/aod2)/logLamdRatio
aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
rtrv = -np.log(aod1/aod2)/logLamdRatio
minAOD = np.percentile(true,1) # BUG: Why is Angstrom >50 in at least one OSSE cases?
maxAOD = np.percentile(true,99)
ax[0,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,1].set_title('Angstrom Exponent')
ax[0,1].set_xlim(minAOD,maxAOD)
ax[0,1].set_ylim(minAOD,maxAOD)
ax[0,1].set_xticks(np.arange(minAOD, maxAOD, 0.5))
ax[0,1].set_yticks(np.arange(minAOD, maxAOD, 0.5))
ax[0,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
tHnd = ax[0,1].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# k
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
    minAOD = np.min(true)
    maxAOD = np.max(true)
    ax[0,3].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[0,3].set_title('k')
    ax[0,3].set_xlabel(xlabel)
    
    if tempInd == 1:
            cmap = 'rainbow'
    else:
             cmap = 'viridis'
    ax[0,3].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha, cmap=cmap)
    tempInd += 1
minAOD = 0.0005
maxAOD = np.max(true)*1.05
ax[0,3].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[0,3].set_title('k')
ax[0,3].set_xscale('log')
ax[0,3].set_yscale('log')
ax[0,3].set_xlim(minAOD,maxAOD)
ax[0,3].set_ylim(minAOD,maxAOD)
ax[0,3].set_xticks([0.001, 0.01])
ax[0,3].set_yticks([0.001, 0.01])
ax[0,3].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[0,3].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[0,3].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# FMF (vol)
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
    ax[1,0].set_xlabel(xlabel)
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
    ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel('true-retrieved')
    ax[1,0].set_xlim(minAOD,maxAOD)
    ax[1,0].set_ylim(-maxAOD/10,maxAOD/10)
    # ax[1,0].set_yscale('log')
    ax[1,0].set_xscale('log')
    ax[1,0].scatter(true, rtrv, c=clrVarAll, s=MS, alpha=pointAlpha)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    # frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[1,0].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    # textstr = frmt % (Rcoef, RMSE, bias)
    # tHnd = ax[1,0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
    #                     textcoords='offset points', color=clrText, fontsize=FS)    
    
    # g
    true = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltBck])[keepInd]
    minAOD = np.min(true)*0.95
    maxAOD = np.max(true)*1.05
    ax[0,4].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[0,4].set_title('SSA')
    ax[0,4].set_xlabel(xlabel)
    ax[0,4].set_ylabel('Retrieved')
    ax[0,4].set_xlim(minAOD,maxAOD)
    ax[0,4].set_ylim(minAOD,maxAOD)
    im_ = ax[0,4].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
    plt.colorbar(im_)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[0,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[0,4].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)

# sph
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
ax[1,1].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
ax[1,1].set_title('spherical vol. frac.')
ax[1,1].set_xlabel(xlabel)
ax[1,1].set_xticks(np.arange(minAOD, maxAOD, 25))
ax[1,1].set_yticks(np.arange(minAOD, maxAOD, 25))
ax[1,1].set_xlim(minAOD,maxAOD)
ax[1,1].set_ylim(minAOD,maxAOD)
ax[1,1].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,1].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,1].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# rEff
#simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]
minAOD = np.min(true)*0.95
maxAOD = np.max(true)*1.05
ax[1,2].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
# ax[1,2].set_title('r_eff Total')
ax[1,2].set_title('Submicron r_eff')
ax[1,2].set_xlabel(xlabel)
ax[1,2].set_xlim(minAOD,maxAOD)
ax[1,2].set_ylim(minAOD,maxAOD)
ax[1,2].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,2].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,2].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

ttlStr = '%s (λ=%5.3fμm, %s surface, AOD≥%4.2f)' % (saveFN, simBase.rsltFwd[0]['lambda'][waveInd], surf2plot, aodMin)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(ttlStr.replace('MERGED_',''))

# n
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
    minAOD = np.min(true)
    maxAOD = np.max(true)
    ax[1,3].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[1,3].set_title('n')
    ax[1,3].set_xlabel(xlabel)
    # ax[1,3].set_xlim(minAOD,maxAOD)
    # ax[1,3].set_ylim(minAOD,maxAOD)
    if tempInd == 1:
            cmap = 'magma'
    else:
             cmap = 'viridis'
    ax[1,3].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha, cmap=cmap)
    tempInd += 1
    plt.figure()
    ax_diff = density_scatter(rtrv-true, clrVar)
    ax_diff.set_xlabel('Retrieved - True')
    ax_diff.set_ylabel('chi2')
    if nMode_ == 0:
        ax_diff.set_title('fine mode')
    else:
        ax_diff.set_title('coarse mode')
Rcoef = np.corrcoef(true, rtrv)[0,1]
RMSE = np.sqrt(np.median((true - rtrv)**2))
bias = np.mean((rtrv-true))
frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
tHnd = ax[1,3].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
            textcoords='offset points', color=clrText, fontsize=FS)
textstr = frmt % (Rcoef, RMSE, bias)
tHnd = ax[1,3].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

# %% intensity
intensity = False
if intensity:
    true = np.sum([rb['meas_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
    rtrv = np.sum([rb['fit_I'][:,waveInd] for rb in simBase.rsltBck[keepInd]], axis=1)
    minAOD = np.min(true)*0.95
    maxAOD = np.max(true)*1.05
    ax[1,4].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
    ax[1,4].set_title('sum(intensity)')
    ax[1,4].set_xlabel(xlabel)
    ax[1,4].set_xlim(minAOD,maxAOD)
    ax[1,4].set_ylim(minAOD,maxAOD)
    ax[1,4].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha)
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[1,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[1,4].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)
else:
    true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd]
    rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd]
    tempInd = 0
    for nMode_ in [0,4]:
        true = np.asarray([rf['vol'] for rf in simBase.rsltFwd])[keepInd][:,nMode_]
        rtrv = np.asarray([rf['vol'] for rf in simBase.rsltBck])[keepInd][:,tempInd]
        minAOD = np.min(true)*0.95
        maxAOD = np.max(true)*1.05
        ax[1,4].plot([minAOD,maxAOD], [minAOD,maxAOD], 'k', linewidth=LW121)
        ax[1,4].set_title('Vol concentration')
        ax[1,4].set_xlabel(xlabel)
        # ax[1,4].set_xlim(minAOD,maxAOD)
        # ax[1,4].set_ylim(minAOD,maxAOD)
        if tempInd == 1:
            cmap = 'magma'
        else:
             cmap = 'viridis'
        ax[1,4].scatter(true, rtrv, c=clrVar, s=MS, alpha=pointAlpha, cmap = cmap)
        tempInd += 1
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f'
    tHnd = ax[1,4].annotate('N=%4d' % len(true), xy=(0, 1), xytext=(85, -124), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias)
    tHnd = ax[1,4].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)


figSavePath = saveFN.replace('.pkl',('_%s_%s_%04dnm.png' % (surf2plot, fnTag, simBase.rsltFwd[0]['lambda'][waveInd]*1000)))
print('Saving figure to: %s' % figSavePath)
plt.savefig(inDirPath + figSavePath, dpi=330)
# plt.show()

