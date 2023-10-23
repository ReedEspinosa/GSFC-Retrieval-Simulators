#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pprint import pprint
import re
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from glob import glob
from simulateRetrieval import simulation # This should ensure GSFC-GRASP-Python-Interface is in the path
from miscFunctions import calculatePM

# conCase = 'case08'+chr(ord('a')+int(sys.argv[1]))

# conCases = ['case08l','case08k']+['case08'+chr(ord('p')+x) for x in range(6)]
# conCase = conCases[int(sys.argv[1])]
# print(conCase)

waveInd = 2
waveInd2 = 5
waveIndAOD = 2
fineIndFwd = [0,2]
fineIndBck = [0]
# pklDataPath = '/Users/wrespino/Synced/Working/OSSE_Test_Run/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl' # None to skip reloading of data
pklDataPath = '/Users/wrespino/Synced/AOS/Phase-A/PLRA_RequirementsAndTraceability/GSFC_ValidationSimulationsData/V0/Run-19_polarAOS_case08*_tFct*_n*_nAng0.pkl' # None to skip reloading of data
# pklDataPath = '/Users/wrespino/Synced/AOS/Phase-A/PLRA_RequirementsAndTraceability/GSFC_ValidationSimulationsData/V0/Run-11_polarAOS_'+conCase+'*_tFct*_n*_nAng0.pkl' # None to skip reloading of data
# pklDataPath = '/Users/wrespino/Synced/AOS/A-CCP/Assessment_8K_Sept2020/SIM17_SITA_SeptAssessment_AllResults_MERGED/DRS_V01_Lidar050+polar07_caseAll_tFct1.00_orbSS_multiAngles_nAll_nAngALL.pkl'
# pklDataPath = None # None to skip reloading of data
# plotSaveDir = '/Users/wrespino/Synced/AOS/PLRA/Figures_AODF_bugFixApr11'
plotSaveDir = '/Users/wrespino/Synced/AOS/Phase-A/PLRA_RequirementsAndTraceability/GSFC_ValidationSimulationPlots'
surf2plot = 'both' # land, ocean or both
aodMax = 9990.801 # only for plot limits
hist2D = False
Nbins = 300 # NbinsxNbins bins in hist density plots
szaMin = 0

varVsAOD = False
saveScatter = True # This can be slow (?)
fineAOD = False # use fine mode for varVsAOD plots AND AOD threshold for intensives
scatAlpha = 0.1
scatSize = 5
MS = 1
FS = 10
FStitle = 8 # title may run off plot, make this smaller to fix
LW121 = 1
clrText = [0.5,0,0.0]
clrText = [0.0,0,0.5]
maxNrmErr = 2.5 # normalized uncertainty max in CDF plots
errFun = lambda t,r : np.abs(r-t)
aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)


orb = '1330LTAN_%s' % surf2plot
pklNmMtch = re.match('.+\/V0\/Run-([0-9]+)\_([a-zA-Z0-9]+)\_case08([a-z\*])', pklDataPath)
runNum = int(pklNmMtch.group(1))
caseLet = pklNmMtch.group(3).replace('*','ALL')
inst = pklNmMtch.group(2)
simType = 'CC8%s_Run%02d_SZAgt%d' % (caseLet, runNum, szaMin)
# simType = 'G5NR' 
version = 'PLRA-Oct2023-%s-%s-%s-V06' % (inst, orb, simType) # for PDF file names

modeIndFwd = [0,2]
modeIndBck = [0]
# waveSeries = [0,3,5,0,3,0] if inst=='3mi' else [0,2,4,0,2,0]
# gvSeries = ['aod', 'aod', 'aod', 'aaod', 'aaod', 'reff']
waveSeries = [0,2,4,0,2,0,2,4]
gvSeries = ['aod', 'aod', 'aod', 'aaod', 'aaod', 'ssa', 'ssa', 'ssa']
# waveSeries = [0,0,0,2,4,0,2,4,0,2,4]
# gvSeries = ['aod','aaod', 'k', 'k', 'k', 'n', 'n', 'n']
# waveSeries = [0,3,5]
# gvSeries = ['aodf', 'aodf', 'aodf']
# waveSeries = [5]
# gvSeries = ['pm25']


# CDF error plot
figC, axC = plt.subplots(1,1, figsize=(6,6))
axC.grid()

if pklDataPath is not None:
    simBase = simulation(picklePath=pklDataPath)
    print('Loaded from %s - %d' % (pklDataPath, len(simBase.rsltBck)))
    if 'reff' in gvSeries:
        print('Calculcating fine-mode effective radii')
        simBase._addReffMode(0.5, True) # reframe with cut at 1 micron diameter
simBase.rsltFwd = np.asarray(simBase.rsltFwd)
simBase.rsltBck = np.asarray(simBase.rsltBck)
print('--')


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

# apply convergence filter
# simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
keepInd = np.logical_and(keepInd, [rb['sza'][0,0]>=szaMin for rb in simBase.rsltBck])
print('%d/%d fit surface type %s, convergence filter and sza≥%3.0f' % (keepInd.sum(), len(simBase.rsltBck), surf2plot, szaMin))


GVlegTxt = []
prctInEE = []
for waveInd, gv in zip(waveSeries, gvSeries):
    if varVsAOD:
        fig, ax = plt.subplots(1,2, figsize=(7.7,3.5))
    else:
        fig, ax = plt.subplots(1,1, figsize=(5.1,4.5))
        ax = [ax]
    ax[0].locator_params(nbins=3)

    wavelng = simBase.rsltFwd[0]['lambda'][waveInd]
    waveName = 'UV' if wavelng<0.4 else 'VIS' if wavelng<0.7 else 'NIR'
    print('Showing %s results for %s (%5.3f μm)' % (gv, waveName, wavelng))
    
    logScatPlot=False
    EEttlTxt = '%s, %s, %s' % (inst, orb, simType)
    if gv=='aod': # AOD Total
        logScatPlot = True
        ylabel = 'AOD (λ=%4.2fμm)' % wavelng
        true = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
        minVar = 0.01
        maxVar = 3
        aodMin = 0.0 # does not apply to AOD plot
        # EE_fun = lambda t : 0.02+0.05*t
        # EEttlTxt = EEttlTxt + ', EE=±0.02+0.05*τ'
        EE_fun = lambda t : 0.03+0.1*t
        EEttlTxt = EEttlTxt + ', EE=±(0.03+0.1*τ)'
        GVlegTxt.append('AOD-%s' % waveName)
    elif gv=='ssa': # SSA Total
        logScatPlot = False
        ylabel = 'SSA (λ=%4.2fμm)' % wavelng
        true = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rf['ssa'][waveInd] for rf in simBase.rsltBck])[keepInd]
        minVar = 0.8
        maxVar = 1
        aodMin = 0.0 # does not apply to AOD plot
        # EE_fun = lambda t : 0.02+0.05*t
        # EEttlTxt = EEttlTxt + ', EE=±0.02+0.05*τ'
        EE_fun = lambda t : 0.03
        EEttlTxt = EEttlTxt + ', EE=±0.03'
        GVlegTxt.append('SSA-%s' % waveName)
    elif gv=='aodf': # AOD Fine
        ylabel = 'AOD_fine (λ=%4.2fμm)' % wavelng
        true = np.asarray([rf['aodMode'][fineIndFwd, waveInd].sum() for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rf['aodMode'][fineIndBck, waveInd].sum() for rf in simBase.rsltBck])[keepInd]
        maxVar = None
        aodMin = 0.0 # does not apply to AOD plot
        EE_fun = lambda t : 0.03+0.1*t
        EEttlTxt = EEttlTxt + ', EE=±(0.03+0.1*τ)'
        GVlegTxt.append('AODF-%s' % waveName)
    elif gv=='aaod': # AAOD
        logScatPlot = True
        ylabel = 'AAOD (λ=%4.2fμm)' % wavelng
        true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
        minVar = 0.0005
        maxVar = 0.3
        # maxVar = None
        aodMin = 0.00001
#         EE_fun = lambda t : np.maximum(0.003, t*0.3) # NEEDS updating 
#         EEttlTxt = EEttlTxt + ', EE=max(0.003, 30%)'
#         EE_fun = lambda t : 0.003 + t*0.3 # NEEDS updating 
#         EEttlTxt = EEttlTxt + ', EE=0.003 + 30%'
        EE_fun = lambda t : 0.003 + t*0.5 # NEEDS updating 
        EEttlTxt = EEttlTxt + ', EE=0.003 + 50%'
        GVlegTxt.append('AAOD-%s' % waveName)
    elif gv=='pm25': # AAOD
        logScatPlot = True
        ylabel = 'PM2.5 (μg/m3)'
        true = calculatePM(simBase.rsltFwd[keepInd], alt=2)
        rtrv = calculatePM(simBase.rsltBck[keepInd], alt=2)
#         true = calculatePM(simBase.rsltFwd[keepInd], alt=800)
#         rtrv = calculatePM(simBase.rsltBck[keepInd], alt=800)
        maxVar = 180
        aodMin = 0.0
        EE_fun = lambda t : np.maximum(5, t*0.1)
        EEttlTxt = EEttlTxt + ', EE=max(5 μg/m3, 10%)'
        GVlegTxt.append('PM2.5 (μg/m3)')

    # 1-SSA
    # ylabel = 'Coalbedo (λ=%4.2fμm)' % wavelng
    # true = np.asarray([(1-rf['ssa'][waveInd]) for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([(1-rb['ssa'][waveInd]) for rb in simBase.rsltBck])[keepInd]
    # maxVar = 0.2
    # aodMin = 0.3 # does not apply to AOD plot
    # ANGSTROM
    # ylabel = 'AE (%4.2f/%4.2f μm)' % (wavelng, simBase.rsltFwd[0]['lambda'][waveInd2])
    # aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    # aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
    # logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
    # true = -np.log(aod1/aod2)/logLamdRatio
    # aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
    # aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
    # rtrv = -np.log(aod1/aod2)/logLamdRatio
    # maxVar = 2.25
    # aodMin = 0.05 # does not apply to AOD plot
    # g
    # ylabel = 'Asym. Param. (λ=%4.2fμm)' % wavelng
    # true = np.asarray([(rf['g'][waveInd]) for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([(rb['g'][waveInd]) for rb in simBase.rsltBck])[keepInd]
    # maxVar = None
    # aodMin = 0.15 # does not apply to AOD plot
    # vol FMF
    # ylabel = 'Submicron Volum. Frac.'
    # def fmfCalc(r,dvdlnr):
    #     cutRadius = 0.5
    #     fInd = r<=cutRadius
    #     logr = np.log(r)
    #     return np.trapz(dvdlnr[fInd],logr[fInd])/np.trapz(dvdlnr,logr)
    # true = np.asarray([fmfCalc(rf['r'], rf['dVdlnr']) for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([fmfCalc(rb['r'][0,:], rb['dVdlnr'].sum(axis=0)) for rb in simBase.rsltBck])[keepInd]
    # maxVar = None
    # aodMin = 0.1 # does not apply to AOD plot
    # SPH
    # ylabel = 'Volum. Frac. Spherical'
    # true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[keepInd]
    # maxVar = None
    # aodMin = 0.1 # does not apply to AOD plot
    elif gv=='brdf': # k
        ylabel = 'BRDF [ISO]  (λ=%4.2fμm)' % wavelng
        true = np.asarray([rf['brdf'][0,waveInd] for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rf['brdf'][0,waveInd] for rf in simBase.rsltBck])[keepInd]
        maxVar = None
        aodMin = 0.2 # does not apply to AOD plot
        logScatPlot=False # This is weird with 2d hist binning
        EE_fun = lambda t : t*0.1
        EEttlTxt = EEttlTxt + ', EE=10%'
        GVlegTxt.append('BRDF[ISO]-%s' % waveName)
    elif gv=='k': # k
        ylabel = 'AOD Wghtd. IRI (λ=%4.2fμm)' % wavelng
#         true = np.asarray([rf['k'][waveInd] for rf in simBase.rsltFwd])[keepInd]
#         rtrv = np.asarray([aodWght(rf['k'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
        true = np.asarray([aodWght(rf['k'][modeIndFwd,waveInd], rf['aodMode'][modeIndFwd,waveInd]) for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.squeeze(np.asarray([rb['k'][modeIndBck,waveInd] for rb in simBase.rsltBck])[keepInd])
        maxVar = None
        aodMin = 0.2 # does not apply to AOD plot
        logScatPlot=False # This is weird with 2d hist binning
        EE_fun = lambda t : t*0.5
        EEttlTxt = EEttlTxt + ', EE=50%'
        GVlegTxt.append('IRI-%s' % waveName)
    elif gv=='n': # n
        ylabel = 'AOD Wghtd. RRI (λ=%4.2fμm)' % wavelng
#         true = np.asarray([rf['n'][waveInd] for rf in simBase.rsltFwd])[keepInd]
        true = np.asarray([aodWght(rf['n'][modeIndFwd,waveInd], rf['aodMode'][modeIndFwd,waveInd]) for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.squeeze(np.asarray([rb['n'][modeIndBck,waveInd] for rb in simBase.rsltBck])[keepInd])
        maxVar = None
        aodMin = 0.1 # does not apply to AOD plot
        EE_fun = lambda t : 0.02
        EEttlTxt = EEttlTxt + ', EE=0.02'
        GVlegTxt.append('RRI-%s' % waveName)
    elif gv=='reff': # REFF
        ylabel = 'Fine Mode Effective Radius'
        true = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltFwd])[keepInd]
        rtrv = np.asarray([rf['rEffMode'][0] for rf in simBase.rsltBck])[keepInd]
        maxVar = 0.09
        maxVar = 0.31
#         rtrv[rtrv>maxVar] = maxVar+0.01
        aodMin = 0.005  # will be for fine mode given fineAOD=True below
        EE_fun = lambda t : 0.05
        EEttlTxt = EEttlTxt + ', EE=0.05μm'
        GVlegTxt.append('rEff_fine')
    else:
        assert False, 'gv %s is not valid' % gv
    if fineAOD:
        trueAOD = np.asarray([rf['aodMode'][fineIndFwd, waveIndAOD].sum() for rf in simBase.rsltFwd])[keepInd]
    else:
        trueAOD = np.asarray([rf['aod'][waveIndAOD] for rf in simBase.rsltFwd])[keepInd]

#     cmap = plt.cm.YlOrBr
    cmap = plt.cm.Reds
    # Scatter Plot
    vldI = np.logical_and(trueAOD>=aodMin, trueAOD<=aodMax)
    if maxVar is None:
        findMaxVar = True
        maxVar = np.max(true[vldI])*1.1
    else:
        findMaxVar = False
    if minVar is None:
        minVar = np.min(true[vldI])-0.1*np.mean(true[vldI])
        if minVar < 0.001 and logScatPlot: minVar=0.001
    
    ax[0].plot([minVar,maxVar], [minVar,maxVar], 'k', linewidth=LW121)
    EE_color = [0.5,0.5,0.5]
    if logScatPlot:
        errX = np.exp(np.linspace(np.log(minVar),np.log(maxVar),200))
    else:
        errX = np.linspace(minVar,maxVar,200)
    errX-EE_fun(errX)
    ax[0].plot(errX, errX+EE_fun(errX), '--', color=EE_color, linewidth=LW121)
    ax[0].plot(errX, errX-EE_fun(errX), '--', color=EE_color, linewidth=LW121)
#     ax[0].plot([minVar,maxVar], [minVar-EE_fun(minVar),maxVar-EE_fun(maxVar)], '--', color=EE_color, linewidth=LW121)
#     ax[0].plot([minVar,maxVar], [minVar+EE_fun(minVar),maxVar+EE_fun(maxVar)], '--', color=EE_color, linewidth=LW121)
    if hist2D:
        cnt = ax[0].hist2d(true[vldI], rtrv[vldI], (Nbins,Nbins), norm=mpl.colors.LogNorm(), cmap=cmap)
        cnt[3].set_edgecolor("face")
    else:
        ax[0].scatter(true[vldI], rtrv[vldI], c='r', s=scatSize, alpha=scatAlpha)
    ax[0].set_xlabel('Simulated True %s' % ylabel)
    ax[0].set_ylabel('Retrieved %s' % ylabel)
    if logScatPlot:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
    ax[0].set_xlim(minVar,maxVar)
    ax[0].set_ylim(minVar,maxVar)

    Rcoef = np.corrcoef(true[vldI], rtrv[vldI])[0,1]
    diffs = rtrv[vldI] - true[vldI]
    RMSE = np.sqrt(np.median(diffs**2))
    bias = np.mean(diffs)
    inEE = np.sum(np.abs(diffs) < (EE_fun(true[vldI])))/sum(vldI)*100
    prctInEE.append(inEE)
    frmt = 'R=%5.3f\nRMS=%5.3f\nbias=%5.3f\nIn EE=%%%4.1f'
    aodStr = 'AOD_fine' if fineAOD else 'AOD'
    tHnd = ax[0].annotate('N=%4d\n(%s>%g)' % (sum(vldI), aodStr, aodMin), xy=(0, 1), xytext=(190, -190), va='top', xycoords='axes fraction',
                textcoords='offset points', color=clrText, fontsize=FS)
    textstr = frmt % (Rcoef, RMSE, bias, inEE)
    tHnd = ax[0].annotate(textstr, xy=(0, 1), xytext=(5.5, -4.5), va='top', xycoords='axes fraction',
                        textcoords='offset points', color=clrText, fontsize=FS)

    # Cumulative error
    normedErr = np.abs(diffs)/EE_fun(true[vldI])
    normedErr[normedErr > maxNrmErr] = maxNrmErr
    #     n, bins, patches = axC.hist(normedErr, 5*int(np.sqrt(vldI.sum())), histtype='step', density=1, cumulative=True)
    n, bins, patches = axC.hist(normedErr, 1000, histtype='step', density=1, cumulative=True)
    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Variable Vs AOD
    if varVsAOD:
        vldI = trueAOD<=aodMax
        # rtrv = true + np.random.normal(0,0.1,len(true))
        err = errFun(true[vldI], rtrv[vldI])
        Ndx = 10
        dx = aodMax/round(Ndx)
        aodsX = np.r_[0:aodMax:dx]
        aodsX = np.linspace(0, aodMax - dx, round(Ndx))
        rmseOfX = [(np.sqrt(np.mean(err[np.logical_and(trueAOD[vldI]>x, trueAOD[vldI]<(x+dx))]**2))) for x in aodsX]
        # rmseOfX = [np.percentile(err[np.logical_and(trueAOD[vldI]>x, trueAOD[vldI]<(x+dx))], 84) for x in aodsX]
        if findMaxVar:
            minVar = np.min(err)
            maxVar = np.max(err)
        else:
            minVar = 0.01 if logScatPlot else 0
        ax[1].set_ylabel('Error in ' + ylabel)
        ax[1].set_xlabel('True %s (λ=%4.2fμm)' % (aodStr, simBase.rsltFwd[0]['lambda'][waveIndAOD]))
        cnt = ax[1].hist2d(trueAOD[vldI], err, (Nbins,Nbins), norm=mpl.colors.LogNorm(), cmap=cmap)
        cnt[3].set_edgecolor("face")
        # ax[0].plot(aodsX,rmseOfX, '-', color='m', alpha=0.99997, linewidth=2)
        ax[1].plot([aodMin,aodMin],[minVar, maxVar], '--', color='k', alpha=0.4, linewidth=2)
        ax[1].set_xlim(0, aodMax)
        ax[1].set_xticks(np.r_[0:aodMax:0.2])
        ax[1].set_ylim(minVar, maxVar)
        tHnd = ax[1].annotate('N=%4d\n(All AODs)' % sum(vldI), xy=(0, 1), xytext=(90, -4.5), va='top', xycoords='axes fraction',
                    textcoords='offset points', color=clrText, fontsize=FS)

    ax[0].set_title(EEttlTxt, fontsize=FStitle)
    fig.tight_layout()
    
    if saveScatter:
        ylabelcln = 'AE' if 'AE' in ylabel else ylabel[0:-11]
        fn = 'ScatterPlots_AODgt%05.3f_%s_%dnm_%s.pdf' % (aodMin, ylabelcln, (wavelng*1000), version)
        fig.savefig(os.path.join(plotSaveDir, fn))
        print('Plot saved as %s' % fn)

axC.set_xlabel('|Normalized GV Error|', fontsize=12)
axC.set_ylabel('C.D.F.', fontsize=12)
axC.legend(GVlegTxt, loc='lower right')
# axC.plot([1,1], [0, 0.65], '..', color=[0.5,0,0], alpha=0.4)
# axC.plot([0,1], [0.65, 0.65], '-', color=[0.5,0,0], alpha=0.4)
axC.plot([1], [0.65], '.', markersize=10, color=[0.5,0,0], alpha=0.4)
axC.set_xlim(0,maxNrmErr)
axC.set_ylim(0,1)
axC.set_yticks([0,0.25, 0.5, 0.65, 0.75, 1])
axC.set_title(version[:-4], fontsize=FStitle)

fn = 'CDFPlots_AODgt%05.3f_%s.pdf' % (aodMin, version)
figC.savefig(os.path.join(plotSaveDir, fn))
print('CDF Plot saved as %s' % fn)

print('Percent in EE array for histogram script:')
print(GVlegTxt)
print(prctInEE)

# plt.ion()
# plt.show()
