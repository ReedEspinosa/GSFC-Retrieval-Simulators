#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will summarize the accuracy of the state parameters retrieved in a simulation through box and whisker plots, among other methods """

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import copy
import pylab
import itertools
from simulateRetrieval import simulation
import miscFunctions as mf
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases'))
import ACCP_functions as af

#simB.analyzeSim()
instruments = ['lidar09+polar07'] #1
#instruments = ['lidar05+polar07', 'lidar09+polar07'] #1
#instruments = ['lidar05+polar07'] #1
#instruments = ['PolOnly_lidar05+polar07', 'lidar09+polar07','lidar05+polar07', 'lidar05'] #1
#conCases = ['marine', 'pollution','smoke','marine+pollution','marine+smoke','Smoke+pollution'] #6
#conCases = ['variablefinenonsph','variablefinenonsph','variablefinenonsph','variablefinenonsph'] #6
#conCases = ['case02a','case02b','case02c','case03','case07']
conCases = ['variable']
#for caseLet in ['a','b','c','d','e','f']:
##    conCases.append('case06'+caseLet)
#    conCases.append('case06'+caseLet+'monomode')
#    if caseLet in ['e','f']:
#        conCases.append('case06'+caseLet+'nonsph')
#        conCases.append('case06'+caseLet+'monomode'+'nonsph') #21 total
#conCases = ['case06amonomode', 'case06bmonomode']
#SZAs = [0.1, 15, 30, 45, 60] # 3
SZAs = [0.1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # 3
Phis = [0] # 1 -> N=18 Nodes
#tauVals = [1.0] #
#tauVals = [0.03, 0.05, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35]
tauVals = [0.04, 0.08, 0.12, 0.18, 0.35] # NEED TO MAKE THIS CHANGE FILE NAME
#N = len(SZAs)*len(conCases)*len(Phis)*len(instruments)
N = len(SZAs)*len(conCases)*len(Phis)*len(tauVals)
N=36

gridPlots = False
#l = 3
#lVals = [0,3,5,10]
wavInd = 4

tag = 'Figure'

totBiasVars = ['aod', 'ssa', 'g','aodMode','n','rEffCalc'] # only used in Plot 4, if it is a multi dim array we take first index (aodmode and n)
#totVars = ['aod', 'ssa', 'rEffCalc','g','height']
totVars = ['aod', 'ssa', 'g', 'LidarRatio']
modVars = ['n', 'aodMode', 'ssaMode']  # reff should be profile

#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[500], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025,0.025,0.025]} # look at total and fine/coarse 
trgt = {'aod':[0.02], 'ssa':[0.04], 'g':[0.02], 'aodMode':[0.02,0.02], 'ssaMode':[0.04,0.04], 'n':[0.025,0.025,0.025], 'LidarRatio':[0.0]} # look at total and fine/coarse 
#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[1000], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025]} # only look at one mode (code below will work even if RMSE is calculated for fine/coarse too as long as n is listed under totVars)
#trgtRel = {'aod':0.05, 'rEffCalc':0.20, 'aodMode':0.05} # this part must be same for every mode but absolute component above can change
trgtRel = {'aod':0.05, 'aodMode':0.05, 'LidarRatio':[0.25]} # this part must be same for every mode but absolute component above can change
#trgt = {'aod':0.025, 'ssa':0.04, 'aodMode':[0.02,0.02], 'n':[0.025,0.025,0.025], 'ssaMode':[0.05,0.05], 'rEffCalc':0.05}
#aod fine Mode: 0.02+/-0.05AOD (same for total AOD)
# n: 0.025 (total)
# g is also in SATM...
# rEffCalc and it should be 20%
# ssa total 0.03

swapModes = False
printRslt = True

#simRsltFile = '/Users/wrespino/Synced/Working/SIM13_lidarTest/SIM43_lidar05+polar07_case-case06cmonomode_sza30_phi0_tFct1.00_V2.pkl'
saveStart = '/Users/wrespino/Synced/Working/SIM14_lidarPolACCP/SIM43V2_2mode_'

cm = pylab.get_cmap('viridis')

# def getGVlabels(totVars, modVars)
gvNames = copy.copy(totVars)
for mv in modVars:
    for i,nm in enumerate(['', '_{fine}','_{coarse}']): # HINT: this will need to change for >2 modes
        if i>0 or mv in ['n','k']: # n and k are formated differently, the first value is total, then fine and coarse
            gvNames.append(mv+nm)
gvNames = ['$'+mv.replace('Mode','').replace('rEffCalc','r_{eff}').replace('ssa', 'SSA').replace('aod','AOD')+'$' for mv in gvNames]

Nvars = len(gvNames)
barLeg = []
totBias = dict([]) # only valid for last of whatever we itterate through on the next line
Nbar = len(instruments)
for barInd, barVal in enumerate(instruments):
    lInd = 0 if barVal == 'lidar05' else wavInd
    figC, axC = plt.subplots(figsize=(10,6))
    axC.set_prop_cycle('color', plt.cm.Dark2(np.linspace(0,1,6)))
    harvest = np.zeros([Nvars, N])
    runNames = []    
    for n in range(N):
        paramTple = list(itertools.product(*[[barVal],conCases,SZAs,Phis,tauVals]))[n]
        if barVal == 'PolOnly_lidar05+polar07': 
            savePath = '/Users/wrespino/Synced/Working/SIM13_lidarTest/SIM42_%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple #SIM3_polar07_case-Smoke+pollution_sza30_phi0_tFct0.04_V2.pkl            
        else:
            savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple #SIM3_polar07_case-Smoke+pollution_sza30_phi0_tFct0.04_V2.pkl
        
        runNames.append('%s($θ_s=%d,φ=%d$)' % paramTple[1:4])
        runNames[-1] = runNames[-1].replace('pollution','POLL').replace('smoke','BB').replace('marine','MRN')
#        if os.path.exists(savePath): # HACK -- worst of all time
        simB = simulation(picklePath=savePath)
#        else:
#            print('=>'+str(n))
        if not type(simB.rsltFwd) is dict: simB.rsltFwd = simB.rsltFwd[0] # HACK [VERY BAD] -- remove when we fix this to work with lists 
        print("***** ", end=" ")
        print(simB.rsltFwd['aod'][4])
        Nsims = len(simB.rsltBck)
#        lInd = l+1 if 'lidar' in instruments[ind[0]] and l>0 else l
        print('---')
        print(runNames[-1])
        print(savePath)
        print('AODf=%4.2f, AODc=%4.2f, Nsim=%d' % (simB.rsltFwd['aodMode'][0,lInd], simB.rsltFwd['aodMode'][1,lInd], Nsims))
        print(paramTple[0])
        print('Spectral variables for λ = %4.2f μm'% simB.rsltFwd['lambda'][lInd])
        try:
#            assert False
            rmse, bias = simB.analyzeSim(lInd, modeCut=None, swapFwdModes=swapModes) # HINT: this will break in cases with differnt number of fwd and back modes
        except (ValueError, AssertionError):
            rmse, bias = simB.analyzeSim(lInd, modeCut=0.5) # HINT: this is much slower than the above
        print('---')
        for vr in totBiasVars:
            if n == 0: # allocate the array
                totBias[vr] = np.ones([N*Nsims, bias[vr].shape[1]])
            totBias[vr][n*Nsims:(n+1)*Nsims,:] = bias[vr]
        harvest[:,n], harvestQ, rmseVal = af.normalizeError(simB.rsltFwd, rmse, lInd, totVars+modVars, bias)
        if printRslt:
            print("Q and σ (col,pbl) --- SSA, g, Lidar, AODF, AOD, n:")
            for err in [np.array(harvestQ), np.array(rmseVal)]:
                print('%5f' % err['$SSA$'==np.array(gvNames)], end =",")
                print('%5f' % err['$SSA_{fine}$'==np.array(gvNames)], end =",")
                print('%5f' % err['$g$'==np.array(gvNames)], end =",")
#                print('---', end =",")
                print('%5f' % err['$LidarRatio$'==np.array(gvNames)], end =",")
                print('---', end =",")
                if 'case06a' in paramTple[1] or 'case06b' in paramTple[1]: # HACK -- all this should be improved...
                    print('%5f' % err['$AOD_{coarse}$'==np.array(gvNames)], end =",")
                elif 'case06c' in paramTple[1] or 'case06d' in paramTple[1]:
                    print('%5f' % err['$AOD$'==np.array(gvNames)], end =",")                
                elif 'variable' in paramTple[1] and not 'swap' in paramTple[1]:
                    print('%5f' % err['$AOD_{fine}$'==np.array(gvNames)], end =",")
                else:
                    print('---', end =",")
                if 'case06c' in paramTple[1] or 'case06d' in paramTple[1]:
                    print('%5f' % err['$AOD_{fine}$'==np.array(gvNames)], end =",")
                else:
                    print('---', end =",")
                print('---', end =",") # NONSPH would go in these two
                print('---', end =",")
                print('%5f' % err['$AOD$'==np.array(gvNames)], end =",")
                print('%5f' % err['$AOD_{fine}$'==np.array(gvNames)], end =",")
                print('%5f' % err['$n$'==np.array(gvNames)], end =",")
                print('%5f' % err['$n_{fine}$'==np.array(gvNames)])

        # PLOT 3: print PDF of AOD as a function of case
        aodDiffRng = 0.08
        kern = st.gaussian_kde(bias['ssa'][np.abs(bias['ssa'])<aodDiffRng])
        xAxisVals = np.linspace(-aodDiffRng, aodDiffRng, 500)                
        axC.plot(xAxisVals, kern.pdf(xAxisVals), '-.')
    axC.legend([cc[0:7] for cc in conCases])
    axC.set_title(barVal)
    if gridPlots: mf.gridPlot(runNames, gvNames, harvest)    
    plt.rcParams.update({'font.size': 14})
    if barInd==0: 
        figB, axB = plt.subplots(figsize=(4.8,6))
        axB.plot([1,1], [0,5*(Nvars+1)], ':', color=0.65*np.ones(3))
    pos = Nbar*np.r_[0:harvest.shape[0]]+0.7*barInd
    indGood = np.nonzero([not 'coarse' in gn for gn in gvNames])[0][::-1]
    hnd = axB.boxplot(harvest[indGood,:].T, vert=0, patch_artist=True, positions=pos[0:len(indGood)], sym='.')
    if barInd == 10:
        [hnd['boxes'][i].set_facecolor([0,0,1]) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor([0,0,1]) for hf in hnd['fliers']]
    elif barInd == 11:
        [hnd['boxes'][i].set_facecolor([1,0,0]) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor([1,0,0]) for hf in hnd['fliers']]
    else:
        [hnd['boxes'][i].set_facecolor(cm((barInd)/(Nbar))) for i in range(len(hnd['boxes']))]
        [hf.set_markeredgecolor(cm((barInd)/(Nbar))) for hf in hnd['fliers']]
    barLeg.append(hnd['boxes'][0])
axB.set_xscale('log')
axB.set_xlim([0.05,15])
axB.set_ylim([-0.8, Nbar*len(indGood)])
plt.sca(axB)
plt.yticks(Nbar*(np.r_[0:len(indGood)]+0.1*Nbar), [mv.replace('fine','PBL')for mv in np.array(gvNames)[indGood]])
#lgHnd = axB.legend(barLeg[::-1], ['%s' % τ for τ in instruments[::-1]], loc='center left')
#lgHnd = axB.legend(barLeg[::-1], ['Pol. Only', 'Pol+Lid09', 'Pol+Lid09', 'Lid05 Only'] , loc='center left')
#lgHnd.draggable()
axB.yaxis.set_tick_params(length=0)
figB.tight_layout()
figSavePath = '/Users/wrespino/Documents/'+tag+'_case-%s.png' % paramTple[1]
figB.savefig(figSavePath, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.1)

# PLOT 4: print PDF as a fuction of variable type
trgt = {'aod':0.03, 'ssa':0.03, 'g':0.02, 'aodMode':0.03, 'n':0.025} # look at total and fine/coarse aod=0.2
figD, axD = plt.subplots(2,3,figsize=(12,7))
for i,vr in enumerate(totBiasVars):
    curAx = axD[i//3,i%3]
    data = totBias[vr][:,0]
    dataCln = data[np.abs(data) < np.percentile(np.abs(data),98)]
    curAx.hist(dataCln,100)
    if vr in trgt:
        t = trgt[vr]
        curAx.plot([-trgt[vr], -trgt[vr]], [curAx.get_ylim()[0], curAx.get_ylim()[1]],'--k')
        curAx.plot([trgt[vr], trgt[vr]], [curAx.get_ylim()[0], curAx.get_ylim()[1]],'--k')
        print('%s - %d%%' % (vr, 100*np.sum(np.abs(dataCln) < trgt[vr])/len(dataCln)))
    curAx.set_yticks([], []) 
figD.tight_layout()




        
#a = [2,2,1,1,0,0,1,0,0,0]
#S = lambda a,σ: 5*np.sum(a*(1+σ**2)**(np.log2(4/5)))/np.sum(a)
#print(np.mean([S(a,σ) for σ in harvest.T]))