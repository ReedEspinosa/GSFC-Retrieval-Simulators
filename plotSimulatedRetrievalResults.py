#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:21:42 2019

@author: wrespino
"""
import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import numpy as np
import matplotlib.pyplot as plt
import copy
import pylab
from simulateRetrieval import simulation
#simB.analyzeSim()
instruments = ['polar07'] #1
#conCases = ['marine', 'pollution','smoke','marine+pollution','marine+smoke','Smoke+pollution'] #6
#conCases = ['marine+pollution','marine+smoke','Smoke+pollution']
conCases = ['marine','smoke','pollution'] #6
SZAs = [0, 30, 60] # 3
Phis = [0] # 1 -> N=18 Nodes
N = 18
tauVals = [0.04, 0.08, 0.12, 0.18, 0.35] # NEED TO MAKE THIS CHANGE FILE NAME

lInd = 3

totVars = ['aod', 'ssa', 'rEffCalc']
#modVars = ['aodMode', 'n', 'k', 'ssaMode', 'rEffMode', 'height']
modVars = ['n', 'aodMode', 'ssaMode']  # reff should be profile
trgt = {'aod':[0.02], 'ssa':[0.04], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.05,0.05], 'n':[0.025,0.025,0.025]}
trgtRel = {'aod':0.05, 'rEffCalc':0.20, 'aodMode':0.05} # this part must be same for every mode but absolute component above can change
#trgt = {'aod':0.025, 'ssa':0.04, 'aodMode':[0.02,0.02], 'n':[0.025,0.025,0.025], 'ssaMode':[0.05,0.05], 'rEffCalc':0.05}
#aod fine Mode: 0.02+/-0.05AOD (same for total AOD)
# n: 0.025 (total)
# g is also in SATM...
# rEffCalc and it should be 20%
# ssa total 0.03

saveStart = '/Users/wrespino/synced/Working/SIM3/SIM3_'

sigDef = 'score'
StrgtVal = 4
S = lambda σ: 5*(1+σ**2)**(np.log2(StrgtVal/5))

cm = pylab.get_cmap('viridis')

gvNames = copy.copy(totVars)
for mv in modVars:
    for i,nm in enumerate(['', '_{fine}','_{coarse}']):
        if i>0 or np.size(trgt[mv]) > 2: # HINT: this assumes two modes!
            gvNames.append(mv+nm)
gvNames = ['$'+mv.replace('Mode','').replace('rEffCalc','r_{eff}').replace('ssa', 'SSA').replace('aod','AOD')+'$' for mv in gvNames]
sizeMat = [1,1,1, len(instruments), len(conCases), len(SZAs), len(Phis)]
Nvars = np.hstack([x for x in trgt.values()]).shape[0]
Ntau = len(tauVals)
tauLeg = []
for tauInd, tau in enumerate(tauVals):
    harvest = np.zeros([Nvars, N])
    farmers = []    
    for n in range(N):
        ind = [n//np.prod(sizeMat[i:i+3])%sizeMat[i+3] for i in range(4)]
        paramTple = (instruments[ind[0]], conCases[ind[1]], SZAs[ind[2]], Phis[ind[3]], tau)
        savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V2.pkl' % paramTple #SIM3_polar07_case-Smoke+pollution_sza30_phi0_tFct0.04_V2.pkl
        farmers.append('%s($θ_s=%d,φ=%d$)' % paramTple[1:4])
        farmers[-1] = farmers[-1].replace('pollution','POLL').replace('smoke','BB').replace('marine','MRN')
        simB = simulation(picklePath=savePath)
        rmse = simB.analyzeSim(lInd)[0]
        i=0
        print(farmers[-1])
        print(simB.rsltFwd['aod'][3])
        for vr in totVars+modVars:
            for t,tg in enumerate(trgt[vr]):
                if vr in trgtRel.keys():
                    if np.isscalar(simB.rsltFwd[vr]):
                        true = simB.rsltFwd[vr]
                    elif simB.rsltFwd[vr].ndim==1:
                        true = simB.rsltFwd[vr][lInd]
                    else:
                        true = simB.rsltFwd[vr][t,lInd]
                    sigNorm = np.atleast_1d(rmse[vr])[t]/(tg+trgtRel[vr]*true)
                else:
                    sigNorm = np.atleast_1d(rmse[vr])[t]/tg
                if sigDef == 'glory':
                    harvest[i,n] = sigNorm
                elif sigDef == 'accp':
                    harvest[i,n] = 1/sigNorm
                elif sigDef == 'score':
                    harvest[i,n] = S(sigNorm)
                i+=1
    print('Spectral variables for λ = %4.2f μm'% simB.rsltFwd['lambda'][lInd])
#    plt.rcParams.update({'font.size': 10})
#    fig, ax = plt.subplots(figsize=(15,6), frameon=False)
#    im = ax.imshow(np.sqrt(harvest), 'seismic', vmin=0, vmax=2)
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.spines['bottom'].set_visible(False)
#    ax.spines['left'].set_visible(False)
#    # We want to show all ticks...
#    ax.set_xticks(np.arange(len(farmers)))
#    ax.set_yticks(np.arange(len(gvNames)))
#    # ... and label them with the respective list entries
#    ax.set_xticklabels(farmers)
#    ax.set_yticklabels(gvNames)
#    ax.xaxis.set_tick_params(length=0)
#    ax.yaxis.set_tick_params(length=0)
#    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")
#    # Loop over data dimensions and create text annotations.
#    for i in range(len(gvNames)):
#        for j in range(len(farmers)):
#            valStr = '%3.1f' % harvest[i, j]
#            clr = 'w' if np.abs(harvest[i, j]-1)>0.5 else 'k'
#    #        clr = np.min([np.abs(harvest[i, j]-1)**3, 1])*np.ones(3)
#            text = ax.text(j, i, valStr,
#                           ha="center", va="center", color=clr, fontsize=9)
#    fig.tight_layout()
     
    plt.rcParams.update({'font.size': 14})
    if tauInd==0: 
        figB, axB = plt.subplots(figsize=(6,6))
        trgtLine = StrgtVal if sigDef == 'score' else 1
        axB.plot([trgtLine,trgtLine], [0,5*(Nvars+1)], ':', color=0.65*np.ones(3))
    pos = Ntau*np.r_[0:harvest.shape[0]]+0.8*tauInd
    hnd = axB.boxplot(harvest.T, vert=0, patch_artist=True, positions=pos, sym='.')
    [hnd['boxes'][i].set_facecolor(cm(tauInd/Ntau)) for i in range(len(hnd['boxes']))]
    tauLeg.append(hnd['boxes'][0])
if sigDef == 'score':
    axB.set_xscale('linear')
    axB.set_xlim([0.,5])    
else:
    axB.set_xscale('log')
    axB.set_xlim([0.1,16])
axB.set_ylim([-0.7, Ntau*(len(gvNames)-0.1)])
plt.sca(axB)
plt.yticks(Ntau*(np.r_[1:(harvest.shape[0]+1)]-0.5), gvNames)
lgHnd = axB.legend(tauLeg[::-1], ['τ = %4.2f' % τ for τ in tauVals[::-1]], loc='center left')
lgHnd.draggable()
axB.yaxis.set_tick_params(length=0)
figB.tight_layout()

if sigDef == 'glory': 
    a = [2,2,1,1,0,0,1,0,0,0]
    S = lambda a,σ: 5*np.sum(a*(1+σ**2)**(np.log2(StrgtVal/5)))/np.sum(a)
    print(np.mean([S(a,σ) for σ in harvest.T]))
        
