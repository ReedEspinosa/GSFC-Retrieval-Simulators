#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will summarize the accuracy of the state parameters retrieved in a simulation through box and whisker plots, among other methods """

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import copy
import pylab
import itertools
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../GRASP_scripts'))
from simulateRetrieval import simulation
import miscFunctions as mf
import ACCP_functions as af

# instruments = ['Lidar090','Lidar050','Lidar060', 'polar07', \
#                 'Lidar090+polar07','Lidar050+polar07','Lidar060+polar07'] # 7 N=231
instruments = ['Lidar090','Lidar050', 'polar07', \
                'Lidar090+polar07','Lidar050+polar07'] # 7 N=231
instruments = ['Lidar09','Lidar05', 'polar07', \
                'Lidar09+polar07','Lidar05+polar07'] # 7 N=231

    # instruments = ['Lidar09','Lidar05','Lidar06', 'polar07', \
#                 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=231
    # instruments = ['polar07', 'Lidar09+polar07'] # 7 N=231

    # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
# casLets = ['a', 'b', 'c', 'd', 'e', 'f','i']
# conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['Desert', 'Vegetation']]
# conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['','Desert', 'Vegetation']]
conCases = ['SPA'+surf for surf in ['','Desert', 'Vegetation']]
SZAs = [0] # 3
Phis = [0] # 1 -> N=18 Nodes
tauVals = [0.07,0.08,0.09,0.10,0.11] 
# tauVals = [0.07,0.08,0.09] 
N = len(SZAs)*len(conCases)*len(Phis)*len(tauVals)
# N = len(SZAs)*len(Phis)*len(instruments)*len(tauVals)
barVals = instruments # each bar will represent on of this category, also need to update definition of N above and the definition of paramTple (~ln75)

trgtλ =  0.5
χthresh= 3.0 # χ^2 threshold on points

def lgTxtTransform(lgTxt):
    if re.match('.*Coarse[A-z]*Nonsph', lgTxt): # conCase in leg
        return 'Fine[Sphere]\nCoarse[Nonsph]'
    if re.match('.*Fine[A-z]*Nonsph', lgTxt): # conCase in leg
        return 'Fine[Nonsph]\nCoarse[Sphere]'
    if 'misr' in lgTxt.lower(): # instrument in leg
        return lgTxt.replace('Misr','+misr').replace('Polar','+polar').upper()
    if 'lidar' in lgTxt.lower() or 'polar' in lgTxt.lower(): # instrument in leg
        return lgTxt
    return 'Fine[Sphere]\nCoarse[Sphere]' # conCase in leg

    
#totVars = ['aod', 'ssa', 'n', 'aodMode_fine', 'ssaMode_fine', 'n_fine', 'rEffCalc','g','height'] # must match to keys in rmse dict
totVars = np.flipud(['aod', 'ssa', 'n', 'rEffCalc', 'aodMode_PBLFT',  'rEffMode_PBLFT'])
# totVars = np.flipud(['aod', 'ssa', 'n', 'rEffCalc'])
totBiasVars = ['aod', 'ssa','aodMode_fine','n','rEffCalc'] # only used in Plot 4, if it is a multi dim array we take first index (aodmode and n)

saveStart = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/DRS_V09_'

plotD = False # PDFs of errors as a fuction of different variables

cm = pylab.get_cmap('viridis')

# def getGVlabels(totVars, modVars)
gvNames = copy.copy(totVars)
gvNames = ['$'+mv.replace('Mode','').replace('_fine','_{fine}').replace('rEffCalc','r_{eff}').replace('ssa', 'SSA').replace('aod','AOD').replace('PBLFT','{PBL}')+'$' for mv in gvNames]

Nvars = len(gvNames)
barLeg = []
totBias = dict([]) # only valid for last of whatever we itterate through on the next line
Nbar = len(barVals)
barVals = [y if type(y) is list else [y] for y in barVals] # barVals should be a list of lists
for barInd, barVal in enumerate(barVals):
    harvest = np.zeros([N*len(barVal), Nvars])
    # harvest = [[] for _ in range(Nvars)]
    runNames = []    
    for n in range(N*len(barVal)):
        # paramTple = list(itertools.product(*[instruments,conCases,SZAs,Phis,tauVals]))[n]
        paramTple = list(itertools.product(*[barVal,conCases,SZAs,Phis,tauVals]))[n]
        # saveStartn = saveStart
        # savePtrn = saveStart + '%s_%s_sza%d_phi%d_tFct%4.2f_V1.pkl' % paramTple[0:1]
        if ['intXXX'] == barVal:
            saveStartn = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/DRS_V02_'
        else:
            saveStartn = saveStart
        savePtrn = saveStartn + '%s_%s_orbGPM_tFct1.00_multiAngles_n*_nAngALL.pkl' % paramTple[0:2]
        savePtrn = saveStartn + '%s_%s_orbSS_tFct%4.2f_multiAngles_n*_nAngALL.pkl' % (paramTple[0:2] + (paramTple[4],))
        savePath = glob.glob(savePtrn)
        if not len(savePath)==1: assert False, 'Wrong number of files found (i.e. not one) for search string %s!' % savePtrn
        simB = simulation(picklePath=savePath[0])
        NsimsFull = len(simB.rsltBck)
        lInd = np.argmin(np.abs(simB.rsltFwd[0]['lambda']-trgtλ))
        Nsims = len(simB.rsltBck)
        print("<><><><><><>")
        print(savePath)
        print('AODf=%4.2f, AODc=%4.2f, Nsim=%d' % (simB.rsltFwd[0]['aodMode'][0,lInd], simB.rsltFwd[0]['aodMode'][1,lInd], Nsims))
        print(paramTple[0])
        print('Spectral variables for λ = %4.2f μm'% simB.rsltFwd[0]['lambda'][lInd])        
        simB.conerganceFilter(χthresh=χthresh, verbose=True)
        fineIndFwd = np.nonzero(['fine' in typ.lower() for typ in paramTple[1].split('+')])[0] 
        if len(fineIndFwd)==0: fineIndFwd = np.nonzero(simB.rsltFwd[0]['rv']<0.5)[0] # fine wasn't in case name, guess from rv
        assert len(fineIndFwd)>0, 'No obvious fine mode could be found in fwd data!'
        fineIndBck = [0]
        if ['polar07'] == barVal:
            rmse, bias, true = simB.analyzeSim(lInd, fineModesFwd=fineIndFwd, fineModesBck=fineIndBck)
            rmse['aodMode_PBLFT'] = np.nan
            rmse['rEffMode_PBLFT'] = np.nan
            # bias['aodMode_PBLFT'] = [[np.nan]]
            # bias['rEffMode_PBLFT'] = [[np.nan]]
        else:
            rmse, bias, true = simB.analyzeSim(lInd, fineModesFwd=fineIndFwd, fineModesBck=fineIndBck, hghtCut=2100)
            rmse['aodMode_PBLFT'] = rmse['aodMode_PBLFT'][0]
            rmse['rEffMode_PBLFT'] = rmse['rEffMode_PBLFT'][0]
        harvest[n, :], harvestQ, rmseVal = af.prepHarvest(simB.rsltFwd[0], rmse, lInd, totVars, bias) # TODO: replace with new normalizeError function 
        # for vInd, key in enumerate(totVars):
        #     harvest[vInd] = harvest[vInd] + np.abs(bias[key].T[0]).tolist()
        print('--------------------')
    # harvest = np.array(harvest).squeeze().T
    plt.rcParams.update({'font.size': 12})
    if barInd==0: 
        figB, axB = plt.subplots(figsize=(4.8,6)) # THIS IS THE BOXPLOT
        axB.plot([1,1], [0,5*(Nvars+1)], ':', color=0.65*np.ones(3)) # vertical line at unity
    pos = Nbar*np.r_[0:harvest.shape[1]]+0.7*barInd
    hnd = axB.boxplot(harvest, vert=0, patch_artist=True, positions=pos[0:Nvars], sym='.')
    [hnd['boxes'][i].set_facecolor(cm((barInd)/(Nbar))) for i in range(len(hnd['boxes']))]
    [hf.set_markeredgecolor(cm((barInd)/(Nbar))) for hf in hnd['fliers']]
    barLeg.append(hnd['boxes'][0])
axB.set_xscale('log')
axB.set_xlim([0.08,31])
axB.set_ylim([-0.8, Nbar*Nvars-1.5])
plt.sca(axB)
plt.yticks(Nbar*(np.r_[0:Nvars]+0.1*Nbar-0.2), gvNames)
lgTxt = [lgTxtTransform('%s' % τ) for τ in np.array(barVals)[:,0]]
# lgTxt = ['Perfect Model','Coarse Nonsph','Fine Nonsph', 'Unmodeled WLR','2 extra modes']
lgHnd = axB.legend(barLeg[::-1], ['%s' % τ for τ in lgTxt[::-1]], loc='center left', prop={'size': 9, 'weight':'bold'})
lgHnd.draggable()
axB.yaxis.set_tick_params(length=0)
figB.tight_layout()

sys.exit()
figSavePath = '/Users/wrespino/Synced/Presentations/ClimateRadiationSeminar_2020/figures/allFiveConditions_misrModis_noFineModeVarsShown_case.png' # % concases[0]
figB.savefig(figSavePath, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.05)
figB.savefig(figSavePath[:-3]+'pdf', dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.05)


# PLOT 4: print PDF as a fuction of variable type
if not plotD: sys.exit()
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
#print(np.mean([S(a,σ) for σ in harvest]))