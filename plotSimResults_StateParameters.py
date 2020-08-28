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

instruments = ['Lidar090','Lidar050','Lidar060', 'polar07', \
                'Lidar090+polar07','Lidar050+polar07','Lidar060+polar07'] # 7 N=231

# casLets = ['a', 'b', 'c', 'd', 'e', 'f','i']
# conCases = ['case06'+caseLet+surf for caseLet in casLets for surf in ['','Desert', 'Vegetation']]
# conCases = ['SPA'+surf for surf in ['','Desert', 'Vegetation']]
conCases = ['case08%c%d' % (let,num) for let in map(chr, range(97, 112)) for num in [1,2]] # a1,a2,b1,..,o2 #30
tauVals = [1.0] 
# tauVals = [0.07,0.08,0.09] 
N = len(conCases)*len(tauVals)
barVals = instruments # each bar will represent on of this category, also need to update definition of N above and the definition of paramTple (~ln75)

trgtλ =  0.355
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
# totVars = np.flipud(['aod', 'ssa', 'n', 'rEff', 'LidarRatio'])
totVars = np.flipud(['aod', 'ssa', 'n'])

saveStart = '/Users/wrespino/Synced/Working/SIM17_SITA_SeptAssessment/TEST_V02_'

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
    runNames = []    
    for n in range(N*len(barVal)):
        paramTple = list(itertools.product(*[barVal,conCases,tauVals]))[n]
        instrument = paramTple[0].replace('GPM','')
        orbitStr = 'GPM' if 'GPM' in paramTple[0] else 'SS'
        caseStr = paramTple[1]
        tauVal = paramTple[2]
        savePtrn = saveStart + '%s_%s_tFct%4.2f_orb%s_multiAngles_n*_nAngALL.pkl' % (instrument,caseStr,tauVal,orbitStr)
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

plt.ion()
plt.show()

sys.exit()
figSavePath = '/Users/wrespino/Synced/Presentations/ClimateRadiationSeminar_2020/figures/allFiveConditions_misrModis_noFineModeVarsShown_case.png' # % concases[0]
figB.savefig(figSavePath, dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.05)
figB.savefig(figSavePath[:-3]+'pdf', dpi=600, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.05)


