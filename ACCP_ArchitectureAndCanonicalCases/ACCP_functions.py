#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[500], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025,0.025,0.025]} # look at total and fine/coarse 
#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[1000], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025]} # only look at one mode (code below will work even if RMSE is calculated for fine/coarse too as long as n is listed under totVars)
#trgtRel = {'aod':0.05, 'rEffCalc':0.20, 'aodMode':0.05} # this part must be same for every mode but absolute component above can change
#trgt = {'aod':0.025, 'ssa':0.04, 'aodMode':[0.02,0.02], 'n':[0.025,0.025,0.025], 'ssaMode':[0.05,0.05], 'rEffCalc':0.05}
def normalizeError(simFwd, rmse, lInd, GVs, bias):
    trgt = {'aod':[0.02], 'ssa':[0.04], 'g':[0.02], 'aodMode':[0.02,0.02], 'ssaMode':[0.04,0.04], 'n':[0.025,0.025,0.025], 'LidarRatio':[0.0]} # look at total and fine/coarse 
    trgtRel = {'aod':0.05, 'aodMode':0.05, 'LidarRatio':0.25} # this part must be same for every mode but absolute component above can change
    i=0
    harvest = []
    harvestQ = []
    rmseVal = []
    for vr in GVs:
        for t,tg in enumerate(trgt[vr]):
            if vr in trgtRel.keys():     
                if np.isscalar(simFwd[vr]):
                    true = simFwd[vr]
                elif simFwd[vr].ndim==1:
                    true = simFwd[vr][lInd]
                else:
                    true = simFwd[vr][t,lInd]
                harvest.append((tg+trgtRel[vr]*true)/np.atleast_1d(rmse[vr])[t])
                harvestQ.append(np.sum((tg+trgtRel[vr]*true)>=np.abs(bias[vr][:,t]))/len(bias[vr][:,t]))
            else:
                harvest.append(tg/np.atleast_1d(rmse[vr])[t])
                harvestQ.append(np.sum(tg>=np.abs(bias[vr][:,t]))/len(bias[vr][:,t]))
            rmseVal.append(np.atleast_1d(rmse[vr])[t])
            i+=1
    return harvest, harvestQ, rmseVal