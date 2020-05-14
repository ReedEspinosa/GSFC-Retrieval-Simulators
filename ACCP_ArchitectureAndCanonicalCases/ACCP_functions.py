#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[500], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025,0.025,0.025]} # look at total and fine/coarse 
#trgt = {'aod':[0.02], 'ssa':[0.03], 'g':[0.02], 'height':[1000], 'rEffCalc':[0.0], 'aodMode':[0.02,0.02], 'ssaMode':[0.03,0.03], 'n':[0.025]} # only look at one mode (code below will work even if RMSE is calculated for fine/coarse too as long as n is listed under totVars)
#trgtRel = {'aod':0.05, 'rEffCalc':0.20, 'aodMode':0.05} # this part must be same for every mode but absolute component above can change
#trgt = {'aod':0.025, 'ssa':0.04, 'aodMode':[0.02,0.02], 'n':[0.025,0.025,0.025], 'ssaMode':[0.05,0.05], 'rEffCalc':0.05}
def normalizeError(simFwd, rmse, lInd, GVs, bias):
    """ Note - we pull the indices that match below, ex. 'n_fine' key value has one element so we only pull the first of n_fine """
    trgt = {'aod':[0.01], 'ssa':[0.03], 'g':[0.02], 'aodMode_fine':[0.01], 'ssaMode_fine':[0.03], 'n':[0.02], 'n_fine':[0.02], 'LidarRatio':[0.0], 'rEffCalc':[0.0]} # look at total and fine/coarse 
    trgtRel = {'aod':0.05, 'aodMode_fine':0.05, 'LidarRatio':0.25, 'rEffCalc':0.10} # this part must be same for every mode but absolute component above can change
    assert np.all([not type(x) is list for x in trgtRel.values()]), 'All entries in trgtRel should be scalars (not lists)'
    i=0
    harvest = []
    harvestQ = []
    rmseVal = []
    for vr in GVs:
        tg = trgt[vr]
        if vr in trgtRel.keys():     
            rsltFwdVr = vr.replace('Mode_fine','') # NOTE: we work relative to the total in the modal variables (e.g. Δτ_fine = ±(0.2 + 0.05*τ_total)
            if np.isscalar(simFwd[rsltFwdVr]):
                true = simFwd[rsltFwdVr]
            elif simFwd[rsltFwdVr].ndim==1:
                true = simFwd[rsltFwdVr][lInd]
            else:
                true = simFwd[vr][0,lInd]
            harvest.append((tg+trgtRel[vr]*true)/np.atleast_1d(rmse[vr])[0])
            harvestQ.append(np.sum((tg+trgtRel[vr]*true)>=np.abs(bias[vr][:,0]))/len(bias[vr][:,0]))
        else:
            harvest.append(tg/np.atleast_1d(rmse[vr])[0])
            harvestQ.append(np.sum(tg>=np.abs(bias[vr][:,0]))/len(bias[vr][:,0]))
        rmseVal.append(np.atleast_1d(rmse[vr])[0])
        i+=1
    return harvest, harvestQ, rmseVal

def writeConcaseVars(rslt):
    """
     AODf, AODc, AODt, AODf, AODc, AODt, AODf, AODc, AODt, AODf, AODc, AODt, Å, Sf (sr) , Sc (sr) , St (sr), Sf (sr) , Sc (sr) , St (sr), Sf (sr) , Sc (sr) , St (sr), SSAf, SSAc, SSAt, ASYf, ASYc, ASYt
     This function assumes mode[0]->fine and mode[1]->coarse
    """
    valVect = []
    # 355 nm AODf, AODc, AODt, 532 nm AODf, AODc, AODt, 550 nm AODf, AODc, AODt, 1064 nm AODf, AODc, AODt
    for l in np.r_[355, 532, 550, 1064]/1000:
        lInd = np.isclose(rslt['lambda'], l, atol=1e-2).nonzero()[0][0]
        for m in range(2):
            valVect.append(rslt['aodMode'][m,lInd])
        valVect.append(rslt['aod'][lInd])
    # Å 440/870nm
    bInd = np.argmin(np.abs(rslt['lambda']-0.440))
    irInd = np.argmin(np.abs(rslt['lambda']-0.870))
    num = np.log(rslt['aod'][bInd]/rslt['aod'][irInd])
    denom = np.log(rslt['lambda'][irInd]/rslt['lambda'][bInd])
    valVect.append(num/denom)
    # Sf (sr) , Sc (sr) , St (sr), Sf (sr) , Sc (sr) , St (sr), Sf (sr) , Sc (sr) , St (sr)
    for l in np.r_[355, 532, 1064]/1000:
        lInd = np.isclose(rslt['lambda'], l, atol=1e-2).nonzero()[0][0]
        for m in range(2):
            valVect.append(rslt['LidarRatioMode'][m,lInd])
        valVect.append(rslt['LidarRatio'][lInd])
    # 532nm δf, δc, δt
    lInd = np.isclose(rslt['lambda'], 532/1000, atol=1e-2).nonzero()[0][0]
    for m in range(2):
        valVect.append(rslt['LidarDepolMode'][m,lInd])
    valVect.append(rslt['LidarDepol'][lInd])
    # 550nm SSAf, SSAc, SSAt, ASYf, ASYc, ASYt
    lInd = np.isclose(rslt['lambda'], 550/1000, atol=1e-2).nonzero()[0][0]
    for m in range(2):
        valVect.append(rslt['ssaMode'][m,lInd])
    valVect.append(rslt['ssa'][lInd])
    for m in range(2):
        valVect.append(rslt['gMode'][m,lInd])
    valVect.append(rslt['g'][lInd])
    print(', '.join([str(x) for x in valVect]))
    
    