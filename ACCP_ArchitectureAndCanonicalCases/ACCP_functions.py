#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import os
from glob import glob
import csv
"""
  rmse['βext_PBL'] = np.sqrt(np.mean(prfRMSE['βext'][lowLayInd]**2))
    rmse['βext_FT'] = np.sqrt(np.mean(prfRMSE['βext'][upLayInd]**2))
    rmse['βextFine_PBL'] = np.sqrt(np.mean(prfRMSE['βextFine'][lowLayInd]**2))
    rmse['βextFine_FT'] = np.sqrt(np.mean(prfRMSE['βextFine'][upLayInd]**2))
    rmse['ssaPrf_PBL'] = np.sqrt(np.mean(prfRMSE['ssa'][lowLayInd]**2))
    rmse['ssaPrf_FT'] = np.sqrt(np.mean(prfRMSE['ssa'][upLayInd]**2))"""


def normalizeError(rmse, bias, true, enhanced=False):
    ssaTrg = 0.02 if enhanced else 0.04 # this (and rEff below) is for integrated quantities, not profiles!
    trgt = {'aod':0.0, 'aodMode_fine':0.0, 'aodMode_PBLFT':0.0, 'aodMode_finePBL':0.0, \
            'ssa':ssaTrg, 'ssaMode_fine':ssaTrg, 'ssaMode_PBLFT':ssaTrg, \
            'rEffCalc':0.1, 'rEffMode_fine':0.1, 'rEffMode_PBLFT':0.1, \
            'n':0.025, 'n_fine':0.025, 'n_PBLFT':0.025, \
            'k':0.002, 'k_fine':0.002, 'k_PBLFT':0.002, \
            'g':0.02, 'LidarRatio':0.0, \
            'βext_PBL':20.0, 'βext_FT':20.0, 'βextFine_PBL':20.0, 'βextFine_FT':20.0, \
            'ssaPrf_PBL':0.03, 'ssaPrf_FT':0.03} 
    trgtRel = {'LidarRatio':0.25, 'rEffCalc':0.1, 
               'βext_PBL':0.20, 'βext_FT':0.20, 'βextFine_PBL':0.20, 'βextFine_FT':0.20}
    aodTrgt = lambda τ: 0.02 + 0.05*τ # this needs to tolerate a 2D array
    qScore = dict()
    σScore = dict()
    mBias = dict()
    for av in set(rmse.keys()) & set(trgt.keys()):
        Nbck = bias[av].shape[0] # profile variables will be longer than other vars
        trNow = np.tile(true[av],(Nbck,1)) if true[av].shape[0]==1 else true[av]
        if av in trgtRel:
            trgtNow = np.array([max([trgt[av]], tr) for tr in trNow*trgtRel[av]]) # this might break if trgtRel is modal
        elif 'aod' in av:
            trgtNow = aodTrgt(trNow)
        else:
            trgtNow = trgt[av]*np.ones(bias[av].shape)
        # print('%s - ' % av, end='')
        # print(trgtNow)
        qScore[av] = np.mean(np.abs(bias[av])<=trgtNow, axis=0)
        σScore[av] = np.mean(trgtNow, axis=0)/rmse[av]
        mBias[av] = np.mean(bias[av], axis=0)
    return qScore, mBias, σScore

def prepHarvest(simFwd, rmse, lInd, GVs, bias):
    """ Note - we pull the indices that match below, ex. 'n_fine' key value has one element so we only pull the first of n_fine """
    trgt = {'aod':[0.04], 'ssa':[0.02], 'g':[0.02], 'aodMode_fine':[0.01], 'aodMode_PBLFT':[0.04], \
            'rEffMode_PBLFT':[0.1],'ssaMode_fine':[0.03], 'n':[0.02], 'n_fine':[0.02], \
                'LidarRatio':[0.0], 'rEffCalc':[0.1]} # look at total and fine/coarse 
    trgtRel = {'aod':0.05, 'aodMode_fine':0.05, 'LidarRatio':0.25, 'rEffCalc':0.0} # this part must be same for every mode but absolute component above can change
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
            try:
                harvestQ.append(np.sum(tg>=np.abs(bias[vr][:,0]))/len(bias[vr][:,0]))
            except:
                harvestQ.append(np.nan)
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
    
def selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nPCA, \
                        orbit=None, pcaVarPtrn='n_row_best_107sets_%s', verbose=False):
    """
    Pull scalars θs, φ (NADIR ONLY) from Pete's files at index specified by Feng's PCA
    There are two ways to select proper orbit data:
        1) rawAngleDir is directory with text files, orbit is None (will be determined from rawAngleDir string)
        2) rawAngleDir is parrent of directory with text files, orbit must be provided by the calling function
    rawAngleDir - directory of Pete's angle files for that particular orbit if orbit is None
                    if obrit provided, should be top level folder with both SS & GPM directories 
    PCAslctMatFilePath - full path of Feng's PCA results for indexing Pete's files
    nPCA - index of Feng's file, will pull index of Pete's data to extract
    orbit - 'GPM', 'SS', etc.; None -> try to extract it from rawAngleDir
    pcaVarPtrn='n_row_best_107sets_%s' - matlab variable, %s will be filled with orbit
    """
    import sys
    import os
    from glob import glob
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from MADCAP_functions import readPetesAngleFiles
    import scipy.io as spio
    if orbit is None: 
        if 'ss' in os.path.basename(rawAngleDir).lower():
            orbit = 'SS'
        elif 'gpm' in os.path.basename(rawAngleDir).lower():
            orbit = 'GPM'
    else:
        rawAngleDirPoss = glob(os.path.join(rawAngleDir, orbit.lower()+'*'+ os.path.sep))
        assert len(rawAngleDirPoss)==1, '%d angle directories found but should be exactly 1' % len(rawAngleDirPoss)
        rawAngleDir = rawAngleDirPoss[0]
    assert not orbit is None, 'Could not determine the orbit, which is needed to select mat file variable'
    angData = readPetesAngleFiles(rawAngleDir, nAng=10, verbose=verbose)
    pcaVar = pcaVarPtrn % orbit
    pcaData = spio.loadmat(PCAslctMatFilePath, variable_names=[pcaVar], squeeze_me=True)
    θs = max(angData['sza'][pcaData[pcaVar][nPCA]], 0.1) # (GRASP doesn't seem to be wild about θs=0)
    φAll = angData['fis'][pcaData[pcaVar][nPCA],:]
    φ = φAll[np.isclose(φAll, φAll.min(), atol=1)].mean() # take the mean of the smallest fis (will fail off-nadir)
    return θs, φ

def readKathysLidarσ(basePath, orbit, wavelength, instrument, concase, LidarRange, measType, verbose=False):
    """
    concase -> e.g. 'case06dDesert'
    measType -> Att, Ext, Bks [string] - Att is returned as relative error, all others absolute
    instrument -> 5, 6, 9 [int]
    wavelength -> λ in μm
    orbit -> GPM, SS
    basePath -> .../Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized
    """
    # resolution = '5kmH_500mV'
    resolution = '50kmH_500mV'
    # determine reflectance string
    wvlMap =   [0.355, 0.532, 1.064]
    if 'vegetation' in concase.lower(): # Vegetative
        rMap = [0.250, 0.140, 0.350]
    elif 'desert' in concase.lower(): # Desert
        rMap = [0.270, 0.250, 0.440] if instrument==9 else [0.270, 0.250, 0.490]
    else: # Ocean 
        rMap = [0.043, 0.050, 0.042] if instrument==9 else [0.043, 0.050, 0.050]
    mtchInd = np.nonzero(np.isclose(wavelength, wvlMap, atol=0.01))[0]
    assert len(mtchInd)==1, 'len(mtchInd)=%d but we expact exactly one match!' % len(mtchInd)
    Rstr = '%4.2f' % rMap[mtchInd[0]]
    # determine other aspects of the filename
    mtchData = re.match('^case([0-9]+)([a-z])', concase)
    assert mtchData, 'Could not parse canoncical case name %s' % concase
    caseNum = int(mtchData.group(1))
    caseLet = mtchData.group(2)
    # build full file path, load the data and interpolate
    fnPrms = (caseNum, caseLet, measType, 1000*wavelength, instrument, resolution, Rstr)
    searchPatern = 'case%1d%c_%s_%d*_L0%d_%s_D_C_0.*_R_%s*.csv' % fnPrms
    fnMtch = glob(os.path.join(basePath, orbit, searchPatern))
    if len(fnMtch)==2: # might be M1 and M2; if so, we drop M2
        fnMtch = (np.array(fnMtch)[[not '_M2.csv' in y for y in fnMtch]]).tolist()
    assert len(fnMtch)==1, 'We want one file but %d matched the patern .../%s/%s' % (len(fnMtch), orbit, searchPatern)
    if verbose: print('Reading lidar uncertainty data from: %s' % fnMtch[0])
    hgt = []; absErr = []
    with open(fnMtch[0], newline='') as csvfile:
        csvReadObj = csv.reader(csvfile, delimiter=',', quotechar='|')
        csvReadObj.__next__()
        for row in csvReadObj:
            hgt.append(float(row[0])*1000) # range km->m
            if measType == 'Att':
                absErr.append(float(row[3])) # relative err 
            else:
                absErr.append(float(row[3])/1) # abs err 1/km/sr -> 1/m/sr
    vldInd = ~np.logical_or(np.isnan(hgt), np.isnan(absErr))
    absErr = np.array(absErr)[vldInd]
    hgt = np.array(hgt)[vldInd]    
    absErr = absErr[np.argsort(hgt)]
    hgt = hgt[np.argsort(hgt)]
    return np.interp(LidarRange, hgt, absErr)



 