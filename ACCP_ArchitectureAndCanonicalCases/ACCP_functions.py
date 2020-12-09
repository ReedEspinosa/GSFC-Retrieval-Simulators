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
    trgt = {'aod':0.0, 'aodMode_fine':0.0, 'aodMode_PBLFT':0.0, 'aodMode_finePBL':0.0,
            'ssa':ssaTrg, 'ssaMode_fine':ssaTrg, 'ssaMode_PBLFT':ssaTrg,
            'rEffCalc':0.0, 'rEffMode_fine':0.0, 'rEffMode_PBLFT':0.0,
            'n':0.025, 'n_fine':0.025, 'n_PBLFT':0.025,
            'k':0.002, 'k_fine':0.002, 'k_PBLFT':0.002,
            'g':0.02, 'LidarRatio':0.0,
            'βext_PBL':20.0, 'βext_FT':20.0, 'βextFine_PBL':20.0, 'βextFine_FT':20.0,
            'ssaPrf_PBL':0.03, 'ssaPrf_FT':0.03, 'LRPrf_PBL':0.0, 'LRPrf_FT':0.0}
    # currently this do not tolerate 2D variables (e.g. rEffMode_PBL which is Nbck x 2 will throw exception)
    trgtRel = {'LidarRatio':0.25, 'rEffCalc':0.1,
               'βext_PBL':0.20, 'βext_FT':0.20, 'βextFine_PBL':0.20, 'βextFine_FT':0.20,
               'LRPrf_PBL':0.25, 'LRPrf_FT':0.25, 'rEffMode_fine':0.1, 'rEffMode_PBL':0.1, 'rEffMode_FT':0.1}
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


def prepHarvest(score, GVs):
    """
    Functions to pull appropriate scores from dicts returned by normalizeError()
    score – one of qScore, mBias or σScore from normalizeError() above
    GVs – a list of variables to include in harvest; options are:
    'ssaMode_fine', 'rEffCalc', 'aodMode_PBL[FT]', 'n_PBL[FT]', 'rEffMode_PBL[FT]',
    'k_fine', 'n', 'n_fine', 'k', 'k_PBL[FT]', 'rEffMode_fine', 'aod', 'ssa',
    'ssaMode_PBL[FT]', 'LidarRatio', 'aodMode_fine
    '"""
    harvest = []
    for vr in GVs:
        ind = 1 if 'coarse' in vr.lower() or 'ft' in vr.lower() else 0
        vr = vr.replace('PBL', 'FT').replace('FT', 'PBLFT') # not a typo: PBL->FT->PBLFT OR FT->PBLFT
        harvest.append(score[vr][ind])
    return harvest


def findLayerSeperation(rsltFwd, defaultVal=None):
    """ Find seperation alt. between 2 layers; designed w/ canoncical cases in mind... mileage may vary"""
    if 'βext' not in rsltFwd:
        assert defaultVal is not None, 'We need either βext or a default value...'
        return defaultVal
    lowLayInd = rsltFwd['βext'][0,1:] - rsltFwd['βext'][-1,1:] < 1e-5 # skip the first (top) index cause it is always zero
    hghtCut = np.sum(rsltFwd['range'][0,1:]*np.gradient(np.float64(lowLayInd))) # second factor is array with 0.5 at bottom of top and 0.5 at top of bottom
    if hghtCut==0: hghtCut = rsltFwd['range'][0,0] # single layer case
    return hghtCut
    
def findFineModes(simB):
    fineIndFwd = np.nonzero(simB.rsltFwd[0]['rv']<0.5)[0] # fine wasn't in case name, guess from rv
    assert len(fineIndFwd)>0, 'No obvious fine mode could be found in fwd data!'
    fineIndBck = np.nonzero(simB.rsltBck[0]['rv']<0.5)[0] # fine wasn't in case name, guess from rv
    assert len(fineIndBck)>0, 'No obvious fine mode could be found in fwd data!'
    return fineIndFwd, fineIndBck

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
    orbit -> GPM, SS – argument is not used in (current) sept. assessment 
    basePath -> .../Remote_Sensing_Projects/A-CCP/lidarUncertainties/organized
    """
    # resolution = '5kmH_500mV'
    resolution = '5kmH_500mV'
    # determine other aspects of the filename
    mtchData = re.match('^case([0-9]+)([a-z][12]*)', concase)
    assert mtchData, 'Could not parse canoncical case name %s' % concase
    mtchCir = re.match('.*_Cirrus([0-9])$', basePath)
    caseType = 'case' if mtchCir is None else 'cir' + mtchCir.group(1)
    caseNum = int(mtchData.group(1))
    caseLet = mtchData.group(2) # can now have tailling number, e.g. 'f2' (required for sept. assessment)
    # build full file path, load the data and interpolate
    dayNghtChar = 'N' if '_night_' in basePath else 'D'
    fnPrms = (caseType, caseNum, caseLet, measType, 1000*wavelength, instrument, resolution, dayNghtChar)
    searchPatern = '%s%1d%s_%s_%d*_L0%d_%s_%c_C_0.*_R_*.csv' % fnPrms 
    instCaseDir = 'Lidar%02d' % instrument
    if instrument==9 and mtchCir is not None: instCaseDir = instCaseDir + '_RFI'
    if '_night_' not in basePath: instCaseDir = instCaseDir+('_desert' if 'desert' in concase.lower() else '_ocean')
    searchPath = os.path.join(basePath, instCaseDir, searchPatern)
    fnMtch = glob(searchPath)
#     if len(fnMtch)==2: # might be M1 and M2; if so, we drop M2
#         fnMtch = (np.array(fnMtch)[['_M2.csv' not in y for y in fnMtch]]).tolist()
    if len(fnMtch)==2: # might be M1 and M2; if so, we drop M1
        fnMtch = (np.array(fnMtch)[['_M1.csv' not in y for y in fnMtch]]).tolist()
    assert len(fnMtch)==1, 'We want one file but %d matched the patern ...%s' % (len(fnMtch), searchPath)
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
                absErr.append(float(row[2])/1000) # abs err 1/km/sr -> 1/m/sr
    vldInd = ~np.logical_or(np.isnan(hgt), np.isnan(absErr))
    absErr = np.array(absErr)[vldInd]
    hgt = np.array(hgt)[vldInd]
    absErr = absErr[np.argsort(hgt)]
    hgt = hgt[np.argsort(hgt)]
    return np.interp(LidarRange, hgt, absErr)


def readSharonsLidarProfs(fnPtrn, verbose=False):
    """
    Takes in a file pattern for glob that should return exactly one match (eg. below)
    Returns 1D Nlayer and  2D 4xNlayer arrays with altitudes and profiles for 4 modes, respectively
    The mode index ordering is [TOP_F, TOP_C, BOT_F, BOT_C]
    fnPtrn ='/Users/wrespino/Synced/A-CCP/Assessment_8K_Sept2020/Case_Definitions/simprofile_vACCP_case8k2_*.csv'
    """
    minProf = 3e-9
    fns = glob(fnPtrn)
    assert len(fns)==1, '%d files matching patern were found. We expect only one!' % len(fns)
    if verbose: print('Reading lidar profiles from: %s' % fns[0])
    data = np.genfromtxt(fns[0], dtype=np.float, names=True, delimiter=',', skip_header=1)
    data = data[1:] # first non-header row contains the units
    layAlt = data['MidAltitude']
    profs = np.full((4, layAlt.shape[0]), minProf)
    if not data['fine_radius'][0]==data['fine_radius'][-1]: # this is a two layer case
        layInds = data['fine_radius']==data['fine_radius'][-1]
        profs[0, layInds] = data['fine_N'][layInds]/data['fine_N'][layInds].max() # TOP_F
        profs[1, layInds] = data['coarse_N'][layInds]/data['coarse_N'][layInds].max() # TOP_C
    layInds = data['fine_radius']==data['fine_radius'][0]
    profs[2, layInds] = data['fine_N'][layInds]/data['fine_N'][layInds].max() # BOT_F
    profs[3, layInds] = data['coarse_N'][layInds]/data['coarse_N'][layInds].max() # BOT_C
    profs[profs<minProf] = 2*minProf # needed b/c sim_builder can produce zeros but doubled so we can distinguish zeros from outside the layer
    profs = np.fliplr(profs)
    layAlt = 1000*np.flipud(layAlt) # convert to m (this one is just 1D)
    profs = np.hstack([np.full((4,1), minProf), profs, profs[:,-1][:,None]]) # pad: zeros above, bottom value at ground
    layAlt = np.r_[layAlt[0]-np.diff(layAlt).mean(), layAlt, 1]
    return layAlt, profs


def _boundBackYamlSearch(typeList, nowPix, rngScale=1, rngScaleVol=1):
    """ helper function designed to be called exclusively by boundBackYaml"""
    from canonicalCaseMap import conCaseDefinitions
    lowLimit = 1e-6 # bounds well never go below this value
    allVals = {'n':[], 'k':[], 'sph':[], 'lgrnm':[]}
    for typeNow in typeList:
        valsTmp,_ = conCaseDefinitions(typeNow, nowPix)
        for key, value in allVals.items():
            value.append(valsTmp[key])
    lowVals = dict()
    uprVals = dict()
    lowVals['nCnst'] = np.min(np.asarray(allVals['n']),axis=0).min(axis=1)[:,None] - 0.01*rngScale
    uprVals['nCnst'] = np.max(np.asarray(allVals['n']),axis=0).max(axis=1)[:,None] + 0.02*rngScale
    lowVals['kCnst'] = np.min(np.asarray(allVals['k']),axis=0).min(axis=1)[:,None] - 0.0003*rngScale
    uprVals['kCnst'] = np.max(np.asarray(allVals['k']),axis=0).max(axis=1)[:,None] + 0.0003*rngScale
    lowVals['sph'] = np.min(allVals['sph'], axis=0)*(1 - 0.000001*rngScale)
    uprVals['sph'] = np.max(allVals['sph'], axis=0)*(1 + 0.000001*rngScale)
    lowVals['lgrnm'] = np.min(allVals['lgrnm'], axis=0)*(1 - 0.15*rngScale)
    uprVals['lgrnm'] = np.max(allVals['lgrnm'], axis=0)*(1 + 0.15*rngScale)
    for key in lowVals.keys(): lowVals[key][lowVals[key]<lowLimit] = lowLimit
    uprVals['sph'][uprVals['sph']>=1] = 0.999999
    return lowVals, uprVals

def boundBackYaml(baseYAML, caseStrs, nowPix, profs, verbose=False):
    """
    This function assumes runGRASP and is already in the python path.
    The mode index ordering is [TOP_F, TOP_C, BOT_F, BOT_C].
    """
    from canonicalCaseMap import splitMultipleCases, conCaseDefinitions
    from runGRASP import graspYAML
    dustAsm1 = np.any(['dust' in caseStr.lower() for caseStr,_ in splitMultipleCases(caseStrs)])
    ocenAsm2 = ~np.all(['desert' in caseStr.lower() for caseStr,_ in splitMultipleCases(caseStrs)])
    hsrlAsm3 = np.any([np.any(mv['meas_type']==36) for mv in nowPix.measVals])
    rosesNIP4 = 'NIP' in caseStrs
    lidarPresent = np.any([np.any(mv['meas_type']==31) for mv in nowPix.measVals]) # even HSRL has 31 at 1064nm
    assert not (lidarPresent and profs is None), 'prof is required if lidar data is present!'
    quadLayer = lidarPresent and (dustAsm1 or hsrlAsm3) and not rosesNIP4
    #   top layer
    msg = ' layer – search space includes: '
    if rosesNIP4:
        posTypes = ['smoke','pollution','plltdmrn','marine','dust']
        if verbose: print('Top'+msg+', '.join(posTypes))
        lowVals, uprVals = _boundBackYamlSearch(posTypes, nowPix, rngScale=1)
    elif dustAsm1:
        if verbose: print('Top'+msg+'dust')
        lowVals, uprVals = _boundBackYamlSearch(['dust'], nowPix, rngScale=4)
    elif ocenAsm2 and not hsrlAsm3: # this is the only layer
        posTypes = ['smoke','pollution','plltdmrn','marine'] # NOTE: we ignore that pollution is never in upper layer
        if verbose: print('Top'+msg+', '.join(posTypes))
        lowVals, uprVals = _boundBackYamlSearch(posTypes, nowPix, rngScale=1)
    else: # HSRL over ocean, or we are over land
        posTypes = ['smoke','pollution']
        if verbose: print('Top'+msg+', '.join(posTypes))
        lowVals, uprVals = _boundBackYamlSearch(posTypes, nowPix, rngScale=1)
    if quadLayer: # we retrieve a second, bottom layer
        if ocenAsm2:
            posTypes = ['plltdmrn','marine']
            if verbose: print('Bottom'+msg+', '.join(posTypes))
            lowValsB, uprValsB = _boundBackYamlSearch(posTypes, nowPix, rngScale=1)
        else: # land
            posTypes = ['smoke','pollution']
            if verbose: print('Bottom'+msg+', '.join(posTypes))
            lowValsB, uprValsB = _boundBackYamlSearch(posTypes, nowPix, rngScale=1)
        for key in lowVals: lowVals[key] = np.vstack([lowVals[key], lowValsB[key]])
        for key in uprVals: uprVals[key] = np.vstack([uprVals[key], uprValsB[key]])
    if lidarPresent: # set vertical profile, assume polar-only YAML template has 2 modes covering correct vertical range
        if quadLayer: # retrieving four layers, and we know the boundary
            lowVals['vrtProf'] = np.full(profs.shape, profs.min()/2)
            uprVals['vrtProf'] = 2*(profs>profs.min())+2*lowVals['vrtProf']
        else: # retrieving just two layers spanning full column
            lowVals['vrtProf'] = np.full((2, profs.shape[1]), profs.min()/2)
            uprVals['vrtProf'] = np.full((2, profs.shape[1]), 2.0)
            uprVals['vrtProf'][:,0] = 2*profs.min() # force top bin to zero
    # set bounds on vol
    vol = []
    for caseStr,loading in splitMultipleCases(caseStrs): # loop over all cases and add them together
        valsTmp = conCaseDefinitions(caseStr, nowPix)[0]
        vol.append(loading*valsTmp['vol'])
    vol = np.vstack(vol) if quadLayer else np.sum(vol,axis=0)
    Nλ = valsTmp['cxMnk'].shape[1] if 'cxMnk' in valsTmp else valsTmp['brdf'].shape[1]
    lowVals['vol'] = vol/10
    uprVals['vol'] = vol*10 
    # find initial values (midpoints) and write YAML
    initialVal = dict()
    for key in lowVals: initialVal[key] = (lowVals[key] + uprVals[key])/2
    yamlObj = graspYAML(baseYAML, newTmpFile=('BCK_%s' % caseStrs))
    yamlObj.setMultipleCharacteristics(initialVal, setField='value', Nlambda=Nλ) # must go first or min/max overwritten
    yamlObj.setMultipleCharacteristics(lowVals, setField='min')
    yamlObj.setMultipleCharacteristics(uprVals, setField='max')
    return yamlObj
