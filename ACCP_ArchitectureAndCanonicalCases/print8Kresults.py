#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:06:44 2020

@author: wrespino
"""
import numpy as np
import os
import sys
from glob import glob
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../GSFC-GRASP-Python-Interface'))
from simulateRetrieval import simulation
from ACCP_functions import normalizeError, findLayerSeperation, findFineModes

instruments = ['polar07','polar07GPM','Lidar090+polar07GPM', 'Lidar090','Lidar050','Lidar060', \
               'Lidar090Night','Lidar050Night','Lidar060Night','Lidar090+polar07','Lidar050+polar07','Lidar060+polar07']
caseIDs = ['8%c%d' % (let,num) for let in map(chr, range(97, 112)) for num in [1,2]] # a1,a2,b1,..,o2 #30
destDir = '/Users/wrespino/Desktop/NGE_Sept_V01'
srcPath = '/Users/wrespino/Synced/Working/SIM17_SITA_SeptAssessment/'
filePatrn = 'DRS_V01_%s_case0%s_tFct1.00_orb*_multiAngles_n*_nAngALL.pkl' # % (instrument, caseID) [wildcards allowed]
lidarRefInst = 'Lidar090' # instrument to pull sim with fwd profile in polarimeter cases

χthresh = 2.2 # χ^2 threshold on points
forceχ2Calc = True
minSaved = 13
trgtλUV = 0.355
trgtλVis = 0.550
trgtλNIR = 1.064

comments = 'No comments'
V='V01'
RES='RES1'
cirrusNumber = 0 # 0->no cirrus, 1->cirrus 1,...
simType = 'DRS'
NNN='NGE'
frmStr = '%2d, %29s, %8.3f, %8.3f, %8.3f'
bad1 = 999 # bad data value
finePBLind = 2 # this is only for 4 fwd & 4 bck mode cases


def main():
    PathFrmt = os.path.join(srcPath, filePatrn)
    runStatus = []
    for instrument in instruments:
        SSPvals = [0,1,2] if not 'lidar' in instrument.lower() else [0]
        for caseID in caseIDs:
            for SSPval in SSPvals: # only runs once, unless polarimeter-only
                runStatus.append(run1case(instrument, caseID, PathFrmt, polOnlyPlat=SSPval))
    return runStatus
                
def run1case(instrument, caseID, PathFrmt, polOnlyPlat=0):
    TYP=simType + caseID.lower()
    PLTF = 'UNSET'
    if 'gpm' in instrument.lower():
        PLTF='SSG3'
    elif 'lidar05' in instrument.lower(): 
        PLTF='SSP0'
    elif 'lidar06' in instrument.lower(): 
        PLTF='SSP1'
    elif 'lidar09' in instrument.lower(): 
        PLTF='SSP2'
    if 'polar07' in instrument.lower():
        if 'lidar' in instrument.lower(): # both lidar and polarimeter
            OBS='NADBC%1d' % cirrusNumber
        else:
            if 'GPM' not in instrument: PLTF='SSP%d' % polOnlyPlat
            OBS='ONDPC%1d' % cirrusNumber
    else:
        dayNightChar = 'N' if 'night' in instrument.lower() else 'D'
        OBS='NA%cLC%1d' % (dayNightChar, cirrusNumber)
    trgtFN = '_'.join([NNN,TYP,PLTF,OBS,RES,V]) + '.csv'
    # find and load file
    simA = load1file(PathFrmt % (instrument, caseID))
    simB = None if 'lidar' in instrument else load1file(PathFrmt % (lidarRefInst, caseID))
    trgtPath = os.path.join(destDir, trgtFN)
    cntnts = ['%s, Comments: %s' % (trgtFN, comments)]
    print('Building conents of output file...')
    cntnts = buildContents(cntnts, simA, simB)
    with open(trgtPath, mode='wt', encoding='utf-8') as file:
        file.write('\n'.join(cntnts))  
    print('Output file save to %s' % os.path.basename(trgtPath))
    return trgtFN

def load1file(simRsltFile):
    print(simRsltFile)
    posFiles = glob(simRsltFile)
    if not len(posFiles)==1:
        print('¡¡¡Glob found %d files but we expect exactly 1!!!' % len(posFiles))
        print('We return the search pattern and move on to the next file...')
        print('<><><><><>')
        return 'FAILED: '+simRsltFile
    print('Loading %s...' % os.path.basename(posFiles[0]))
    return simulation(picklePath=posFiles[0])
    

def prep4normError(rmse, bs, tr, prfRMSE, prfBias, prfTrue, upLayInd, lowLayInd):
    lidScale = 1e6 # m-1 -> Mm-1
    # lower layer
    if 'aodMode' in rmse and len(rmse['aodMode'])==4:
        rmse['aodMode_finePBL'] = np.r_[rmse['aodMode'][finePBLind]]
        bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
        tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
    else:
        rmse['aodMode_finePBL'] = np.r_[bad1]
        bs['aodMode_finePBL'] = np.r_[bad1]
        tr['aodMode_finePBL'] = np.r_[bad1]
    rmse['βext_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['βext'][lowLayInd]**2))]*lidScale # THESE CONFUSE ME...
    rmse['βextFine_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine'][lowLayInd]**2))]*lidScale
    rmse['ssaPrf_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa'][lowLayInd]**2))]
    rmse['LRPrf_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['LR'][lowLayInd]**2))]
    bs['βext_PBL'] = prfBias['βext'][:,lowLayInd].reshape(-1,1)*lidScale
    bs['βextFine_PBL'] = prfBias['βextFine'][:,lowLayInd].reshape(-1,1)*lidScale
    bs['ssaPrf_PBL'] = prfBias['ssa'][:,lowLayInd].reshape(-1,1)
    bs['LRPrf_PBL'] = prfBias['LR'][:,lowLayInd].reshape(-1,1)
    tr['βext_PBL'] = prfTrue['βext'][:,lowLayInd].reshape(-1,1)*lidScale
    tr['βextFine_PBL'] = prfTrue['βextFine'][:,lowLayInd].reshape(-1,1)*lidScale
    tr['ssaPrf_PBL'] = prfTrue['ssa'][:,lowLayInd].reshape(-1,1)
    tr['LRPrf_PBL'] = prfTrue['LR'][:,lowLayInd].reshape(-1,1)
    # upper layer
    if prfTrue['βext'][:,upLayInd].max() < 1/lidScale: # no meaningful upper lay (never exceeds 1 Mm-1)
        nanVars = ['βext_FT', 'βextFine_FT',  'ssaPrf_FT'] 
        for vr in nanVars:
            rmse[vr] = np.r_[bad1]
            bs[vr] = np.r_[bad1]
            tr[vr] = np.r_[bad1]
        return rmse, bs, tr
    rmse['βext_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['βext'][upLayInd]**2))]*lidScale
    rmse['βextFine_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine'][upLayInd]**2))]*lidScale
    rmse['ssaPrf_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa'][upLayInd]**2))]
    rmse['LRPrf_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['LR'][upLayInd]**2))]
    bs['βext_FT'] = prfBias['βext'][:,upLayInd].reshape(-1,1)*lidScale
    bs['βextFine_FT'] = prfBias['βextFine'][:,upLayInd].reshape(-1,1)*lidScale
    bs['ssaPrf_FT'] = prfBias['ssa'][:,upLayInd].reshape(-1,1)
    bs['LRPrf_FT'] = prfBias['LR'][:,upLayInd].reshape(-1,1)
    tr['βext_FT'] = prfTrue['βext'][:,upLayInd].reshape(-1,1)*lidScale
    tr['βextFine_FT'] = prfTrue['βextFine'][:,upLayInd].reshape(-1,1)*lidScale
    tr['ssaPrf_FT'] = prfTrue['ssa'][:,upLayInd].reshape(-1,1)
    tr['LRPrf_FT'] = prfTrue['LR'][:,upLayInd].reshape(-1,1)
    return rmse, bs, tr

def buildString(num, name, qScr, mBs, rmse, key):
    if key in qScr and key in rmse and not rmse[key][0]==bad1:
        q = qScr[key][0]
        m = mBs[key][0]
        r = rmse[key][0]
    else:
        q = bad1
        m = bad1
        r = bad1
    vals = (num, name, q, m, r)
    return frmStr % vals

def buildContents(cntnts, simA, simB):
    
    # prep/filter this data and find λInds
    cntnts.append('GV#, GVname, QIscore, mean_unc, RMS')
    simA.conerganceFilter(χthresh=χthresh, forceχ2Calc=forceχ2Calc, verbose=True, minSaved=minSaved)
    lIndUV = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλUV))
    lIndVIS = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλVis))
    lIndNIR = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλNIR))
    
    if simB is None:
        rangeBins = None 
        hghtCut = findLayerSeperation(simA.rsltFwd[0])
    else: 
        rangeBins = simB.rsltFwd[0]['range'][0,:]
        hghtCut = findLayerSeperation(simB.rsltFwd[0])
    simA.addProfFromGuassian(rsltRange=rangeBins)
    lowLayInd = np.where(simA.rsltFwd[0]['range'][0,:]<hghtCut)[0]
    upLayInd = np.where(simA.rsltFwd[0]['range'][0,:]>=hghtCut)[0]
    if len(upLayInd)>1: upLayInd = upLayInd[upLayInd!=0] # the top index is always clean
    fineModeInd, fineModeIndBck = findFineModes(simA)
    #VIS
    rmseVis,bs,tr = simA.analyzeSim(lIndVIS, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, hghtCut=hghtCut)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndVIS, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, fwdSim=simB)
    rmseVis,bs,tr = prep4normError(rmseVis, bs, tr, prfRMSE, prfBias, prfTrue, upLayInd, lowLayInd)
    qScrVis, mBsVis, _ = normalizeError(rmseVis,bs,tr)
    qScrVis_EN, _0, _ = normalizeError(rmseVis,bs,tr, enhanced=True)
    #NIR
    rmseNIR,bs,tr = simA.analyzeSim(lIndNIR, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, hghtCut=hghtCut)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndNIR, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, fwdSim=simB)
    rmseNIR,bs,tr = prep4normError(rmseNIR, bs, tr, prfRMSE, prfBias, prfTrue, upLayInd, lowLayInd)
    qScrNIR, mBsNIR, _ = normalizeError(rmseNIR,bs,tr)
    #UV - find error stats for column and PBL and profiles 
    if simA.rsltFwd[0]['lambda'][lIndUV] < 0.4: # we have at least one UV channel
        rmseUV,bs,tr = simA.analyzeSim(lIndUV, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, hghtCut=hghtCut)
        prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndUV, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, fwdSim=simB)
        rmseUV,bs,tr = prep4normError(rmseUV, bs, tr, prfRMSE, prfBias, prfTrue, upLayInd, lowLayInd)
        qScrUV, mBsUV, _ = normalizeError(rmseUV,bs,tr)
        qScrUV_EN, _0, _ = normalizeError(rmseUV,bs,tr, enhanced=True)
        print('UV: %5.3f μm, VIS: %5.3f μm, NIR: %5.3f μm' % tuple(simA.rsltFwd[0]['lambda'][np.r_[lIndUV,lIndVIS,lIndNIR]]))
    else: # no UV available
        qScrUV = {key: np.r_[bad1] for key in qScrVis}
        mBsUV = {key: np.r_[bad1] for key in mBsVis}
        rmseUV = {key: np.r_[bad1] for key in rmseVis}
        qScrUV_EN = {key: np.r_[bad1] for key in qScrVis_EN}
        print('NO UV DATA, VIS: %5.3f μm, NIR: %5.3f μm' % tuple(simA.rsltFwd[0]['lambda'][np.r_[lIndVIS,lIndNIR]]))
    
    # build file contents, line-by-line
    cntnts.append(buildString(1,  'AAOD_l_UV_column', qScrUV, mBsUV, rmseUV, 'ssa'))
    cntnts.append(buildString(2,  'AAOD_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'ssaMode_PBLFT'))
    cntnts.append(buildString(3,  'AAOD_l_VIS_column', qScrVis, mBsVis, rmseVis, 'ssa'))
    cntnts.append(buildString(4,  'AAOD_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'ssaMode_PBLFT'))
    cntnts.append(buildString(5,  'AAOD_l_UV_column', qScrUV_EN, mBsUV, rmseUV, 'ssa'))
    cntnts.append(buildString(6,  'AAOD_l_UV_PBL', qScrUV_EN, mBsUV, rmseUV, 'ssaMode_PBLFT'))
    cntnts.append(buildString(7,  'AAOD_l_VIS_column', qScrVis_EN, mBsVis, rmseVis, 'ssa'))
    cntnts.append(buildString(8,  'AAOD_l_VIS_PBL', qScrVis_EN, mBsVis, rmseVis, 'ssaMode_PBLFT'))
    cntnts.append(buildString(9,  'ASYM_VIS', qScrVis, mBsVis, rmseVis, 'g'))
    cntnts.append(buildString(10, 'ASYM_NIR', qScrNIR, mBsNIR, rmseNIR, 'g'))
    cntnts.append(buildString(11, 'AEFRF_l_column', qScrVis, mBsVis, rmseVis, 'rEffMode_fine'))
    cntnts.append(frmStr %   (12, 'AEFRF_l_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (13, 'AEFRC_l_column', bad1, bad1, bad1))
    cntnts.append(frmStr %   (14, 'AEFRC_l_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(15, 'AE2BR_l_UV_column', qScrUV, mBsUV, rmseUV, 'LidarRatio'))
    cntnts.append(buildString(16, 'AE2BR_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'LidarRatioMode_PBLFT'))
    cntnts.append(buildString(17, 'AE2BR_l_VIS_column', qScrVis, mBsVis, rmseVis, 'LidarRatio'))
    cntnts.append(buildString(18, 'AE2BR_l_VIS_PBL', qScrUV, mBsUV, rmseUV, 'LidarRatioMode_PBLFT'))
    cntnts.append(buildString(19, 'AODF_l_VIS_column', qScrVis, mBsVis, rmseVis, 'aodMode_fine'))
    cntnts.append(buildString(20, 'AODF_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'aodMode_finePBL'))
    cntnts.append(frmStr %   (21, 'ANSPH_l_VIS_column', bad1, bad1, bad1))
    cntnts.append(frmStr %   (22, 'ANSPH_l_VIS_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(23, 'AOD_l_UV_column', qScrUV, mBsUV, rmseUV, 'aod'))
    cntnts.append(buildString(24, 'AOD_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'aodMode_PBLFT'))
    cntnts.append(buildString(25, 'AOD_l_VIS_column', qScrVis, mBsVis, rmseVis, 'aod'))
    cntnts.append(buildString(26, 'AOD_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'aodMode_PBLFT'))
    cntnts.append(buildString(27, 'AOD_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'aod'))
    cntnts.append(buildString(28, 'AOD_l_NIR_PBL', qScrNIR, mBsNIR, rmseNIR, 'aodMode_PBLFT'))
    cntnts.append(frmStr %   (29, 'APM25', bad1, bad1, bad1)) # TODO: if we have time we could add this
    cntnts.append(buildString(30, 'ARIR_l_UV_column', qScrUV, mBsUV, rmseUV, 'n'))
    cntnts.append(buildString(31, 'ARIR_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'n_PBLFT'))
    cntnts.append(buildString(32, 'ARIR_l_VIS_column', qScrVis, mBsVis, rmseVis, 'n'))
    cntnts.append(buildString(33, 'ARIR_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'n_PBLFT'))
    cntnts.append(buildString(34, 'ARIR_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'n'))
    cntnts.append(buildString(35, 'ARIR_l_NIR_PBL', qScrNIR, mBsNIR, rmseNIR, 'n_PBLFT'))
    cntnts.append(buildString(36, 'AIIR_l_UV_column', qScrUV, mBsUV, rmseUV, 'k'))
    cntnts.append(buildString(37, 'AIIR_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'k_PBLFT'))
    cntnts.append(buildString(38, 'AIIR_l_VIS_column', qScrVis, mBsVis, rmseVis, 'k'))
    cntnts.append(buildString(39, 'AIIR_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'k_PBLFT'))
    cntnts.append(buildString(40, 'AIIR_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'k'))
    cntnts.append(buildString(41, 'AIIR_l_NIR_PBL', qScrNIR, mBsNIR, rmseNIR, 'k_PBLFT'))
    cntnts.append(buildString(42, 'AABS_z_UV_profile_above_PBL', qScrUV, mBsUV, rmseUV, 'ssaPrf_FT'))
    cntnts.append(buildString(43, 'AABS_z_UV_profile_in_PBL', qScrUV, mBsUV, rmseUV, 'ssaPrf_PBL'))
    cntnts.append(buildString(44, 'AABS_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'ssaPrf_FT'))
    cntnts.append(buildString(45, 'AABS_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'ssaPrf_PBL'))
    cntnts.append(frmStr %   (46, 'AEFRF_z_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (47, 'AEFRF_z_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (48, 'AEFRC_z_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (49, 'AEFRC_z_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(50, 'AEXT_z_UV_profile_above_PBL', qScrUV, mBsUV, rmseUV, 'βext_FT'))
    cntnts.append(buildString(51, 'AEXT_z_UV_profile_in_PBL', qScrUV, mBsUV, rmseUV, 'βext_PBL'))
    cntnts.append(buildString(52, 'AEXT_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'βext_FT'))
    cntnts.append(buildString(53, 'AEXT_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'βext_PBL'))
    cntnts.append(buildString(54, 'AEXT_z_NIR_profile_above_PBL', qScrNIR, mBsNIR, rmseNIR, 'βext_FT'))
    cntnts.append(buildString(55, 'AEXT_z_NIR_profile_in_PBL', qScrNIR, mBsNIR, rmseNIR, 'βext_PBL'))
    cntnts.append(buildString(56, 'AE2BR_z_UV_profile_above_PBL', qScrUV, mBsUV, rmseUV, 'LRPrf_FT'))
    cntnts.append(buildString(57, 'AE2BR_z_UV_profile_in_PBL', qScrUV, mBsUV, rmseUV, 'LRPrf_PBL'))
    cntnts.append(buildString(58, 'AE2BR_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'LRPrf_FT'))
    cntnts.append(buildString(59, 'AE2BR_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'LRPrf_PBL'))
    cntnts.append(buildString(60, 'AEXTF_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'βextFine_FT'))
    cntnts.append(buildString(61, 'AEXTF_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'βextFine_PBL'))
    cntnts.append(frmStr %   (62, 'ANSPH_z_VIS_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (63, 'ANSPH_z_VIS_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (64, 'ANC_z_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (65, 'ANC_z_profile_in_PBL', bad1, bad1, bad1))
    return cntnts

if __name__ == "__main__": runStatus = main()