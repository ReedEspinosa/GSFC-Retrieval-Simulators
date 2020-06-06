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
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../GRASP_scripts'))
from simulateRetrieval import simulation
from ACCP_functions import normalizeError


# instruments = ['Lidar09','Lidar05','Lidar06', 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=189
instruments = ['Lidar090','Lidar050','Lidar060', 'Lidar090+polar07','Lidar050+polar07','Lidar060+polar07'] # 7 N=189
# instruments = ['polar07'] # 7 N=189
# instruments = ['Lidar090+polar07GPM'] # 7 N=189
# caseIDs = ['6a', '6b', '6c', '6d', '6e', '6f', '6g', '6h', '6i']
caseIDs = ['6All']
surfaces = ['','Vegetation','Desert']
simType = 'DRS'
# simType = 'SPA'
orbit = 'SS'

srcPath = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFilesLev2/'
# srcPath = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/'
# filePatrn = 'DRS_V11_%s_case0%s%s_orb%s_tFct1.00_multiAngles_n*_nAngALL.pkl'
# filePatrn = 'DRS_V08_%s_case0%s%s_orbSS%s_tFct1.00_multiAngles_n*_nAngALL.pkl'
filePatrn = 'DRS_V08_%s_case0%s%s_orb%s.pkl'
# SPA_V11_Lidar09+polar07GPM_SPADesert_orbGPM.pkl
PathFrmt = os.path.join(srcPath, filePatrn)
destDir = '/Users/wrespino/Desktop/8Kfiles_EspinosaJune05_GSFC_V3'
comments = 'tighter fit filter - no scores for nonexistent layers or UV GVs when no UV channel - file name SSP3->SSG3 - extra AABS_z_NIR_profile... variables removed'

# χthresh = 10
χthresh = 10
trgtλUV = 0.355
trgtλVis = 0.532
trgtλNIR = 1.064
finePBLind = 2
fineModeInd = [0,2]
# fineModeInd = [0]

assert 'lidar' not in instruments[0].lower() or len(fineModeInd)==2, 'Wrong fineModeInd???'
assert 'lidar' in instruments[0].lower() or len(fineModeInd)==1, 'Wrong fineModeInd???'
upLayInd = [1,2,3,4]
lowLayInd = [5,6,7,8,9]
frmStr = '%2d, %29s, %8.3f, %8.3f, %8.3f'
bad1 = 999 # bad data value

polarSSPn = 0
def main():
    runStatus = []
    for instrument in instruments:
        for caseID in caseIDs:
            for surface in surfaces:
                runStatus.append(run1case(instrument, orbit, caseID, surface, PathFrmt, polOnlyPlat=polarSSPn))
    return runStatus
                
def run1case(instrument, orbit, caseID, surface, PathFrmt, polOnlyPlat=0):
    NNN='NGE'
    polarOnly = False
    TYP=simType + caseID.lower()
    if 'Lidar05' in instrument: 
        PLTF='SSP0'
    elif 'Lidar06' in instrument: 
        PLTF='SSP1'
    elif 'Lidar09' in instrument: 
        PLTF='SSP2' if orbit=='SS' else 'SSG3'
    if 'polar07' in instrument:
        if 'Lidar' in instrument:
            OBS='NAD'
        else:
            PLTF='SSG3'  if polOnlyPlat==3 else 'SSP%d' % polOnlyPlat
            OBS='OND'
            polarOnly = True
    else:
        OBS='NAN'
    SRFC='OCEN' if len(surface)==0 else surface.replace('Desert','LNDD').replace('Vegetation','LNDV')
    RES='RES1'
    V='V01'

    trgtFN = '_'.join([NNN,TYP,PLTF,OBS,SRFC,RES,V]) + '.csv'
    # find and load file
    simRsltFile = PathFrmt % (instrument, caseID, surface, orbit)
    # simRsltFile = PathFrmt % (instrument, caseID, surface, 'SS')
    print(simRsltFile)
    posFiles = glob(simRsltFile)
    if not len(posFiles)==1:
        print('¡¡¡Glob found %d files but we expect exactly 1!!!' % len(posFiles))
        print('We return the search pattern and move on to the next file...')
        print('<><><><><>')
        return 'FAILED: '+simRsltFile
    print('Loading %s...' % os.path.basename(posFiles[0]))
    simA = simulation(picklePath=posFiles[0])
    trgtPath = os.path.join(destDir, trgtFN)
    cntnts = ['%s, Comments: %s' % (trgtFN, comments)]
    print('Building conents of output file...')
    cntnts = buildContents(cntnts, simA, polarOnly)
    with open(trgtPath, mode='wt', encoding='utf-8') as file:
        file.write('\n'.join(cntnts))  
    print('Output file save to %s' % os.path.basename(trgtPath))
    return trgtFN


def prep4normError(rmse, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd, polarOnly):
    lidScale = 1e6 # m-1 -> Mm-1
    if polarOnly:
        nanVars = ['aodMode_finePBL', 'βext_PBL', 'βext_FT', 'βextFine_PBL', 'βextFine_FT', 'ssaPrf_PBL', 'ssaPrf_FT']
        for vr in nanVars:
            rmse[vr] = np.r_[bad1]
            bs[vr] = np.r_[bad1]
            tr[vr] = np.r_[bad1]
        return rmse, bs, tr
    # lower layer
    rmse['aodMode_finePBL'] = np.r_[rmse['aodMode'][finePBLind]]
    bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
    tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
    rmse['βext_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['βext'][lowLayInd]**2))]*lidScale
    rmse['βextFine_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine'][lowLayInd]**2))]*lidScale
    rmse['ssaPrf_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa'][lowLayInd]**2))]
    bs['βext_PBL'] = prfBias['βext'][:,lowLayInd].reshape(-1,1)*lidScale
    bs['βextFine_PBL'] = prfBias['βextFine'][:,lowLayInd].reshape(-1,1)*lidScale
    bs['ssaPrf_PBL'] = prfBias['ssa'][:,lowLayInd].reshape(-1,1)
    tr['βext_PBL'] = prfTrue['βext'][:,lowLayInd].reshape(-1,1)*lidScale
    tr['βextFine_PBL'] = prfTrue['βextFine'][:,lowLayInd].reshape(-1,1)*lidScale
    tr['ssaPrf_PBL'] = prfTrue['ssa'][:,lowLayInd].reshape(-1,1)
    # upper layer
    if prfTrue['βext'][:,upLayInd].max() < 1/lidScale: # no meaningful upperlay (never exceeds 1 Mm-1)
        nanVars = ['βext_FT', 'βextFine_FT',  'ssaPrf_FT'] 
        for vr in nanVars:
            rmse[vr] = np.r_[bad1]
            bs[vr] = np.r_[bad1]
            tr[vr] = np.r_[bad1]
        return rmse, bs, tr
    rmse['βext_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['βext'][upLayInd]**2))]*lidScale
    rmse['βextFine_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine'][upLayInd]**2))]*lidScale
    rmse['ssaPrf_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa'][upLayInd]**2))]
    bs['βext_FT'] = prfBias['βext'][:,upLayInd].reshape(-1,1)*lidScale
    bs['βextFine_FT'] = prfBias['βextFine'][:,upLayInd].reshape(-1,1)*lidScale
    bs['ssaPrf_FT'] = prfBias['ssa'][:,upLayInd].reshape(-1,1)
    tr['βext_FT'] = prfTrue['βext'][:,upLayInd].reshape(-1,1)*lidScale
    tr['βextFine_FT'] = prfTrue['βextFine'][:,upLayInd].reshape(-1,1)*lidScale
    tr['ssaPrf_FT'] = prfTrue['ssa'][:,upLayInd].reshape(-1,1)
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

def buildContents(cntnts, simA, polarOnly):
    
    # prep/filter this data and find λInds
    cntnts.append('GV#, GVname, QIscore, mean_unc, RMS')
    simA.conerganceFilter(χthresh=χthresh, verbose=True, minSaved=7)
    lIndUV = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλUV))
    lIndVIS = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλVis))
    lIndNIR = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλNIR))
    
    hghtCut = None if polarOnly else 2100
    #VIS
    rmseVis,bs,tr = simA.analyzeSim(lIndVIS, fineModesFwd=fineModeInd, fineModesBck=fineModeInd, hghtCut=hghtCut)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndVIS)
    rmseVis,bs,tr = prep4normError(rmseVis, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd, polarOnly)
    qScrVis, mBsVis, _ = normalizeError(rmseVis,bs,tr)
    qScrVis_EN, _0, _ = normalizeError(rmseVis,bs,tr, enhanced=True)
    #NIR
    rmseNIR,bs,tr = simA.analyzeSim(lIndNIR, fineModesFwd=fineModeInd, fineModesBck=fineModeInd, hghtCut=hghtCut)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndNIR)
    rmseNIR,bs,tr = prep4normError(rmseNIR, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd, polarOnly)
    qScrNIR, mBsNIR, _ = normalizeError(rmseNIR,bs,tr)
    #UV - find error stats for column and PBL and profiles 
    if simA.rsltFwd[0]['lambda'][lIndUV] < 0.4: # we have at least one UV channel
        rmseUV,bs,tr = simA.analyzeSim(lIndUV, fineModesFwd=fineModeInd, fineModesBck=fineModeInd, hghtCut=hghtCut) # todo, drop hghtCut if polar (or better fix in anaylze sim w/ nans)
        prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndUV)
        rmseUV,bs,tr = prep4normError(rmseUV, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd, polarOnly)
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
    cntnts.append(frmStr %   (9,  'ASYM_UV', bad1, bad1, bad1)) # should be able to replace these with calls to BS variable in buildString
    cntnts.append(frmStr %   (10, 'ASYM_VIS', bad1, bad1, bad1))
    cntnts.append(buildString(11, 'AEFR_l_column', qScrUV, mBsUV, rmseUV, 'rEffCalc'))
    cntnts.append(buildString(12, 'AEFR_l_PBL', qScrUV, mBsUV, rmseUV, 'rEffMode_PBLFT'))
    cntnts.append(buildString(13, 'AE2BR_l_UV_column', qScrUV, mBsUV, rmseUV, 'LidarRatio'))
    cntnts.append(frmStr %   (14, 'AE2BR_l_UV_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(15, 'AE2BR_l_VIS_column', qScrVis, mBsVis, rmseVis, 'LidarRatio'))
    cntnts.append(frmStr %   (16, 'AE2BR_l_VIS_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(17, 'AE2BR_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'LidarRatio'))
    cntnts.append(frmStr %   (18, 'AE2BR_l_NIR_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(19, 'AODF_l_UV_column', qScrUV, mBsUV, rmseUV, 'aodMode_fine'))
    cntnts.append(buildString(20, 'AODF_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'aodMode_finePBL'))
    cntnts.append(buildString(21, 'AODF_l_VIS_column', qScrVis, mBsVis, rmseVis, 'aodMode_fine'))
    cntnts.append(buildString(22, 'AODF_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'aodMode_finePBL'))
    cntnts.append(buildString(23, 'AODF_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'aodMode_fine'))
    cntnts.append(buildString(24, 'AODF_l_NIR_PBL', qScrNIR, mBsNIR, rmseNIR, 'aodMode_finePBL'))
    cntnts.append(frmStr %   (25, 'ANSPH_l_UV_column', bad1, bad1, bad1))
    cntnts.append(frmStr %   (26, 'ANSPH_l_UV_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (27, 'ANSPH_l_VIS_column', bad1, bad1, bad1))
    cntnts.append(frmStr %   (28, 'ANSPH_l_VIS_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (29, 'ANSPH_l_NIR_column', bad1, bad1, bad1))
    cntnts.append(frmStr %   (30, 'ANSPH_l_NIR_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(31, 'AOD_l_UV_column', qScrUV, mBsUV, rmseUV, 'aod'))
    cntnts.append(buildString(32, 'AOD_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'aodMode_PBLFT'))
    cntnts.append(buildString(33, 'AOD_l_VIS_column', qScrVis, mBsVis, rmseVis, 'aod'))
    cntnts.append(buildString(34, 'AOD_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'aodMode_PBLFT'))
    cntnts.append(buildString(35, 'AOD_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'aod'))
    cntnts.append(buildString(36, 'AOD_l_NIR_PBL', qScrNIR, mBsNIR, rmseNIR, 'aodMode_PBLFT'))
    cntnts.append(frmStr %   (37, 'APM25', bad1, bad1, bad1)) # TODO: if we have time we could add this
    cntnts.append(buildString(38, 'ARIR_l_column', qScrVis, mBsVis, rmseVis, 'n'))
    cntnts.append(buildString(39, 'ARIR_l_PBL', qScrVis, mBsVis, rmseVis, 'n_PBLFT'))
    cntnts.append(buildString(40, 'AIIR_l_column', qScrVis, mBsVis, rmseVis, 'k'))
    cntnts.append(buildString(41, 'AIIR_l_PBL', qScrVis, mBsVis, rmseVis, 'k_PBLFT'))
    cntnts.append(buildString(42, 'AABS_z_UV_profile_above_PBL', qScrUV, mBsUV, rmseUV, 'ssaPrf_FT'))
    cntnts.append(buildString(43, 'AABS_z_UV_profile_in_PBL', qScrUV, mBsUV, rmseUV, 'ssaPrf_PBL'))
    cntnts.append(buildString(44, 'AABS_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'ssaPrf_FT'))
    cntnts.append(buildString(45, 'AABS_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'ssaPrf_PBL'))
    cntnts.append(frmStr %   (46, 'AEFR_z_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (47, 'AEFR_z_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(48, 'AEXT_z_UV_profile_above_PBL', qScrUV, mBsUV, rmseUV, 'βext_FT'))
    cntnts.append(buildString(49, 'AEXT_z_UV_profile_in_PBL', qScrUV, mBsUV, rmseUV, 'βext_PBL'))
    cntnts.append(buildString(50, 'AEXT_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'βext_FT'))
    cntnts.append(buildString(51, 'AEXT_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'βext_PBL'))
    cntnts.append(buildString(52, 'AEXT_z_NIR_profile_above_PBL', qScrNIR, mBsNIR, rmseNIR, 'βext_FT'))
    cntnts.append(buildString(53, 'AEXT_z_NIR_profile_in_PBL', qScrNIR, mBsNIR, rmseNIR, 'βext_PBL'))
    cntnts.append(frmStr %   (54, 'AE2BR_z_UV_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (55, 'AE2BR_z_UV_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (56, 'AE2BR_z_VIS_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (57, 'AE2BR_z_VIS_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (58, 'AE2BR_z_NIR_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (59, 'AE2BR_z_NIR_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(buildString(60, 'AEXTF_z_UV_profile_above_PBL', qScrUV, mBsUV, rmseUV, 'βextFine_FT'))
    cntnts.append(buildString(61, 'AEXTF_z_UV_profile_in_PBL', qScrUV, mBsUV, rmseUV, 'βextFine_PBL'))
    cntnts.append(buildString(62, 'AEXTF_z_VIS_profile_above_PBL', qScrVis, mBsVis, rmseVis, 'βextFine_FT'))
    cntnts.append(buildString(63, 'AEXTF_z_VIS_profile_in_PBL', qScrVis, mBsVis, rmseVis, 'βextFine_PBL'))
    cntnts.append(buildString(64, 'AEXTF_z_NIR_profile_above_PBL', qScrNIR, mBsNIR, rmseNIR, 'βextFine_FT'))
    cntnts.append(buildString(65, 'AEXTF_z_NIR_profile_in_PBL', qScrNIR, mBsNIR, rmseNIR, 'βextFine_PBL'))
    cntnts.append(frmStr %   (66, 'ANSPH_z_UV_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (67, 'ANSPH_z_UV_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (68, 'ANSPH_z_VIS_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (69, 'ANSPH_z_VIS_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (70, 'ANSPH_z_NIR_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (71, 'ANSPH_z_NIR_profile_in_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (72, 'ANC_z_profile_above_PBL', bad1, bad1, bad1))
    cntnts.append(frmStr %   (73, 'ANC_z_profile_in_PBL', bad1, bad1, bad1))
    return cntnts

if __name__ == "__main__": runStatus = main()