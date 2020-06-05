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
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../GRASP_scripts'))
from simulateRetrieval import simulation
from ACCP_functions import normalizeError


instruments = ['Lidar09','Lidar05','Lidar06', 'Lidar09+polar07','Lidar05+polar07','Lidar06+polar07'] # 7 N=189
case = 'case06a'
surface = 'Ocean'
# surface = 'Vegetation'
# surface = 'Desert'


# simRsltFile = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/DRS_V02_%s_%s%s_orbSS_tFct1.00_multiAngles_n*_nAngALL.pkl'
simRsltFile = '/Users/wrespino/Synced/Working/SIM16_SITA_JuneAssessment_SummaryFiles/DRS_V08_Lidar060+polar07_case06iDesert_orbSS_tFct1.00_multiAngles_n106_nAngALL.pkl'

χthresh = 15
trgtλUV = 0.355
trgtλVis = 0.532
trgtλNIR = 1.064
finePBLind = 2
upLayInd = [1,2,3,4]
lowLayInd = [5,6,7,8,9]
frmStr = '%2d, %29s, %8.3f, %8.3f, %8.3f'
bad1 = 999 # bad data value


def main():
    # find and load file
    posFiles = glob(simRsltFile)
    assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
    simA = simulation(picklePath=posFiles[0])
    trgtFile = '/Users/wrespino/Desktop/8Kfiles_X0/test.csv'
    cntnts = ['%s, Comments: None' % trgtFile]
    cntnts = buildContents(cntnts, simA)
    with open(trgtFile, mode='wt', encoding='utf-8') as file:
        file.write('\n'.join(cntnts))  


def prep4normError(rmse, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd):
    lidScale = 1e6 # m-1 -> Mm-1
    rmse['aodMode_finePBL'] = np.r_[rmse['aodMode'][finePBLind]]
    bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
    tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
    rmse['βext_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['βext'][lowLayInd]**2))]*lidScale
    rmse['βext_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['βext'][upLayInd]**2))]*lidScale
    rmse['βextFine_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine'][lowLayInd]**2))]*lidScale
    rmse['βextFine_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine'][upLayInd]**2))]*lidScale
    rmse['ssaPrf_PBL'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa'][lowLayInd]**2))]
    rmse['ssaPrf_FT'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa'][upLayInd]**2))]
    bs['βext_PBL'] = prfBias['βext'][:,lowLayInd].reshape(-1,1)*lidScale
    bs['βext_FT'] = prfBias['βext'][:,upLayInd].reshape(-1,1)*lidScale
    bs['βextFine_PBL'] = prfBias['βextFine'][:,lowLayInd].reshape(-1,1)*lidScale
    bs['βextFine_FT'] = prfBias['βextFine'][:,upLayInd].reshape(-1,1)*lidScale
    bs['ssaPrf_PBL'] = prfBias['ssa'][:,lowLayInd].reshape(-1,1)
    bs['ssaPrf_FT'] = prfBias['ssa'][:,upLayInd].reshape(-1,1)
    tr['βext_PBL'] = prfTrue['βext'][:,lowLayInd].reshape(-1,1)*lidScale
    tr['βext_FT'] = prfTrue['βext'][:,upLayInd].reshape(-1,1)*lidScale
    tr['βextFine_PBL'] = prfTrue['βextFine'][:,lowLayInd].reshape(-1,1)*lidScale
    tr['βextFine_FT'] = prfTrue['βextFine'][:,upLayInd].reshape(-1,1)*lidScale
    tr['ssaPrf_PBL'] = prfTrue['ssa'][:,lowLayInd].reshape(-1,1)
    tr['ssaPrf_FT'] = prfTrue['ssa'][:,upLayInd].reshape(-1,1)
    return rmse, bs, tr

def buildString(num, name, qScr, mBs, rmse, key):
    q = qScr[key][0]
    m = mBs[key][0]
    r = rmse[key][0]
    vals = (num, name, q, m, r)
    return frmStr % vals

def buildContents(cntnts, simA):
    
    # prep/filter this data and find λInds
    cntnts.append('GV#, GVname, QIscore, mean_unc, RMS')
    simA.conerganceFilter(χthresh=χthresh, verbose=True)
    lIndUV = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλUV))
    lIndVIS = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλVis))
    lIndNIR = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλNIR))
    
    #UV - find error stats for column and PBL and profiles
    rmseUV,bs,tr = simA.analyzeSim(lIndUV, fineModesFwd=[0,2], fineModesBck=[0,2], hghtCut=2100) # todo, drop hghtCut if polar (or better fix in anaylze sim w/ nans)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndUV)
    rmseUV,bs,tr = prep4normError(rmseUV, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd)
    qScrUV, mBsUV, _ = normalizeError(rmseUV,bs,tr)
    qScrUV_EN, _0, _ = normalizeError(rmseUV,bs,tr, enhanced=True)
    #VIS
    rmseVis,bs,tr = simA.analyzeSim(lIndVIS, fineModesFwd=[0,2], fineModesBck=[0,2], hghtCut=2100)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndVIS)
    rmseVis,bs,tr = prep4normError(rmseVis, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd)
    qScrVis, mBsVis, _ = normalizeError(rmseVis,bs,tr)
    qScrVis_EN, _0, _ = normalizeError(rmseVis,bs,tr, enhanced=True)
    #NIR
    rmseNIR,bs,tr = simA.analyzeSim(lIndNIR, fineModesFwd=[0,2], fineModesBck=[0,2], hghtCut=2100)
    prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndNIR)
    rmseNIR,bs,tr = prep4normError(rmseNIR, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd)
    qScrNIR, mBsNIR, _ = normalizeError(rmseNIR,bs,tr)
    
    # build file contents, line-by-line
    cntnts.append(buildString(1,  'AAOD_l_UV_column', qScrUV, mBsUV, rmseUV, 'ssa'))
    cntnts.append(buildString(2,  'AAOD_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'ssaMode_PBLFT'))
    cntnts.append(buildString(3,  'AAOD_l_VIS_column', qScrVis, mBsVis, rmseVis, 'ssa'))
    cntnts.append(buildString(4,  'AAOD_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'ssaMode_PBLFT'))
    cntnts.append(buildString(5,  'AAOD_l_UV_column', qScrUV_EN, mBsUV, rmseUV, 'ssa'))
    cntnts.append(buildString(6,  'AAOD_l_UV_PBL', qScrUV_EN, mBsUV, rmseUV, 'ssaMode_PBLFT'))
    cntnts.append(buildString(7,  'AAOD_l_VIS_column', qScrVis_EN, mBsVis, rmseVis, 'ssa'))
    cntnts.append(buildString(8,  'AAOD_l_VIS_PBL', qScrVis_EN, mBsVis, rmseVis, 'ssaMode_PBLFT'))
    cntnts.append(frmStr %   (9,  'ASYM_UV', bad1, bad1, bad1))
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
    cntnts.append(buildString(44, 'AABS_z_NIR_profile_above_PBL', qScrNIR, mBsNIR, rmseNIR, 'ssaPrf_FT'))
    cntnts.append(buildString(45, 'AABS_z_NIR_profile_in_PBL', qScrNIR, mBsNIR, rmseNIR, 'ssaPrf_PBL'))
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

if __name__ == "__main__": main()