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

# find and load file
posFiles = glob(simRsltFile)
assert len(posFiles)==1, 'glob found %d files but we expect exactly 1' % len(posFiles)
simA = simulation(picklePath=posFiles[0])
# prep/filter this data and find λInds
simA.conerganceFilter(χthresh=χthresh, verbose=True)
lIndUV = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλUV))
lIndVIS = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλVis))
lIndNIR = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλNIR))

 # each w/ keys: βext, βextFine, ssa

def prep4normError(rmse, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd):
    lidScale = 1e6 # m-1 -> Mm-1
    rmse['aodMode_finePBL'] = np.r_[rmse['aodMode'][finePBLind]]
    bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
    tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
    rmse['βext_PBL'] = np.sqrt(np.mean(prfRMSE['βext'][lowLayInd]**2))
    rmse['βext_FT'] = np.sqrt(np.mean(prfRMSE['βext'][upLayInd]**2))
    rmse['βextFine_PBL'] = np.sqrt(np.mean(prfRMSE['βextFine'][lowLayInd]**2))
    rmse['βextFine_FT'] = np.sqrt(np.mean(prfRMSE['βextFine'][upLayInd]**2))
    rmse['ssaPrf_PBL'] = np.sqrt(np.mean(prfRMSE['ssa'][lowLayInd]**2))
    rmse['ssaPrf_FT'] = np.sqrt(np.mean(prfRMSE['ssa'][upLayInd]**2))
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

# find error stats for column and PBL
rmseUV,bs,tr = simA.analyzeSim(lIndUV, fineModesFwd=[0,2], fineModesBck=[0,2], hghtCut=2100) # todo, drop hghtCut if polar (or better fix in anaylze sim w/ nans)
prfRMSE, prfBias, prfTrue = simA.analyzeSimProfile(wvlnthInd=lIndUV)
rmseUV,bs,tr = prep4normError(rmseUV, bs, tr, prfRMSE, prfBias, prfTrue, finePBLind, upLayInd, lowLayInd)
rmseUV['aodMode_finePBL'] = np.r_[rmseUV['aodMode'][finePBLind]]
bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
qScrUV, mBsUV, _ = normalizeError(rmseUV,bs,tr)
qScrUV_EN, _0, _ = normalizeError(rmseUV,bs,tr, enhanced=True)
rmseVis,bs,tr = simA.analyzeSim(lIndVIS, fineModesFwd=[0,2], fineModesBck=[0,2], hghtCut=2100)
rmseVis['aodMode_finePBL'] = np.r_[rmseVis['aodMode'][finePBLind]]
bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
qScrVis, mBsVis, _ = normalizeError(rmseVis,bs,tr)
qScrVis_EN, _0, _ = normalizeError(rmseVis,bs,tr, enhanced=True)
rmseNIR,bs,tr = simA.analyzeSim(lIndNIR, fineModesFwd=[0,2], fineModesBck=[0,2], hghtCut=2100)
rmseNIR['aodMode_finePBL'] = np.r_[rmseNIR['aodMode'][finePBLind]]
bs['aodMode_finePBL'] = np.r_[[bs['aodMode'][:,finePBLind]]].T
tr['aodMode_finePBL'] = np.r_[[tr['aodMode'][:,finePBLind]]].T
qScrNIR, mBsNIR, _ = normalizeError(rmseVis,bs,tr)
# find error stats for profile variables
extRMSE = simA.analyzeSimProfile(wvlnthInd=lIndUV)[0]

cntnts = ['FINENAME - Version Info'] # TODO: fill this out
cntnts = ['GV#, GVname, QIscore, mean_unc, RMS']
frmStr = '%2d, %29s, %8.3f, %8.3f, %8.3f'
bad1 = 999
q = qScrUV['ssa'][0]
m = mBsUV['ssa'][0]
r = rmseUV['ssa'][0]
cntnts.append(frmStr % ( 1, 'AAOD_l_UV_column', q, m, r))
q = qScrUV['ssaMode_PBLFT'][0]
m = mBsUV['ssaMode_PBLFT'][0]
r = rmseUV['ssaMode_PBLFT'][0]
cntnts.append(frmStr % ( 2, 'AAOD_l_UV_PBL', q, m, r))
q = qScrVis['ssa'][0]
m = mBsVis['ssa'][0]
r = rmseVis['ssa'][0]
cntnts.append(frmStr % ( 3, 'AAOD_l_VIS_column', q, m, r))
q = qScrVis['ssaMode_PBLFT'][0]
m = mBsVis['ssaMode_PBLFT'][0]
r = rmseVis['ssaMode_PBLFT'][0]
cntnts.append(frmStr % ( 4, 'AAOD_l_VIS_PBL', q, m, r))
q = qScrUV_EN['ssa'][0]
m = mBsUV['ssa'][0]
r = rmseUV['ssa'][0]
cntnts.append(frmStr % ( 5, 'AAOD_l_UV_column', q, m, r)) # enhacnced
q = qScrUV_EN['ssaMode_PBLFT'][0]
m = mBsUV['ssaMode_PBLFT'][0]
r = rmseUV['ssaMode_PBLFT'][0]
cntnts.append(frmStr % ( 6, 'AAOD_l_UV_PBL', q, m, r)) # enhacnced
q = qScrVis_EN['ssa'][0]
m = mBsVis['ssa'][0]
r = rmseVis['ssa'][0]
cntnts.append(frmStr % ( 7, 'AAOD_l_VIS_column', q, m, r)) # enhacnced
q = qScrVis_EN['ssaMode_PBLFT'][0]
m = mBsVis['ssaMode_PBLFT'][0]
r = rmseVis['ssaMode_PBLFT'][0]
cntnts.append(frmStr % ( 8, 'AAOD_l_VIS_PBL', q, m, r)) # enhacnced
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % ( 9, 'ASYM_UV', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (10, 'ASYM_VIS', q, m, r))
q = qScrUV['rEffCalc'][0]
m = mBsUV['rEffCalc'][0]
r = rmseUV['rEffCalc'][0]
cntnts.append(frmStr % (11, 'AEFR_l_column', q, m, r))
q = qScrUV['rEffMode_PBLFT'][0]
m = mBsUV['rEffMode_PBLFT'][0]
r = rmseUV['rEffMode_PBLFT'][0]
cntnts.append(frmStr % (12, 'AEFR_l_PBL', q, m, r))
q = qScrUV['LidarRatio'][0]
m = mBsUV['LidarRatio'][0]
r = rmseUV['LidarRatio'][0]
cntnts.append(frmStr % (13, 'AE2BR_l_UV_column', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (14, 'AE2BR_l_UV_PBL', q, m, r))
q = qScrVis['LidarRatio'][0]
m = mBsVis['LidarRatio'][0]
r = rmseVis['LidarRatio'][0]
cntnts.append(frmStr % (15, 'AE2BR_l_VIS_column', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (16, 'AE2BR_l_VIS_PBL', q, m, r))
q = qScrNIR['LidarRatio'][0]
m = mBsNIR['LidarRatio'][0]
r = rmseNIR['LidarRatio'][0]
cntnts.append(frmStr % (17, 'AE2BR_l_NIR_column', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (18, 'AE2BR_l_NIR_PBL', q, m, r))
q = qScrUV['aodMode_fine'][0]
m = mBsUV['aodMode_fine'][0]
r = rmseUV['aodMode_fine'][0]
cntnts.append(frmStr % (19, 'AODF_l_UV_column', q, m, r))
q = qScrUV['aodMode_finePBL'][0]
m = mBsUV['aodMode_finePBL'][0]
r = rmseUV['aodMode_finePBL'][0]
cntnts.append(frmStr % (20, 'AODF_l_UV_PBL', q, m, r))
q = qScrVis['aodMode_fine'][0]
m = mBsVis['aodMode_fine'][0]
r = rmseVis['aodMode_fine'][0]
cntnts.append(frmStr % (21, 'AODF_l_VIS_column', q, m, r))
q = qScrVis['aodMode_finePBL'][0]
m = mBsVis['aodMode_finePBL'][0]
r = rmseVis['aodMode_finePBL'][0]
cntnts.append(frmStr % (22, 'AODF_l_VIS_PBL', q, m, r))
q = qScrNIR['aodMode_fine'][0]
m = mBsNIR['aodMode_fine'][0]
r = rmseNIR['aodMode_fine'][0]
cntnts.append(frmStr % (23, 'AODF_l_NIR_column', q, m, r))
q = qScrNIR['aodMode_finePBL'][0]
m = mBsNIR['aodMode_finePBL'][0]
r = rmseNIR['aodMode_finePBL'][0]
cntnts.append(frmStr % (24, 'AODF_l_NIR_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (25, 'ANSPH_l_UV_column', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (26, 'ANSPH_l_UV_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (27, 'ANSPH_l_VIS_column', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (28, 'ANSPH_l_VIS_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (29, 'ANSPH_l_NIR_column', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (30, 'ANSPH_l_NIR_PBL', q, m, r))
q = qScrUV['aod'][0]
m = mBsUV['aod'][0]
r = rmseUV['aod'][0]
cntnts.append(frmStr % (31, 'AOD_l_UV_column', q, m, r))
q = qScrUV['aodMode_PBLFT'][0]
m = mBsUV['aodMode_PBLFT'][0]
r = rmseUV['aodMode_PBLFT'][0]
cntnts.append(frmStr % (32, 'AOD_l_UV_PBL', q, m, r))
q = qScrVis['aod'][0]
m = mBsVis['aod'][0]
r = rmseVis['aod'][0]
cntnts.append(frmStr % (33, 'AOD_l_VIS_column', q, m, r))
q = qScrVis['aodMode_PBLFT'][0]
m = mBsVis['aodMode_PBLFT'][0]
r = rmseVis['aodMode_PBLFT'][0]
cntnts.append(frmStr % (34, 'AOD_l_VIS_PBL', q, m, r))
q = qScrNIR['aod'][0]
m = mBsNIR['aod'][0]
r = rmseNIR['aod'][0]
cntnts.append(frmStr % (35, 'AOD_l_NIR_column', q, m, r))
q = qScrNIR['aodMode_PBLFT'][0]
m = mBsNIR['aodMode_PBLFT'][0]
r = rmseNIR['aodMode_PBLFT'][0]
cntnts.append(frmStr % (36, 'AOD_l_NIR_PBL', q, m, r))
q = bad1 # TODO: if we have time we could add this
m = bad1
r = bad1
cntnts.append(frmStr % (37, 'APM25', q, m, r))
q = qScrVis['n'][0]
m = mBsVis['n'][0]
r = rmseVis['n'][0]
cntnts.append(frmStr % (38, 'ARIR_l_column', q, m, r))
q = qScrVis['n_PBLFT'][0]
m = mBsVis['n_PBLFT'][0]
r = rmseVis['n_PBLFT'][0]
cntnts.append(frmStr % (39, 'ARIR_l_PBL', q, m, r))
q = qScrVis['k'][0]
m = mBsVis['k'][0]
r = rmseVis['k'][0]
cntnts.append(frmStr % (40, 'AIIR_l_column', q, m, r))
q = qScrVis['k_PBLFT'][0]
m = mBsVis['k_PBLFT'][0]
r = rmseVis['k_PBLFT'][0]
cntnts.append(frmStr % (41, 'AIIR_l_PBL', q, m, r))
q = bad1 # TODO: if we have time we could add this
m = bad1
r = bad1
cntnts.append(frmStr % (42, 'AABS_z_UV_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (43, 'AABS_z_UV_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (44, 'AABS_z_VIS_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (45, 'AABS_z_VIS_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (46, 'AEFR_z_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (47, 'AEFR_z_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (48, 'AEXT_z_UV_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (49, 'AEXT_z_UV_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (50, 'AEXT_z_VIS_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (51, 'AEXT_z_VIS_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (52, 'AEXT_z_NIR_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (53, 'AEXT_z_NIR_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (54, 'AE2BR_z_UV_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (55, 'AE2BR_z_UV_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (56, 'AE2BR_z_VIS_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (57, 'AE2BR_z_VIS_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (58, 'AE2BR_z_NIR_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (59, 'AE2BR_z_NIR_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (60, 'AEXTF_z_UV_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (61, 'AEXTF_z_UV_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (62, 'AEXTF_z_VIS_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (63, 'AEXTF_z_VIS_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (64, 'AEXTF_z_NIR_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (65, 'AEXTF_z_NIR_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (66, 'ANSPH_z_UV_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (67, 'ANSPH_z_UV_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (68, 'ANSPH_z_VIS_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (69, 'ANSPH_z_VIS_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (70, 'ANSPH_z_NIR_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (71, 'ANSPH_z_NIR_profile_in_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (72, 'ANC_z_profile_above_PBL', q, m, r))
q = bad1
m = bad1
r = bad1
cntnts.append(frmStr % (73, 'ANC_z_profile_in_PBL', q, m, r))


