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
from ACCP_functions import normalizeError, findLayerSeperation, findFineModes

instruments = ['Lidar090+polar07','Lidar050+polar07','Lidar060+polar07']

destDir = '/Users/wrespino/Desktop/'
srcPathA = '/Users/wrespino/Synced/Working/SIM_SITA_DecAssess_nonPrime_mediumRes_MERGED_LEV2/'
srcPathB = '/Users/wrespino/Synced/Working/SIM_SITA_DecAssess_Prime_mediumRes_MERGED_LEV2/'
filePatrn = 'DRS_V01_%s_caseAll_tFct1.00_orbSS_multiAngles_nAll_nAngALL.pkl'

χthresh = 2.2 # χ^2 threshold on points
forceχ2Calc = True
minSaved = 13
trgtλUV = 0.360
trgtλVis = 0.532
trgtλNIR = 1.064
bad1 = np.nan

finePBLind = 2 # this is only for 4 fwd & 4 bck mode cases

def main():
    instrument = instruments[0] # loop this later...      
    
    simOjb = simulation(picklePath=os.path.join(srcPathA, filePatrn % instrument))
    simOjb = splitSim(simOjb, 2)
    dictA = buildContents(simA=simOjb) # non-prime

    simOjb = simulation(picklePath=os.path.join(srcPathB, filePatrn % instrument))
    simOjb = splitSim(simOjb, 2)
    dictB = buildContents(simA=simOjb) # prime
    
    dictRatio = {kA: ratioFun(vB,vA) for (kA,vA),vB in zip(dictA.items(),dictB.values())}
    return dictRatio, dictA, dictB

def ratioFun(vB,vA):
    r = vB/vA
    if r > 1: return r**0.3
    return r

def splitSim(simOjb, Nmodes):
    simMind = np.array([rb['aodMode'].shape[0] for rb in simOjb.rsltBck])==Nmodes
    simOjb.rsltBck = simOjb.rsltBck[simMind]
    simOjb.rsltFwd = simOjb.rsltFwd[simMind]
    return simOjb

def prep4normError(rmse, prfRMSE):
    lidScale = 1e6 # m-1 -> Mm-1
    # lower layer
    if 'aodMode' in rmse and len(rmse['aodMode'])==4:
        rmse['aodMode_finePBL'] = np.r_[rmse['aodMode'][finePBLind]]
    else:
        rmse['aodMode_finePBL'] = np.r_[bad1]
    rmse['βext'] = np.r_[np.sqrt(prfRMSE['βext']**2)]*lidScale
    rmse['βextFine'] = np.r_[np.sqrt(np.mean(prfRMSE['βextFine']**2))]*lidScale
    rmse['ssaPrf'] = np.r_[np.sqrt(np.mean(prfRMSE['ssa']**2))]
    rmse['LRPrf'] = np.r_[np.sqrt(np.mean(prfRMSE['LR']**2))]
    return rmse

def buildString(num, name, qScr, mBs, rmse, key):
    return {name:rmse[key][0]}

def buildContents(cntnts=[], simA=None):
    
    # prep/filter this data and find λInds
    simA.conerganceFilter(χthresh=χthresh, forceχ2Calc=forceχ2Calc, verbose=True, minSaved=minSaved)
    lIndUV = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλUV))
    lIndVIS = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλVis))
    lIndNIR = np.argmin(np.abs(simA.rsltFwd[0]['lambda']-trgtλNIR))
    
    hghtCut = np.array([findLayerSeperation(rf) for rf in simA.rsltFwd])
    fineModeInd, fineModeIndBck = findFineModes(simA)
    
    #VIS
    rmseVis = simA.analyzeSim(lIndVIS, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, hghtCut=hghtCut)[0]
    prfRMSE = simA.analyzeSimProfile(wvlnthInd=lIndVIS, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck)[0]
    rmseVis = prep4normError(rmseVis, prfRMSE)
    #NIR
    rmseNIR = simA.analyzeSim(lIndNIR, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, hghtCut=hghtCut)[0]
    prfRMSE = simA.analyzeSimProfile(wvlnthInd=lIndNIR, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck)[0]
    rmseNIR = prep4normError(rmseNIR, prfRMSE)
    #UV - find error stats for column and PBL and profiles 
    if simA.rsltFwd[0]['lambda'][lIndUV] < 0.4: # we have at least one UV channel
        rmseUV = simA.analyzeSim(lIndUV, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck, hghtCut=hghtCut)[0]
        prfRMSE = simA.analyzeSimProfile(wvlnthInd=lIndUV, fineModesFwd=fineModeInd, fineModesBck=fineModeIndBck)[0]
        rmseUV = prep4normError(rmseUV, prfRMSE)
        print('UV: %5.3f μm, VIS: %5.3f μm, NIR: %5.3f μm' % tuple(simA.rsltFwd[0]['lambda'][np.r_[lIndUV,lIndVIS,lIndNIR]]))
    else: # no UV available
        rmseUV = {key: np.r_[bad1] for key in rmseVis}
        print('NO UV DATA, VIS: %5.3f μm, NIR: %5.3f μm' % tuple(simA.rsltFwd[0]['lambda'][np.r_[lIndVIS,lIndNIR]]))
    
    # dummy values from when this was print8Kresults.py
    mBsUV = None
    mBsVis = None
    mBsNIR = None
    qScrUV = None
    qScrVis = None
    qScrNIR = None
    qScrUV_EN = None
    qScrVis_EN = None
    
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
    cntnts.append(buildString(15, 'AE2BR_l_UV_column', qScrUV, mBsUV, rmseUV, 'LidarRatio'))
    cntnts.append(buildString(16, 'AE2BR_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'LidarRatioMode_PBLFT'))
    cntnts.append(buildString(17, 'AE2BR_l_VIS_column', qScrVis, mBsVis, rmseVis, 'LidarRatio'))
    cntnts.append(buildString(18, 'AE2BR_l_VIS_PBL', qScrUV, mBsUV, rmseUV, 'LidarRatioMode_PBLFT'))
    cntnts.append(buildString(19, 'AODF_l_VIS_column', qScrVis, mBsVis, rmseVis, 'aodMode_fine'))
    cntnts.append(buildString(20, 'AODF_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'aodMode_finePBL'))
    cntnts.append(buildString(23, 'AOD_l_UV_column', qScrUV, mBsUV, rmseUV, 'aod'))
    cntnts.append(buildString(24, 'AOD_l_UV_PBL', qScrUV, mBsUV, rmseUV, 'aodMode_PBLFT'))
    cntnts.append(buildString(25, 'AOD_l_VIS_column', qScrVis, mBsVis, rmseVis, 'aod'))
    cntnts.append(buildString(26, 'AOD_l_VIS_PBL', qScrVis, mBsVis, rmseVis, 'aodMode_PBLFT'))
    cntnts.append(buildString(27, 'AOD_l_NIR_column', qScrNIR, mBsNIR, rmseNIR, 'aod'))
    cntnts.append(buildString(28, 'AOD_l_NIR_PBL', qScrNIR, mBsNIR, rmseNIR, 'aodMode_PBLFT'))
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
    cntnts.append(buildString(43, 'AABS_z_UV_profile_column', qScrUV, mBsUV, rmseUV, 'ssaPrf'))
    cntnts.append(buildString(45, 'AABS_z_VIS_profile_column', qScrVis, mBsVis, rmseVis, 'ssaPrf'))
    cntnts.append(buildString(51, 'AEXT_z_UV_profile_column', qScrUV, mBsUV, rmseUV, 'βext'))
    cntnts.append(buildString(53, 'AEXT_z_VIS_profile_column', qScrVis, mBsVis, rmseVis, 'βext'))
    cntnts.append(buildString(55, 'AEXT_z_NIR_profile_column', qScrNIR, mBsNIR, rmseNIR, 'βext'))
    cntnts.append(buildString(57, 'AE2BR_z_UV_profile_column', qScrUV, mBsUV, rmseUV, 'LRPrf'))
    cntnts.append(buildString(59, 'AE2BR_z_VIS_profile_column', qScrVis, mBsVis, rmseVis, 'LRPrf'))
    cntnts.append(buildString(61, 'AEXTF_z_VIS_profile_column', qScrVis, mBsVis, rmseVis, 'βextFine'))

    cntntsDict = {k: v for d in cntnts for k, v in d.items()}
    return cntntsDict

if __name__ == "__main__": dictRatio, dictA, dictB = main()