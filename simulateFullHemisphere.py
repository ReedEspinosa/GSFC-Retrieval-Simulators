#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will simulate a full hemispherical TOA obsrevation with GRASP's forward model
"""
import os
import sys
import numpy as np
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML
import runGRASP as rg

archName = 'polarHemi'
#caseStrs = 'cleanDesert'
caseStrs = ['cleanDesert', 'cleanVegetation'] # seperate pixels for each of these scenes
baseYAML = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_POL.yml'
binPathGRASP = '/usr/local/bin/grasp'
intrnlFileGRASP = None
outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/Polar07_reflectanceTOA_cleanAtmosphere_landSurface_V2.nc4'
seaLevel = True # True -> ROD (corresponding to masl = 0 m) & rayleigh depol. saved to nc4 file

nowPix = returnPixel(archName)
rslts = []
for caseStr in caseStrs:
    fwdYAMLPath, landPrct = setupConCaseYAML(caseStr, nowPix, baseYAML)
    nowPix.land_prct = landPrct
    gObjFwd = rg.graspRun(fwdYAMLPath)
    gObjFwd.addPix(nowPix)
    gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
    rslts.append(np.take(gObjFwd.readOutput(),0)) # we need take because readOutput returns list, even if just one element
gObjFwd.output2netCDF(outFile, rsltDict=rslts, seaLevel=True)