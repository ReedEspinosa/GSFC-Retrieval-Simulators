#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will simulate a full hemispherical TOA obsrevation with GRASP's forward model
It will also write single scattering properties to a CSV file
"""
import os
import sys
import numpy as np
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML
from ACCP_functions import writeConcaseVars
import runGRASP as rg

caseStrs = ['plltdmrn'] # seperate pixels for each of these scenes (CSV will only be written for first case)
# caseStrs = ['DustNonsph'] # seperate pixels for each of these scenes (CSV will only be written for first case)
tauFactor = 1
hemiNetCDF = None
singleScatCSV = None
# caseStrs = ['cleanDesert', 'cleanVegetation'] # seperate pixels for each of these scenes
# hemiNetCDF = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/Polar07_reflectanceTOA_cleanAtmosphere_landSurface_V2.nc4'
# singleScatCSV = None
baseYAML = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_POL.yml'
archName = 'polarHemi'
binPathGRASP = '/usr/local/bin/grasp'
intrnlFileGRASP = None
seaLevel = True # True -> ROD (corresponding to masl = 0 m) & rayleigh depol. saved to nc4 file

nowPix = returnPixel(archName)
rslts = []
for caseStr in caseStrs:
    fwdYAMLPath = setupConCaseYAML(caseStr, nowPix, baseYAML, caseLoadFctr=tauFactor)
    gObjFwd = rg.graspRun(fwdYAMLPath)
    gObjFwd.addPix(nowPix)
    gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
    rslts.append(np.take(gObjFwd.readOutput(),0)) # we need take because readOutput returns list, even if just one element
if hemiNetCDF:
    gObjFwd.output2netCDF(hemiNetCDF, rsltDict=rslts, seaLevel=True)
if singleScatCSV:
    gObjFwd.singleScat2CSV(singleScatCSV)
# print results to console in order of Canoncial case XLSX
writeConcaseVars(rslts[0])