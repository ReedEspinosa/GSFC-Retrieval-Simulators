#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will simulate a full hemispherical TOA obsrevation with GRASP's forward model
It will also write single scattering properties to a CSV file
"""
import os
import sys
import numpy as np
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML
from ACCP_functions import writeConcaseVars
import runGRASP as rg

caseStrs = ['plltdmrn'] # seperate pixels for each of these scenes (CSV will only be written for first case)
archName = 'polarHemi'
tauFactor = 1
seaLevel = True # True -> ROD (corresponding to masl = 0 m) & rayleigh depol. saved to nc4 file

singleScatCSV = None
hemiNetCDF = '/Users/wrespino/Downloads/Polar07_reflectanceTOA_cleanAtmosphere_landSurface_V2.nc4'
baseYAML = '/Users/wrespino/Synced/Local_Code_MacBook/GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_POLAR_1lambda.yml'

path2repoGRASP = '/Users/wrespino/Synced/Local_Code_MacBook/grasp_open'
binPathGRASP = os.path.join(path2repoGRASP, 'build/bin/grasp') # Path grasp binary
intrnlFileGRASP = os.path.join(path2repoGRASP,'src/retrieval/internal_files') # Path grasp precomputed single scattering kernels


nowPix = returnPixel(archName)
rslts = []
for caseStr in caseStrs:
    fwdYAMLPath = setupConCaseYAML(caseStr, nowPix, baseYAML, caseLoadFctr=tauFactor)
    gObjFwd = rg.graspRun(fwdYAMLPath)
    gObjFwd.addPix(nowPix) # TODO: Couldn't we stick everything into a graspDB at this point?
    gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)
    rslts.append(np.take(gObjFwd.readOutput(),0)) # we need take because readOutput returns list, even if just one element
if hemiNetCDF:
    gObjFwd.output2netCDF(hemiNetCDF, rsltDict=rslts, seaLevel=True)
if singleScatCSV:
    gObjFwd.singleScat2CSV(singleScatCSV)
# print results to console in order of Canoncial case XLSX
writeConcaseVars(rslts[0])