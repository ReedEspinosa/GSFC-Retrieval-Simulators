#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will simulate a full hemispherical TOA obsrevation with GRASP's forward model
"""
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML
import runGRASP as rg

archName = 'polarHemi'
#caseStrs = 'cleanDesert'
caseStrs = 'cleanVegetation'
baseYAML = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_3lambda_POL.yml'
binPathGRASP = '/usr/local/bin/grasp'
intrnlFileGRASP = None


nowPix = returnPixel(archName)
fwdYAMLPath, landPrct = setupConCaseYAML(caseStrs, nowPix, baseYAML)
nowPix.land_prct = landPrct
gObjFwd = rg.graspRun(fwdYAMLPath)
gObjFwd.addPix(nowPix)
gObjFwd.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=intrnlFileGRASP)

