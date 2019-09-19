#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import simulateRetrieval as rs
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML


# MacBook Air
fwdModelYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_1lambda_general_V0_fast.yml'
bckYAMLpath = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_5lambda_Template_fast.yml'
saveStart = '/Users/wrespino/Desktop/testCase_' # end will be appended
dirGRASP = None
krnlPath = None
Nsims = 3
maxCPU = 3

# DISCOVER
#basePath = os.environ['NOBACKUP']
#dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
#krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
#fwdModelYAMLpath = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_FWD_IQU_5lambda_CASE-6a-onlyMARINE_V0.yml')
#bckYAMLpath = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/settings_BCK_IQU_5lambda_Template.yml')
#savePath = os.path.join(basePath, 'synced/Working/testDISCOVER_PolMISR_6aMARINE_Sep7_V1.pkl')
#Nsims = 84
#maxCPU = 28

#n = sys.arg[1] # (0,1,2,...,N-1)
n=0

instruments = ['polar0700', 'lidar05'] #1
conCases = ['marine', 'pollution','Smoke','Marine+Smoke'] #3
SZAs = [60, 0] # 2
Phis = [0, 90] # 2 -> N=12 Nodes

sizeMat = [1,1,1, len(instruments), len(conCases), len(SZAs), len(Phis)]
ind = [n//np.prod(sizeMat[i:i+3])%sizeMat[i+3] for i in range(4)]
paramTple = (instruments[ind[0]], conCases[ind[1]], SZAs[ind[2]], Phis[ind[3]])
savePath = saveStart + '%s_case-%s_sza%d_phi%d_V1.pkl' % paramTple
print('-- Processing ' + os.path.basename(savePath) + ' --')

# RUN SIMULATION
nowPix = returnPixel(paramTple[0], sza=paramTple[2], landPrct=100, relPhi=paramTple[3], nowPix=None)
cstmFwdYAML, landPrct = setupConCaseYAML(conCases[ind[1]], nowPix.nwl, fwdModelYAMLpath)
simA = rs.simulation(nowPix) # defines new instance for this architecture
# runs the simulation for given set of conditions, releaseYAML=True -> index of wavelength involved YAML fields MUST cover every wavelength BUT bckYAML Nλ does not have to match fwd calulcation
simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True)
