#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will export our simulation results to the Google Docs spreadsheet format used to report A-CCP SIT-A results """

import numpy as np
import os
import sys
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
from simulateRetrieval import simulation

simRsltFile = '/Users/wrespino/synced/Working/SIM14_lidarPolACCP/SIM202_4mode_lidar05+polar07_case-case06c_sza30_phi0_tFct1.00_V2.pkl'
lInd = 4 # LIDAR λ to plot
tf = 0
tc = 1
bf = 2
bc = 3
simA = simulation(picklePath=simRsltFile)
if not type(simA.rsltFwd) is dict: simA.rsltFwd = simA.rsltFwd[0] # HACK [VERY BAD] -- remove when we fix this to work with lists 
datRaw = np.nan*np.ones([11,28])
dat = np.nan*np.ones([11,2])

aodCacl = lambda F,B,m,l: [np.sum(rb['aodMode'][m,l]) for rb in B] - np.sum(F['aodMode'][m,l]) 
datRaw[5,:] = aodCacl(simA.rsltFwd,simA.rsltBck,[tf,bf],lInd) # Fine total AOD
datRaw[6,:] = aodCacl(simA.rsltFwd,simA.rsltBck,bf,lInd) # PBL Fine AOD
datRaw[7,:] = aodCacl(simA.rsltFwd,simA.rsltBck,np.r_[0:4],lInd) # column AOD
datRaw[8,:] = aodCacl(simA.rsltFwd,simA.rsltBck,[bf,bc],lInd) # PBL total AOD

[rb['n'][:,lInd] for rb in simA.rsltBck] - simA.rsltFwd['n'][:,lInd]
[rb['n'][[bf,bc],2] for rb in simA.rsltBck] - simA.rsltFwd['n'][[bf,bc],lInd]