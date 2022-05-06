#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Merge many simulation result pkl files """

import numpy as np
import sys
import os
from simulateRetrieval import simulation
from glob import glob
import re
inDirPath = '/Users/aputhukkudy/Working_Data/ACCDAM/2022/Campex_Simulations/Mar2022/'\
            'Flight#1/Non-Spherical/Linear/16bins/'
outDirPath = '/Users/aputhukkudy/Working_Data/ACCDAM/2022/Campex_Simulations/Mar2022/'\
            'Flight#1/Non-Spherical/Linear/16bins_combined/'
# fnPtrn = 'DRS_V11_*_orb*_tFct0.*_sza*_phi*_n%d_nAng*.pkl' # %d -> n
# DRS_V10_Lidar05_case08k2_tFct1.00_orbSS_sza22_phi104_n*_nAng0.pkl
fnPtrn ='Camp2ex_16bins_AOD_*_550nm.pkl'
# fnPtrn = 'TEST_V06_*_case*_tFct*_orb*_sza*_phi*_n%d_nAng*.pkl' # %d -> n
nVals = np.r_[0:10]
for n in nVals: # loop over n
    files = glob(os.path.join(inDirPath, fnPtrn % n))
    if len(files)>0:
        simBase = simulation()
        simBase.rsltFwd = np.empty(len(files), dtype=dict)
        simBase.rsltBck = np.empty(len(files), dtype=dict)
        saveFN = re.sub('_nAng[0-9]+\.pkl','_nAngALL.pkl',os.path.basename(files[0]))
        saveFN = re.sub('_sza[0-9]+_phi[0-9]+_n', '_multiAngles_n', saveFN)
        print('%85s - NAng=%d - costVals:' % (saveFN, len(files)), end='')
        for i,file in enumerate(files): # loop over all available nAng
            simA = simulation(picklePath=file)
            bestInd = np.argmin([rb['costVal'] for rb in simA.rsltBck])
            simBase.rsltFwd[i] = simA.rsltFwd[0]
            simBase.rsltBck[i] = simA.rsltBck[bestInd]
            if i<10: print(' %5.2f,' % simBase.rsltBck[i]['costVal'], end='')
        simBase.saveSim(os.path.join(outDirPath,saveFN))
        print('...')
    else:
        print('No files found for n=%d' % n)
