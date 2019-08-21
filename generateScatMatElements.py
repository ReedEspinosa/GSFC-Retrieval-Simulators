#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:40:28 2019

@author: wrespino
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", "GRASP_scripts"))
from runGRASP import graspRun, pixel
from MADCAP_functions import loadVARSnetCDF

# Paths to files
outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/bench_inversionRslts.txt'
radianceFNfrmtStr = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/optics_SU.v1_5_gpmom_sizedist.nc'

Nplts = 2

gr = graspRun()
rslt = gr.readOutput(outFile)

plt.figure(figsize=(8,4))
plt.subplot(1,Nplts,1)
plt.plot(rslt[0]['angle'],rslt[0]['p11'])
plt.yscale('log')
plt.ylabel('$S_{11}$')

plt.subplot(1,Nplts,2)
plt.plot(rslt[0]['angle'],-rslt[0]['p12']/rslt[0]['p11'])
plt.ylabel('$-S_{12}/S_{11}$')

# plot data from netCDF file
wvInds = [2,5,9,13,15] #2->350nm, 5->500nm, 9->700nm, 13->1000nm, 15->1500nm
varNames = ['scatangle', 'phasefunc', 'lambda']
optTbl = loadVARSnetCDF(radianceFNfrmtStr, varNames)
for i in range(Nplts):
    plt.subplot(1,Nplts,i+1)
    plt.gca().set_prop_cycle(None)
    Pij = optTbl['phasefunc'][i,:,0,14,wvInds].squeeze()
    if i==1: Pij = -Pij/optTbl['phasefunc'][0,:,0,14,wvInds].squeeze()
    plt.plot(optTbl['scatangle'],Pij.T, '--')
    plt.xlim([0,180])   
    plt.xlabel('scattering angle (deg)')
plt.subplot(1,Nplts,1)
plt.legend(['λ = %4.2f μm' % l for l in optTbl['lambda'][wvInds]*1e6])
plt.suptitle('GRASP (solid) vs Optics Tables (dashed) - binSet01')
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])


