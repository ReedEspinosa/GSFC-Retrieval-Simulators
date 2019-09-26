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
from runGRASP import graspRun
from MADCAP_functions import loadVARSnetCDF
import miscFunctions as mf

# Paths to files
#outFile = '/var/folders/lt/3kt1ddms7211cvcclg7yq3pc5f5bnm/T/tmp_db1lcdf/bench_inversionRslts.txt'
#outFile = '/var/folders/lt/3kt1ddms7211cvcclg7yq3pc5f5bnm/T/tmp6fq7ekhf/bench_inversionRslts.txt.V0.8.2'
outFile = '/var/folders/lt/3kt1ddms7211cvcclg7yq3pc5f5bnm/T/tmp6fq7ekhf/bench_inversionRslts_oskuCorrect.txt'
rmtPrjctPath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/VLIDORTbench_graspConfig_12/'
radianceFNfrmtStr = os.path.join(rmtPrjctPath, 'benchmark_rayleigh+simple_aerosol_nosurface_Osku_dry/calipso-g5nr.vlidort.vector.LAMBERTIAN.865d00.nc4')
aeroMode = 0
caseName = 'graspConfig_12_Osku_dry'

Nplts = 4

gr = graspRun()
rslt = gr.readOutput(outFile)

plt.figure(figsize=(8,1+Nplts*1.5))
elmntsNetCDF = ['p11', 'p12', 'p33', 'p34', 'p22', 'p44']
#elmnts = ['ph11osca', 'ph12osca', 'ph33osca', 'ph34osca', 'ph22osca', 'ph44osca']
elmnts = elmntsNetCDF
wvInds = [2,5,9,13,15] #2->350nm, 5->500nm, 9->700nm, 13->1000nm, 15->1500nm
varNames = ['angle', 'scatangle', 'phasefunc', 'lambda', 'colTOTdist', 'radius']
optTbl = loadVARSnetCDF(radianceFNfrmtStr, varNames+[s.upper() for s in elmntsNetCDF])
for i in range(Nplts):
    #GRASP
    plt.subplot(np.ceil(Nplts/2),2,i+1)
    ang = rslt[0]['angle'][:,aeroMode,:]
    if i==0:
        plt.plot(ang,rslt[0][elmnts[0]][:,aeroMode,:])
        plt.yscale('log')
        plt.ylabel('$S_{11}$')
    else:
        plt.plot(ang,rslt[0][elmnts[i]][:,aeroMode,:]/rslt[0][elmnts[0]][:,aeroMode,:])
        plt.ylabel('$S_{'+elmnts[i][1::]+'}/S_{11}}$')
    
    #OPTICS TABLES
#    plt.gca().set_prop_cycle(None)
    if 'phasefunc' in optTbl: # true optics tables from Pete
        ang = optTbl['scatangle']
        Pij = optTbl['phasefunc'][i,:,0,14,wvInds].squeeze()
        if i>0: Pij = Pij/optTbl['phasefunc'][0,:,0,14,wvInds].squeeze()
    else: # OSSE output from Patricia
        ang = optTbl['angle']
        Pij = optTbl[elmntsNetCDF[i].upper()]
        if i>0: Pij = Pij/optTbl[elmntsNetCDF[0].upper()]
    plt.plot(ang,Pij.T, '--')
    plt.xlim([0,180])   
    plt.xlabel('scattering angle (deg)')
    
#PyMieScat [NOTE: this isn't tested with multiple modes or λ's]
r = np.atleast_2d(rslt[0]['r'])[0,:]
dvdlnr = np.atleast_2d(rslt[0]['dVdlnr'])[0,:] # THIS SHOULD NOT BE NEED, EITHER GRASP OR PYMIESCATT IS WRONG!
#r = optTbl['radius']
#dvdlnr = optTbl['colTOTdist']
#plt.subplot(np.ceil(Nplts/2),2,1).set_prop_cycle(None)
if Nplts>1: plt.subplot(np.ceil(Nplts/2),2,2).set_prop_cycle(None)
for l,n,k in zip(np.atleast_1d(rslt[0]['lambda']), np.atleast_1d(rslt[0]['n']), np.atleast_1d(rslt[0]['k'])):
    ang, p11, p12 = mf.phaseMat(r, dvdlnr, n, k, l)
    plt.subplot(np.ceil(Nplts/2),2,1)
    plt.plot(ang,p11, 'r:')
    if Nplts>1:
        plt.subplot(np.ceil(Nplts/2),2,2)
        plt.plot(ang,-p12, 'r:')
        
plt.subplot(np.ceil(Nplts/2),2,1)
#plt.legend(['λ = %4.2f μm' % l for l in np.atleast_1d(rslt[0]['lambda'])])
#plt.suptitle(caseName + ': GRASP (solid) vs Optics Tables (dashed) vs PyMieScatt (dotted)')
#plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])


