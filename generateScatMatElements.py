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
import copy
import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", "GRASP_scripts"))
from runGRASP import graspRun
from MADCAP_functions import loadVARSnetCDF
import miscFunctions as mf

# Paths to files
#outFile = '/var/folders/lt/3kt1ddms7211cvcclg7yq3pc5f5bnm/T/tmp_db1lcdf/bench_inversionRslts.txt'
#outFile = '/var/folders/lt/3kt1ddms7211cvcclg7yq3pc5f5bnm/T/tmp6fq7ekhf/bench_inversionRslts.txt.V0.8.2'
outFile = '/Users/wrespino/Synced/Working/bench_inversionRslts.txt'
rmtPrjctPath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/VLIDORTbench_graspConfig_12/'
radianceFNfrmtStr = os.path.join(rmtPrjctPath, 'benchmark_rayleigh+simple_aerosol_nosurface_Osku_dry_V3/calipso-g5nr.vlidort.vector.LAMBERTIAN.800d00.nc4')
aeroMode = 0
caseName = 'graspConfig_12_Osku_dry'

Nplts = 4

gr = graspRun()
rslt = gr.readOutput(outFile)

fig, ax = plt.subplots(-(-Nplts//2),2, figsize=(8,1+Nplts*1.5))
figD, axD = plt.subplots(-(-Nplts//2),2, figsize=(8,1+Nplts*1.5))

elmntsNetCDF = ['p11', 'p12', 'p33', 'p34', 'p22', 'p44']
#elmnts = ['ph11osca', 'ph12osca', 'ph33osca', 'ph34osca', 'ph22osca', 'ph44osca']
elmnts = elmntsNetCDF
wvInds = [2,5,9,13,15] #2->350nm, 5->500nm, 9->700nm, 13->1000nm, 15->1500nm
varNames = ['angle', 'scatangle', 'phasefunc', 'lambda', 'colTOTdist', 'radius']
optTbl = loadVARSnetCDF(radianceFNfrmtStr, varNames+[s.upper() for s in elmntsNetCDF])
for i in range(Nplts):
    #GRASP
    ang = rslt[0]['angle'][:,aeroMode,:]
    if i==0:
        PijRef = rslt[0][elmnts[0]][:,aeroMode,:]
        ax[i//2,i%2].set_yscale('log')
        ax[i//2,i%2].set_ylabel('$S_{11}$')
        axD[i//2,i%2].set_ylabel('$ΔS_{11}$ [$2\\frac{GRASP-X}{GRASP+X}$ (%)]')
    else:
        PijRef = rslt[0][elmnts[i]][:,aeroMode,:]/rslt[0][elmnts[0]][:,aeroMode,:]
        ax[i//2,i%2].set_ylabel('$S_{'+elmnts[i][1::]+'}/S_{11}}$')
        axD[i//2,i%2].set_ylabel('$ΔS_{'+elmnts[i][1::]+'}/S_{11}}$ [$GRASP-X$ (%)]')
    ax[i//2,i%2].plot(ang, PijRef)
    if i==0: P11Ref = copy.copy(PijRef)
    if i==1: P12Ref = copy.copy(PijRef)
    
    #OPTICS TABLES
    if not 'float' in type(rslt[0]['lambda']).__name__: ax[i//2,i%2].set_prop_cycle(None) # color by λ not source if multi-λ case
    if 'phasefunc' in optTbl: # true optics tables from Pete
        ang = optTbl['scatangle']
        Pij = optTbl['phasefunc'][i,:,0,14,wvInds].squeeze()
        if i>0: Pij = Pij/optTbl['phasefunc'][0,:,0,14,wvInds].squeeze()
    else: # OSSE output from Patricia
        ang = optTbl['angle']
        Pij = optTbl[elmntsNetCDF[i].upper()]
        if i>0: Pij = Pij/optTbl[elmntsNetCDF[0].upper()]
    Pij = np.atleast_2d(Pij).T
    ax[i//2,i%2].plot(ang,Pij, '--')
    ax[i//2,i%2].set_xlim([0,180])   
    ax[i//2,i%2].set_xlabel('scattering angle (deg)')
    PijDiff = 100*(PijRef-Pij) if i>0 else 200*(PijRef-Pij)/(PijRef+Pij)
    axD[i//2,i%2].plot(ang,PijDiff, '--')
    axD[i//2,i%2].set_xlim([0,180])   
    axD[i//2,i%2].set_xlabel('scattering angle (deg)')
    
#PyMieScat [NOTE: this isn't tested with multiple modes or λ's]
r = np.atleast_2d(rslt[0]['r'])[0,:]
dvdlnr = np.atleast_2d(rslt[0]['dVdlnr'])[0,:]

#r = optTbl['radius']
#dvdlnr = optTbl['colTOTdist']
if not 'float' in type(rslt[0]['lambda']).__name__:
    ax[0,0].set_prop_cycle(None)
    if Nplts>1: ax[0,1].set_prop_cycle(None)
for l,n,k in zip(np.atleast_1d(rslt[0]['lambda']), np.atleast_1d(rslt[0]['n']), np.atleast_1d(rslt[0]['k'])):
    ang, p11, p12 = mf.phaseMat(r, dvdlnr, n, k, l)
    p11 = np.atleast_2d(p11).T
    p12 = np.atleast_2d(p12).T
    ax[0,0].plot(ang,p11, ':')
    axD[0,0].plot(ang, 200*(P11Ref-p11)/(P11Ref+p11), ':')
    if Nplts>1:
        ax[0,1].plot(ang, -p12, ':')
        axD[0,1].plot(ang, 100*(P12Ref+p12), ':')
        
if 'float' in type(rslt[0]['lambda']).__name__:
    ax[0,0].legend(['GRASP', 'Optics Table', 'PyMieScat'])
    axD[0,0].legend(['Optics Table', 'PyMieScat'])
    fig.suptitle(caseName + '_V2 Sulfate ' + '(λ = %4.2f μm)' % rslt[0]['lambda'])
else:
    ax[0,0].legend(['λ = %4.2f μm' % l for l in np.atleast_1d(rslt[0]['lambda'])])
    fig.suptitle(caseName + ': GRASP (solid) vs Optics Tables (dashed) vs PyMieScatt (dotted)')
fig.tight_layout(rect=[0.01, 0.01,0.98, 0.98])
figD.tight_layout(rect=[0.01, 0.01,0.98, 0.98])

