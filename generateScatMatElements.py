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
rhi = 0
outFile = '/Users/wrespino/Synced/Working/GRASP_PMgenerationRun/bench_inversionRslts_optics_SU.v5_7.GSFun.nc_RHind%d_oldBnd.txt' % rhi
radianceFNfrmtStr = '/Users/wrespino/Synced/Working/GRASP_PMgenerationRun/optics_SU.v5_7.GSFun.nc'
aeroMode = 0
caseName = os.path.basename(outFile)

Nplts = 4

gr = graspRun()
rslt = gr.readOutput(outFile)

fig, ax = plt.subplots(-(-Nplts//2),2, figsize=(8,1+Nplts*1.5))
figD, axD = plt.subplots(-(-Nplts//2),2, figsize=(8,1+Nplts*1.5))

elmntsNetCDF = ['p11', 'p12', 'p33', 'p34', 'p22', 'p44']
#elmnts = ['ph11osca', 'ph12osca', 'ph33osca', 'ph34osca', 'ph22osca', 'ph44osca']
elmnts = elmntsNetCDF
varNames = ['angle', 'scatangle', 'scattering_angle', 'phasefunc', 'phase_matrix', 'lambda', 'colTOTdist', 'radius']
optTbl = loadVARSnetCDF(radianceFNfrmtStr, varNames+[s.upper() for s in elmntsNetCDF])
wvInds = [np.isclose(x,optTbl['lambda']*1e6).nonzero()[0][0] for x in rslt[0]['lambda']]
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
    pmName = None
    for nm in ['phasefunc', 'phase_matrix']: # Pete's original or new Osku
        if nm in optTbl: pmName = nm
    for nm in ['scatangle', 'scattering_angle']: # Pete's original or new Osku
        if nm in optTbl: angName = nm
    if pmName:# true optics tables 
        ang = optTbl[angName]
        Pij = optTbl[pmName][i,:,0,rhi,wvInds].squeeze()
        if i>0: Pij = Pij/optTbl[pmName][0,:,0,rhi,wvInds].squeeze()
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
    axD[0,0].set_prop_cycle(None)
    if Nplts>1: axD[0,1].set_prop_cycle(None)
lCut = 0.5 # do not show pyplot diff below this λ in μm (too much noise)
for i,(l,n,k) in enumerate(zip(np.atleast_1d(rslt[0]['lambda']), np.atleast_1d(rslt[0]['n']), np.atleast_1d(rslt[0]['k']))):
    ang, p11, p12 = mf.phaseMat(r, dvdlnr, n, k, l)
#    p11 = np.atleast_2d(p11).T
#    p12 = np.atleast_2d(p12).T
    ax[0,0].plot(ang,p11, ':')
    if l>lCut:
        axD[0,0].plot(ang, 200*(P11Ref[:,i]-p11)/(P11Ref[:,i]+p11), ':')
    else:
        axD[0,0].plot([],[])
    if Nplts>1:
        ax[0,1].plot(ang, -p12, ':')
        if l>lCut:
            axD[0,1].plot(ang, 100*(P12Ref[:,i]+p12), ':')
        else:
            axD[0,1].plot([],[])
        
if 'float' in type(rslt[0]['lambda']).__name__:
    ax[0,0].legend(['GRASP', 'Optics Table', 'PyMieScat'])
    axD[0,0].legend(['Optics Table', 'PyMieScat'])
    fig.suptitle(caseName + '(λ = %4.2f μm)' % rslt[0]['lambda'])
else:
    ax[0,0].legend(['λ = %4.2f μm' % l for l in np.atleast_1d(rslt[0]['lambda'])])
    fig.suptitle(caseName + ': GRASP (-), OptTbls (--), PyMieScatt (:)')
fig.tight_layout(rect=[0.01, 0.01,0.98, 0.98])
figD.tight_layout(rect=[0.01, 0.01,0.98, 0.98])

