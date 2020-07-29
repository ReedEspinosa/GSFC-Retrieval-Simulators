#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import matplotlib.pyplot as plt
from runGRASP import graspDB
from MADCAP_functions import readVILDORTnetCDF, findNewestMatch
import numpy as np
import re
from scipy import interpolate as intrp
import scipy.integrate.quadrature
#import scipy.ndimage as ndimage


# Paths to files benchmark_GRASP_basedLUTs_V1/graspConfig_12_Osku_DrySU_V1
# basePath = '/Users/wrespino/Synced/' # NASA MacBook
#rmtPrjctPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/VLIDORTbench_graspConfig_12')
radianceFNfrmtStr = '/Users/wrespino/Synced/MADCAP_CAPER/VLIDORTbench_graspConfig_12/benchmark_rayleigh+simple_aerosol_CX/calipso-g5nr.vlidort.vector.CX.%dd00.nc4'
rsltsFile = findNewestMatch(os.path.split(radianceFNfrmtStr)[0], pattern='VLIDORTMatch_vC*.pkl')
savePlotPath = os.path.split(radianceFNfrmtStr)[0]

#varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'sensor_zenith', 'sensor_azimuth']
varNames = ['ROT', 'TAU', 'I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'surf_reflectance_Q_scatplane','surf_reflectance_U_scatplane', 'sensor_zenith', 'sensor_azimuth']

pltVar = 'I'
noRayleigh = False # only compare with surface reflectance
relDeltaI = True # relative (True) or absolute (False) I/Q/U difference (no effect on DOLP)
singScat = False # plot single scattering aerosol (no surf or rayleigh) instead of GRASP

print('Loading results from: ', os.path.split(rsltsFile)[1])
gDB = graspDB()
gDB.loadResults(rsltsFile)

# custom tag to append to plot file names
cstmTag = re.search('^[A-z_0-9]+_([0-9]+[nmLambda]+_YAML[0-9a-f]+).pkl', os.path.split(rsltsFile)[1])
assert not cstmTag is None, 'Could not parse the PKL filename'
cstmTag = cstmTag.group(1) + "_P11fixed"
ttlStr = 'λ = 800 nm'

maxZnth = 70; # difference plots scales will accommodate values beyond this zenith angle
hemi = False

# PLOTTING CODE
wvls = np.atleast_1d(gDB.rslts[0]['lambda'])
#wvls = np.r_[0.4]
lbl1 = '$%s_{VLIDORT}$' % pltVar
if singScat:
    assert pltVar=='I', "only intensity is available in singScat mode"
    lbl2 = '$%s_{SScalc}$' % pltVar
    lbl3 = "$2(%s_{SScalc} - %s_{vildort})/(%s_{SScalc} + %s_{vildort})$ [%%]"  % (pltVar, pltVar, pltVar, pltVar) if relDeltaI else '$1000*(%s_{grasp} - %s_{vildort})$'  % (pltVar, pltVar) 
else:
    lbl2 = '$%s_{GRASP}$' % pltVar
    lbl3 = "$2(%s_{grasp} - %s_{vildort})/(%s_{grasp} + %s_{vildort})$ [%%]"  % (pltVar, pltVar, pltVar, pltVar) if relDeltaI else '$1000*(%s_{grasp} - %s_{vildort})$'  % (pltVar, pltVar) 


# Read in radiances, solar spectral irradiance and find reflectances
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls, verbose=True)
Nwvl = wvls.shape[0]

fig, ax = plt.subplots(Nwvl, 3, subplot_kw=dict(projection='polar'), figsize=(14,3+3*Nwvl))
if Nwvl == 1: ax = ax[None,:]
for l in range(Nwvl):
    # Get VLIDORT netCDF values
    if noRayleigh: 
        pltVarApnd = '_surf'
    elif pltVar in ['Q', 'U']:
        pltVarApnd = '_scatplane'
    else:
        pltVarApnd = ''
    pltVarStr = pltVar+pltVarApnd    
    vildort = measData[l][pltVarStr]
#    if (pltVar in ['Q', 'U']) and (measData[l]['Q'+pltVarApnd].sum() < 0): # we might need a 90 deg cord. sys. rotation
#            vildort = -vildort
#            pltVarStr = '-'+pltVarStr
    print('Plotting netCDF variable %s at %dnm' % (pltVarStr, 1000*wvls[l]))
    # Get GRASP values and find delta
    if pltVar in ['I', 'Q', 'U']:
        if singScat:
            mu = np.atleast_2d(np.cos(measData[l]['sensor_zenith']*np.pi/180)).T
            mu0 = np.cos(30*np.pi/180)
            tau = gDB.rslts[0]['aod']
            ssa = gDB.rslts[0]['ssa']
            ang = gDB.rslts[0]['angle'].squeeze()*np.pi/180
            p11 = intrp.interp1d(ang, gDB.rslts[0]['p11'].squeeze())
            p11Val = np.array([p11(rslt['sca_ang'][:,l]*np.pi/180) for rslt in gDB.rslts]).T
            p11wKrn = intrp.interp1d(ang, gDB.rslts[0]['p11'].squeeze()*np.sin(ang)) # renormalize P11 (next 3 lines)
            p11nrm = scipy.integrate.quadrature(p11wKrn,0,np.pi,maxiter=500)[0]
            p11Val = p11Val*2/p11nrm
            fit = np.pi*ssa*p11Val/(4*np.pi)*(mu0/(mu0+mu))*(1 - np.exp(-tau*(1/mu+1/mu0)))
#            vildort = np.pi*ssa*p11Val/(4*np.pi)*(mu0/(mu0+mu))*(1 - np.exp(-tau*(1/mu+1/mu0)))
            fitStr = 'calculated single scattering (from renormalized GRASP P11)'
        else:
            fitStr = 'fit_%s' % pltVar
            fit = np.array([rslt[fitStr][:,l] for rslt in gDB.rslts])    
        if fit.shape[::-1]==vildort.shape and np.diff(fit.shape)!=0: fit=fit.T # fix sloppy array dimensions, if matrix is not square
        
#        vildort = ndimage.gaussian_filter(fit,(0,0.3),order=0) # BIG HACK
#        lbl1 = '$%s_{GRASP_smth}$'  % pltVar
#        lbl3 = '$I_{GRASP_smth} vs %I_{GRASP}$ [%]'
#        
#        fit = ndimage.gaussian_filter(vildort,(0,0.3),order=0) # BIG HACK
#        lbl2 = '$%s_{VLIDORTsmth}$'  % pltVar
#        lbl3 = '$I_{VLIDORT} vs I_{VLIDORTsmth}$ [%]'
        
        delta = 200*(fit-vildort)/(fit+vildort) if relDeltaI else 1000*(fit-vildort)
    elif pltVar=='DOLP':
#        fit = np.array([rslt['fit_PoI'][:,l] for rslt in gDB.rslts])
#        fit = np.array([rslt['fit_P'][:,l]/rslt['fit_I'][:,l] for rslt in gDB.rslts])
        fitStr = '(fit_Q^2+fit_U^2)/fit_I'
        fit = np.array([np.sqrt(rslt['fit_Q'][:,l]**2 + rslt['fit_U'][:,l]**2)/rslt['fit_I'][:,l] for rslt in gDB.rslts])
        if fit.shape[::-1]==vildort.shape and np.diff(fit.shape)!=0: fit=fit.T # fix sloppy array dimensions, if matrix is not square
#        fit = np.array([np.sqrt(rslt['fit_Q'][:,l]**2 + rslt['fit_U'][:,l]**2) for rslt in gDB.rslts])

        lbl1 = '$DoLP_{VLIDORT}$ [absolute]' # these are differnt from I,Q,U labels
        lbl2 = '$DoLP_{GRASP}$ [absolute]'
        lbl3 = '$DoLP_{GRASP} - DoLP_{VLIDORT}$ [%]'

#        vildort = ndimage.gaussian_filter(fit,(0,0.3),order=0) # BIG HACK
#        lbl1 = '$DoLP_{GRASP_smth}$'
#        lbl3 = '$DoLP_{GRASP_smth} - DoLP_{GRASP}$ [%]'
#        
#        fit = ndimage.gaussian_filter(vildort,(0,0.3),order=0) # BIG HACK
#        lbl2 = '$DoLP_{VLIDORTsmth}$'
#        lbl3 = '$DoLP_{VLIDORT} - DoLP_{VLIDORTsmth}$ [%]'

        delta = 100*(fit-vildort)
    
    print('Plotting GRASP variable %s at %dnm' % (fitStr, 1000*wvls[l]))
    azMax = np.pi if hemi else 2*np.pi
    azimth=measData[l]['sensor_azimuth']*np.pi/180
    zenith=measData[l]['sensor_zenith']
    r, theta = np.meshgrid(zenith, azimth[azimth<=azMax])
    if pltVar in ['I','DOLP']:
        clrMin = 1*np.minimum(vildort.min(), fit.min())
        clrMax = 1*np.array([vildort.max(), fit.max(), 1e-10]).max()
        if pltVar=='DOLP': clrMax = np.minimum(clrMax, 1)
        clrMap = cmap=plt.cm.jet
    else:
        mag = np.abs(np.r_[vildort.min(),vildort.max(), fit.min(),fit.max()]).max()
        mag = np.maximum(mag, 1e-10)
        clrMin = -mag
        clrMax = mag
        clrMap = plt.cm.seismic
    v = np.linspace(clrMin, clrMax, 200, endpoint=True)
    ticks = np.linspace(clrMin, clrMax, 3, endpoint=True)
    for i in range(3):
        if i == 0: 
            data=vildort.T[azimth<=azMax,:]
        elif i==1: 
            data=fit.T[azimth<=azMax,:]
        else: 
#            data=np.log10(delta.T)
            data=delta.T[azimth<=azMax,:]
            mag = np.abs(np.r_[data[:,zenith<=maxZnth].min(),data[:,zenith<=maxZnth].max(), 1e-10]).max()
#            mag = np.r_[np.abs(data.max())] # HACK
            v = np.linspace(-mag, mag, 256, endpoint=True)
            ticks = np.linspace(-mag, mag, 3, endpoint=True)
            clrMap = plt.cm.seismic
        c = ax[l,i].contourf(theta, r, data, v, cmap=clrMap)        
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
        if Nwvl > 1: ax[l,i].set_yticks(range(0, 90, 20))
        if hemi: ax[l,i].set_thetalim([0,np.pi])
    wvStr = ' ($%4.2f\\mu m$)' % wvls[l]
    ax[l,0].set_ylabel(lbl1 + wvStr, labelpad=30)
    ax[l,1].set_ylabel(lbl2 + wvStr, labelpad=30)
    ax[l,2].set_ylabel(lbl3, labelpad=30)
plt.suptitle(ttlStr)
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])

A = 'NO' if noRayleigh else ''
B = os.path.split(os.path.split(radianceFNfrmtStr)[0])[1][10:] # [10:] removes 'benchmark_'
if pltVar == 'DOLP':
    C = ''
elif relDeltaI:
    C = '_relative'
else:
    C = '_absolute'
figFN =  A+B+'_'+cstmTag+'-'+pltVar+C+'.png'
fig.savefig(os.path.join(savePlotPath, figFN), bbox_inches='tight')
