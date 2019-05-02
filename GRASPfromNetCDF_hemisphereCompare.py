#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import matplotlib.pyplot as plt
from runGRASP import graspDB
from MADCAP_functions import readVILDORTnetCDF
import numpy as np

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
savePlotPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_nosurface_PP_SSCORR_OUTGOING/')
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_nosurface_PP_SSCORR_OUTGOING/calipso-g5nr.vlidort.vector.LAMBERTIAN.%dd00.nc4')
rsltsFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_nosurface_PP_SSCORR_OUTGOING/rayleigh_bench_MS.pkl'

#varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'sensor_zenith', 'sensor_azimuth']
varNames = ['I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'surf_reflectance_Q_scatplane','surf_reflectance_U_scatplane', 'sensor_zenith', 'sensor_azimuth']

pltVar = 'I'
noRayleigh = False # only compare with surface reflectance
relDeltaI = False # relative (True) or absolute (False) I/Q/U difference (no effect on DOLP)

gDB = graspDB()
gDB.loadResults(rsltsFile)

# PLOTTING CODE
wvls = np.atleast_1d(gDB.rslts[0]['lambda'])


lbl1 = '$%s_{VLIDORT}$' % pltVar
lbl2 = '$%s_{GRASP}$' % pltVar
lbl3 = "$2(%s_{grasp} - %s_{vildort})/(%s_{grasp} + %s_{vildort})$ [%%]"  % (pltVar, pltVar, pltVar, pltVar) if relDeltaI else '$1000*(%s_{grasp} - %s_{vildort})$'  % (pltVar, pltVar) 


# Read in radiances, solar spectral irradiance and find reflectances
#measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls)
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, [0.865])
Nwvl = wvls.shape[0]
if Nwvl == 1:
    fig, ax = plt.subplots(1,3, subplot_kw=dict(projection='polar'), figsize=(14,6))
    ax = ax[:,None]
else:
    fig, ax = plt.subplots(3,Nwvl, subplot_kw=dict(projection='polar'), figsize=(20,15))
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
    if (pltVar in ['Q', 'U']) and (measData[l]['Q'+pltVarApnd].sum() > 0): # we probably need a 90 deg cord. sys. rotation
            vildort = -vildort
            pltVarStr = '-'+pltVarStr
    print('Plotting netCDF variable %s' % pltVarStr)
    # Get GRASP values and find delta
    if pltVar in ['I', 'Q', 'U']:
        fitStr = 'fit_%s' % pltVar
        fit = np.array([rslt[fitStr][:,l] for rslt in gDB.rslts])
        delta = 200*(fit-vildort)/(fit+vildort) if relDeltaI else 1000*(fit-vildort)
    elif pltVar=='DOLP':
#        fit = np.array([rslt['fit_PoI'][:,l] for rslt in gDB.rslts])
#        fit = np.array([rslt['fit_P'][:,l]/rslt['fit_I'][:,l] for rslt in gDB.rslts])
        fit = np.array([np.sqrt(rslt['fit_Q'][:,l]**2 + rslt['fit_U'][:,l]**2)/rslt['fit_I'][:,l] for rslt in gDB.rslts])
#        fit = np.array([np.sqrt(rslt['fit_Q'][:,l]**2 + rslt['fit_U'][:,l]**2) for rslt in gDB.rslts])
        delta = 100*(fit-vildort)
        lbl1 = '$DoLP_{VLIDORT}$ [absolute]' # these are differnt from I,Q,U labels
        lbl2 = '$DoLP_{GRASP}$ [absolute]'
        lbl3 = '$DoLP_{grasp} - DoLP_{vildort}$ [%]'
    # Plot the results
    azimth=measData[l]['sensor_azimuth']*np.pi/180
    zenith=measData[l]['sensor_zenith']
    r, theta = np.meshgrid(zenith, azimth)
    if pltVar in ['I','DOLP']:
        clrMin = 0.9*np.minimum(vildort.min(), fit.min())
        clrMax = 1.1*vildort.max()
        if pltVar=='DOLP': clrMax = np.minimum(clrMax, 1)
        clrMap = cmap=plt.cm.jet
    else:
        mag = np.abs(np.r_[vildort.min(),vildort.max(), fit.min(),fit.max()]).max()
        clrMin = -mag
        clrMax = mag
        clrMap = plt.cm.seismic
    v = np.linspace(clrMin, clrMax, 200, endpoint=True)
    ticks = np.linspace(clrMin, clrMax, 3, endpoint=True) 
    for i in range(3):
        if i == 0: 
            data=vildort.T
            if Nwvl > 1:
                ax[i,l].set_title('$\\lambda=%4.2f\\mu m$' % wvls[l], y=1.25)
        elif i==1: 
            data=fit.T
        else: 
#            data=np.log10(delta.T)
            data=delta.T
            mag = np.abs(np.r_[data.min(),data.max()]).max()
#            mag = np.r_[np.abs(data.max())] # HACK
            v = np.linspace(-mag, mag, 256, endpoint=True)
            ticks = np.linspace(-mag, mag, 3, endpoint=True)
            clrMap = plt.cm.seismic
        c = ax[i,l].contourf(theta, r, data, v, cmap=clrMap)
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[i,l], ticks=ticks)
ax[0,0].set_ylabel(lbl1, labelpad=30)
ax[1,0].set_ylabel(lbl2, labelpad=30)
ax[2,0].set_ylabel(lbl3, labelpad=30)
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])

A = 'NO' if noRayleigh else ''
B = os.path.split(os.path.split(radianceFNfrmtStr)[0])[1][10:] # [10:] removes 'benchmark_'
if pltVar == 'DOLP':
    C = ''
elif relDeltaI:
    C = '_relative'
else:
    C = '_absolute'
figFN =  A+B+'-'+pltVar+C+'.png'
fig.savefig(os.path.join(savePlotPath, figFN), bbox_inches='tight')
