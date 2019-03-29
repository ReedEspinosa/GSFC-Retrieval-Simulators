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
savePlotPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/March29_BenchmarkPlots')
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_BRDF_BPDF_PP/calipso-g5nr.vlidort.vector.MODIS_BRDF_BPDF.%dd00.nc4')
rsltsFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_BRDF_BPDF_PP/NOrayleigh_bench.pkl'

#varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'sensor_zenith', 'sensor_azimuth']
varNames = ['I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'sensor_zenith', 'sensor_azimuth']

pltVar = 'I'
noRayleigh = True # only compare with surface reflectance
relDeltaI = False # relative (True) or absolute (False) I/Q/U difference (no effect on DOLP)

gDB = graspDB()
gDB.loadResults(rsltsFile)

# PLOTTING CODE
wvls = np.atleast_1d(gDB.rslts[0]['lambda'])


lbl1 = '$%s_{VLIDORT}$' % pltVar
lbl2 = '$%s_{GRASP}$' % pltVar
lbl3 = "$2(I_{grasp} - I_{vildort})/(I_{grasp} + I_{vildort})$ [%]" if relDeltaI else '$1000*(%s_{grasp} - %s_{vildort})$'  % (pltVar, pltVar) 


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
    if noRayleigh: 
        vildort = measData[l][pltVar+'_surf']
    elif pltVar in ['Q', 'U']:
        vildort = -measData[l][pltVar+'_scatplane'] # negative sign due to 90 deg cord. sys. rotation 
    else:
        vildort = measData[l][pltVar]
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
    azimth=measData[l]['sensor_azimuth']*np.pi/180
    zenith=measData[l]['sensor_zenith']
    r, theta = np.meshgrid(zenith, azimth)
    if pltVar in ['I','DOLP']:
        clrMin = 0.9*vildort.min()
        clrMax = 1.1*vildort.max()
        clrMap = cmap=plt.cm.jet
    else:
#        mag = np.abs(np.r_[vildort.min(),vildort.max()]).max()
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
#            mag = np.r_[0.3]
            v = np.linspace(-mag, mag, 256, endpoint=True)
            ticks = np.linspace(-mag, mag, 3, endpoint=True)
            clrMap = plt.cm.seismic
        c = ax[i,l].contourf(theta, r, data, v, cmap=clrMap)
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[i,l], ticks=ticks)
ax[0,0].set_ylabel(lbl1, labelpad=30)
ax[1,0].set_ylabel(lbl2, labelpad=30)
ax[2,0].set_ylabel(lbl3, labelpad=30)
plt.tight_layout(rect=[0.01, 0.01,0.98, 0.98])

pltVar = 'I'

relDeltaI = True # relative (True) or absolute (False) I/Q/U difference (no effect on DOLP)

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
