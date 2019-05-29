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
#from scipy import interpolate as intrp
#import scipy.ndimage as ndimage


# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
rmtPrjctPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_simple_aerosol+grasp_nosurface')
radianceFNfrmtStr = os.path.join(rmtPrjctPath, 'calipso-g5nr.vlidort.vector.LAMBERTIAN.%dd00.nc4')
rsltsFile = findNewestMatch(os.path.split(radianceFNfrmtStr)[0], pattern='SSbench_sixteenQuadExpnd*.pkl')
savePlotPath = os.path.split(radianceFNfrmtStr)[0]

#varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'sensor_zenith', 'sensor_azimuth']
varNames = ['I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'surf_reflectance_Q_scatplane','surf_reflectance_U_scatplane', 'sensor_zenith', 'sensor_azimuth']

pltVar = 'I'
noRayleigh = False # only compare with surface reflectance
relDeltaI = False # relative (True) or absolute (False) I/Q/U difference (no effect on DOLP)

print('Loading results from: ', os.path.split(rsltsFile)[1])
gDB = graspDB()
gDB.loadResults(rsltsFile)

# custom tag to append to plot file names
cstmTag = re.search('^[A-z_]+_([0-9]+[nmLambda]+_YAML[0-9a-f]+).pkl', os.path.split(rsltsFile)[1])
assert not cstmTag is None, 'Numbers are not allowed in the initial tag of the PKL filename'
cstmTag = cstmTag.group(1)

maxZnth = 65; # difference plots scales will accommodate values beyond this zenith angle

# PLOTTING CODE
wvls = np.atleast_1d(gDB.rslts[0]['lambda'])


lbl1 = '$%s_{VLIDORT}$' % pltVar
lbl2 = '$%s_{GRASP}$' % pltVar
lbl3 = "$2(%s_{grasp} - %s_{vildort})/(%s_{grasp} + %s_{vildort})$ [%%]"  % (pltVar, pltVar, pltVar, pltVar) if relDeltaI else '$1000*(%s_{grasp} - %s_{vildort})$'  % (pltVar, pltVar) 


# Read in radiances, solar spectral irradiance and find reflectances
#measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls)
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls)
Nwvl = wvls.shape[0]

fig, ax = plt.subplots(Nwvl, 3, subplot_kw=dict(projection='polar'), figsize=(14,3+3*Nwvl))
if Nwvl == 1: ax = ax[None,:]
#else:
#    fig, ax = plt.subplots(3,Nwvl, subplot_kw=dict(projection='polar'), figsize=(20,15))
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
    if (pltVar in ['Q', 'U']) and (measData[l]['Q'+pltVarApnd].sum() < 0): # we might need a 90 deg cord. sys. rotation
            vildort = -vildort
            pltVarStr = '-'+pltVarStr
    print('Plotting netCDF variable %s at %dnm' % (pltVarStr, 1000*wvls[l]))
    # Get GRASP values and find delta
    if pltVar in ['I', 'Q', 'U']:
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
        # code to pull P11 and scale VLIDORT by it        
#        p11 = intrp.interp1d(gDB.rslts[0]['angle'].squeeze(), gDB.rslts[0]['ph11osca'].squeeze())
#        p11Val = np.array([p11(rslt['sca_ang'][:,l]) for rslt in gDB.rslts]).T
#        vildort = vildort/p11Val

        delta = 100*(fit-vildort)
    
    print('Plotting GRASP variable %s at %dnm' % (fitStr, 1000*wvls[l]))
    # Plot the results
    azimth=measData[l]['sensor_azimuth']*np.pi/180
    zenith=measData[l]['sensor_zenith']
    r, theta = np.meshgrid(zenith, azimth)
    if pltVar in ['I','DOLP']:
        clrMin = 0.9*np.minimum(vildort.min(), fit.min())
        clrMax = 1.1*np.array([vildort.max(), fit.max(), 1e-10]).max()
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
            data=vildort.T
        elif i==1: 
            data=fit.T
        else: 
#            data=np.log10(delta.T)
            data=delta.T
            mag = np.abs(np.r_[data[:,zenith<=maxZnth].min(),data[:,zenith<=maxZnth].max(), 1e-10]).max()
#            mag = np.r_[np.abs(data.max())] # HACK
#            mag = 0.5
            v = np.linspace(-mag, mag, 256, endpoint=True)
            ticks = np.linspace(-mag, mag, 3, endpoint=True)
            clrMap = plt.cm.seismic
        c = ax[l,i].contourf(theta, r, data, v, cmap=clrMap)        
        cb = plt.colorbar(c, orientation='horizontal', ax=ax[l,i], ticks=ticks)
        if Nwvl > 1: ax[l,i].set_yticks(range(0, 90, 20))
    wvStr = ' ($%4.2f\\mu m$)' % wvls[l]
    ax[l,0].set_ylabel(lbl1 + wvStr, labelpad=30)
    ax[l,1].set_ylabel(lbl2 + wvStr, labelpad=30)
    ax[l,2].set_ylabel(lbl3, labelpad=30)
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
