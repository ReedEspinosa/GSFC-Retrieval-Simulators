#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from runGRASP import graspDB
from MADCAP_functions import readVILDORTnetCDF, findNewestMatch
import numpy as np
import re
from scipy import interpolate as intrp
import scipy.integrate.quadrature
#import scipy.ndimage as ndimage


# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
rmtPrjctPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/VLIDORTbench_graspConfig/')
radianceFNfrmtStr = os.path.join(rmtPrjctPath, 'benchmark_rayleigh_BRDF/calipso-g5nr.vlidort.vector.MODIS_BRDF.%dd00.nc4')
savePlotPath = os.path.split(radianceFNfrmtStr)[0]

rsltsFiles = [findNewestMatch(os.path.split(radianceFNfrmtStr)[0], pattern='GRASPa9f26b2_stock*.pkl')] # first file should have largest error
rsltsFiles.append(findNewestMatch(os.path.split(radianceFNfrmtStr)[0], pattern='GRASP4fc8ba9_newBuild*.pkl'))

#rsltTitle = ['VLIDORT Ouput', '$\mathregular{VLIDORT-GRASP\ (Standard)}$', '$\mathregular{VLIDORT-GRASP\ (Revised)}$']
rsltTitle = ['VLIDORT Ouput', '$\mathregular{VLIDORT-GRASP\ (\mathbf{Original})}$', '$\mathregular{VLIDORT-GRASP\ (\mathbf{Improved})}$']


#varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'sensor_zenith', 'sensor_azimuth']
varNames = ['TAU', 'I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'surf_reflectance_Q_scatplane','surf_reflectance_U_scatplane', 'sensor_zenith', 'sensor_azimuth']

pltVars = ['I', 'DOLP']
relDeltaI = [False, True] # relative (True) or absolute (False) I/Q/U difference (no effect on DOLP)
noRayleigh = False # only compare with surface reflectance
ttlStr = ''
maxZnth = 80 # difference plots scales will accommodate values beyond this zenith angle
hemi = False
wvl = 0.470 # wavelength to plot in um

font = {'weight' : 'regular',
        'size'   : 12}

matplotlib.rc('font', **font)

# PLOTTING CODE
# Read in radiances, solar spectral irradiance and find reflectances
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, [wvl])

Ntyps = len(pltVars)
cstmTag = '-'
magE = np.ones(Ntyps)
fig, ax = plt.subplots(Ntyps, 3, subplot_kw=dict(projection='polar'), figsize=(13/1.3/1.25,(0.5+3*Ntyps)/1.3))
#fig.tight_layout()
#fig.subplots_adjust(top=0.96, bottom=0.01, right=0.95, wspace=0.4, hspace=0.00)
fig.subplots_adjust(top=0.88, bottom=0.06, right=0.95, wspace=0.4, hspace=0.47)
for k, rsltsFile in enumerate(rsltsFiles):
    # custom tag to append to plot file names
    ptrnMtch = re.search('^([A-z_0-9])+_[0-9]+[nmLambda]+(_YAML[0-9a-f]+).pkl', os.path.split(rsltsFile)[1])
    assert not ptrnMtch is None, 'Could not parse the PKL filename'
    cstmTag = ptrnMtch.group(1)+ptrnMtch.group(2)+'-'

    gDB = graspDB()
    print('Loading results from: ', os.path.split(rsltsFile)[1])
    gDB.loadResults(rsltsFile)
    wvlInd = (gDB.rslts[0]['lambda'] == wvl).nonzero()[0][0]    
    
    if Ntyps == 1: ax = ax[None,:]
    for l, pltVar in enumerate(pltVars):
        # Get VLIDORT netCDF values
        if noRayleigh: 
            pltVarApnd = '_surf'
        elif pltVar in ['Q', 'U']:
            pltVarApnd = '_scatplane'
        else:
            pltVarApnd = ''
        pltVarStr = pltVar+pltVarApnd    
        vildort = 100*measData[0][pltVarStr]
    #    if (pltVar in ['Q', 'U']) and (measData[l]['Q'+pltVarApnd].sum() < 0): # we might need a 90 deg cord. sys. rotation
    #            vildort = -vildort
    #            pltVarStr = '-'+pltVarStr
        print('Plotting netCDF variable %s at %dnm' % (pltVarStr, 1000*wvl))
        # Get GRASP values and find delta
        if pltVar in ['I', 'Q', 'U']:
            fitStr = 'fit_%s' % pltVar
            fit = 100*np.array([rslt[fitStr][:,wvlInd] for rslt in gDB.rslts])      
            if fit.shape[::-1]==vildort.shape and np.diff(fit.shape)!=0: fit=fit.T # fix sloppy array dimensions, if matrix is not square
                    
            delta = 200*(vildort-fit)/(fit+vildort) if relDeltaI[l] else 1*(vildort-fit)
            thrsh = 0.05*vildort.max()
            print('Finding delta from I/Q/U fit...')
        elif pltVar=='DOLP':
            fitStr = '(fit_Q^2+fit_U^2)/fit_I)'            
            fit = 100*np.array([np.sqrt(rslt['fit_Q'][:,wvlInd]**2 + rslt['fit_U'][:,wvlInd]**2)/rslt['fit_I'][:,wvlInd] for rslt in gDB.rslts])
            if fit.shape[::-1]==vildort.shape and np.diff(fit.shape)!=0: fit=fit.T # fix sloppy array dimensions, if matrix is not square
            
            print('Finding delta from DOLP fit...')
            delta = (vildort-fit)
            thrsh = 1.0
        delta[delta<-thrsh] = -thrsh
        delta[delta>thrsh] = thrsh
        print('Plotting GRASP variable %s at %dnm' % (fitStr, 1000*wvl))
        azMax = np.pi if hemi else 2*np.pi
        azimth=measData[0]['sensor_azimuth']*np.pi/180
        zenith=measData[0]['sensor_zenith']
        for i in range(2):
            if i == 0 and k == 0: # absolute plots 
                data=vildort.T[azimth<=azMax,:]
                if pltVar in ['I','DOLP']: # positive valued plots
                    clrMin = 1*np.minimum(vildort.min(), fit.min())
                    clrMax = 1*np.array([vildort.max(), fit.max(), 1e-10]).max()
                    if pltVar=='DOLP': clrMax = np.minimum(clrMax, 100)
                    clrMap = cmap=plt.cm.jet
                else: # positive/negative valued plots
                    mag = np.abs(np.r_[vildort.min(),vildort.max(), fit.min(),fit.max()]).max()
                    mag = np.maximum(mag, 1e-10)
                    clrMin = -mag
                    clrMax = mag
                    clrMap = plt.cm.seismic
                r, theta = np.meshgrid(zenith, azimth[azimth<=azMax])
                v = np.linspace(clrMin, clrMax, 256, endpoint=True)
                ticks = np.linspace(clrMin, clrMax, 4, endpoint=True)
            else: # difference plots 
                data=delta.T[azimth<=azMax,:]
                clrMap = plt.cm.seismic
                if k == 0: magE[l] = np.abs(np.r_[data[:,zenith<=maxZnth].min(),data[:,zenith<=maxZnth].max(), 1e-10]).max() # first diff plot of this type
                v = np.linspace(-magE[l], magE[l], 256, endpoint=True)
                ticks = np.linspace(-magE[l], magE[l], 3, endpoint=True)
            if (i == 0 and k == 0) or i>0: # first col on first run (absolute plots) OR not first col (diff plots)
                print('Ploting on row=%d col=%d' % (l, i+k))
                c = ax[l,i+k].contourf(theta, r, data, v, cmap=clrMap)
                bw = 0.03
                bh = 0.20
                if i==0:
                    frmtStr = "% 2.0f"
                    h = 0.542 if l==0 else 0.055
                    cbaxes = fig.add_axes([0.33, h, 6.5/13*1.35*bw, 11.2/6.5*bh]) # left
                    orient = 'vertical'
                    box = ax[l,i+k].get_position()
                    box.x0 = box.x0 - 0.044
                    box.x1 = box.x1 - 0.044
                    ax[l,i+k].set_position(box)
                elif i==1:
                    frmtStr = "%3.1f" # if l==1 else "%5.1f"
                    h = 0.523 if l==0 else 0.051
                    cbaxes = fig.add_axes([0.61, h, bh*0.8, bw]) # right
                    orient = 'horizontal'
                if i<2: cb = plt.colorbar(c, orientation=orient, ticks=ticks, cax = cbaxes, format=frmtStr)
                if i==1: # add > and < symbols
                    txtArry = cb.ax.get_xticklabels()
                    txtArry[0].set_text("<"+cb.ax.get_xticklabels()[0].get_text())
                    txtArry[-1].set_text(">"+cb.ax.get_xticklabels()[-1].get_text())
                    cb.ax.set_xticklabels(txtArry)                
            if Ntyps > 1: ax[l,i+k].set_yticks(range(0, 90, 20))
            if hemi: ax[l,i+k].set_thetalim([0,np.pi])
labelWght = 'normal'
FS = 11
ax[0,0].set_ylabel('Reflectance ($\mathregular{100 \cdot sr^{-1}}$)', fontweight=labelWght, labelpad=32, fontsize=FS)
ax[1,0].set_ylabel('DoLP (%)', fontweight=labelWght, labelpad=32, fontsize=FS)
[an.set_title(tstr, pad=22, fontweight=labelWght, fontsize=FS) for an, tstr in zip(ax[0,:], rsltTitle)]

LblTxt = np.array(['%d$\\degree$' % x for x in np.r_[0:360:90]]) 
tckLblTxt = np.array([[LblTxt],[np.repeat("",4)]]).T.flatten()
[[ax[i,k].set_xticklabels(tckLblTxt) for i in range(len(pltVars))] for k in range(len(rsltsFiles)+1)]


A = 'NO' if noRayleigh else ''
B = os.path.split(os.path.split(radianceFNfrmtStr)[0])[1][10:] # [10:] removes 'benchmark_'
figFN =  A+B+'_'+cstmTag+pltVar+'.png'
fig.savefig(os.path.join(savePlotPath, figFN), bbox_inches='tight')
