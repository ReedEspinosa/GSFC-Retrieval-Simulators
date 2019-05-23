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
from runGRASP import graspDB, graspRun, pixel
from MADCAP_functions import hashFileSHA1, loadVARSnetCDF

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
rmtPrjctPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/LIDAR_tests')
dirGRASPworking = False # use sytem temp directories as path to store GRASP SDATA and output files 
pathYAML = os.path.join(basePath, 'Local_Code_MacBook/MADCAP_Analysis/YAML_settingsFiles/settings_HARP_16bin_LIDAR.yml') # path to GRASP YAML file
FNfrmtStr = os.path.join(rmtPrjctPath, 'calipso-g5nr.lc2.ext.20060801_00z_%dd00nm.nc4')
binPathGRASP = os.path.join(basePath, 'Local_Code_MacBook/grasp_open/build/bin/grasp')
savePathTag = 'lidarTest' # preprend tag for save file, A-z and _ only


# Constants
wvls = [0.532, 1.064] # wavelengths to read from levC files
lndPrct = 100; # land cover amount (%)
grspChnkSz = 25 # number of pixles in a single SDATA file
orbHghtKM = 0 # sensor height (km)
GRASP_MIN = 1e-10 # SDATA measurements smaller than GRASP_MIN will be replaced by GRASP_MIN
maxCPUs = 4; # maximum number of simultaneous grasp run threads
#indRng = [351] # tau = 1.96
indRng = np.r_[320:340] # 346 way to low AOD, SHOULD EXPLORE, convergence issue
Nprofs = 45 # number of profiles used, starting at ground, 45 goes to just under 30km which seems to be all GRASP will take
grpSize = 5

# build pkl save path
waveTag = '_%dnm' % (wvls[0]*1000) if len(wvls)==1 else '_%dLambda' % len(wvls)
yamlTag = '_YAML%s' % hashFileSHA1(pathYAML)[0:8]
savePath = os.path.join(os.path.split(FNfrmtStr)[0], savePathTag+waveTag+yamlTag+'.pkl')

tauCDF = []
gObj = []
lon = 0
lat = 0
masl = 0
sza = 0 # LIDAR points directly at nadir
for ind in indRng:
    if np.mod(ind-indRng[0],grpSize)==0:
        gObj.append(graspRun(pathYAML, orbHghtKM, dirGRASPworking))
    dtNm = 730123.0 +ind # dummy value
    nowPix = pixel(dtNm, 1, 1, lon, lat, masl, lndPrct)
    for l,wl in enumerate(wvls): # LOOP OVER WAVELENGTHS
         FNstr = FNfrmtStr % (wl*1000)
         measData = loadVARSnetCDF(FNstr, ['ext','tau','backscat','reff'])
         levWdth = measData['tau'][ind,:]/measData['ext'][ind,:]
         levHeigh = np.cumsum(levWdth[::-1])[::-1]
         tauCDF.append(np.sum(measData['tau'][ind,-Nprofs:]))
         thtv = levHeigh[-Nprofs:]*1000 # km -> m
         nbvm = thtv.shape[0] 
         msTyp = np.r_[36, 39] #Vertical Extinction profile
         msrmnts = np.r_[measData['ext'][ind,-Nprofs:]/1000, measData['backscat'][ind,-Nprofs:]/1000] # 1/km -> 1/m
#         msTyp = np.r_[36] #Vertical Extinction profile
#         msrmnts = measData['ext'][ind,-Nprofs:]/1000 # 1/km -> 1/m
         msrmnts[np.abs(msrmnts) < GRASP_MIN] = GRASP_MIN # HINT: could change Q or U sign but still small absolute shift
         nip = msTyp.shape[0]
         thtv = np.tile(thtv, nip) # ex. 11, 35, 55, 11, 35, 55...
         phi = np.tile(0, nbvm*nip) # this shouldn't matter for LIDAR
         nowPix.addMeas(wl, msTyp, np.repeat(nbvm, nip), sza, thtv, phi, msrmnts)                 
    gObj[-1].addPix(nowPix)


gDB = graspDB(gObj)
gDB.processData(maxCPUs, binPathGRASP, savePath)

if len(indRng)==1:
    ReffCDF=np.sum(measData['reff'][ind,-Nprofs:]*measData['tau'][ind,-Nprofs:])/np.sum(measData['tau'][ind,-Nprofs:])*1e6
    frmtStr = 'Reff = %6.4f; AOD 532nm = %6.4f; AOD 1064nm = %6.4f'
    print('GRASP : '+frmtStr % (gDB.rslts[0]['rEff'], gDB.rslts[0]['aod'][0], gDB.rslts[0]['aod'][1]))
    print('netCDF: '+frmtStr  % (ReffCDF, tauCDF[0], tauCDF[1]))
    sys.exit()

vals = np.array([np.sum(np.diff(measData['ext'][ind,-Nprofs:])**2) for ind in indRng])
maxAOD = 1;
plt.figure()
#plt.plot(tauCDF[::len(wvls)],[rslt['aod'][0] for rslt in gDB.rslts],'.')
#plt.plot(tauCDF[1::len(wvls)],[rslt['aod'][1] for rslt in gDB.rslts],'.')
plt.plot(vals, (np.array(tauCDF[::len(wvls)])-np.array([rslt['aod'][0] for rslt in gDB.rslts]))/tauCDF[::len(wvls)],'.')
plt.plot(vals, (np.array(tauCDF[1::len(wvls)])-np.array([rslt['aod'][1] for rslt in gDB.rslts]))/tauCDF[::len(wvls)],'.')
plt.legend(('532nm', '1064nm'))
plt.xlabel('τ True')
plt.ylabel('τ GRASP')
plt.plot(np.r_[0,maxAOD],np.r_[0,maxAOD],'k')
plt.xlim([0,maxAOD])
plt.ylim([0,maxAOD])

maxReff = 5
plt.figure()
reff = [np.sum(measData['reff'][ind,-Nprofs:]*measData['tau'][ind,-Nprofs:])/np.sum(measData['tau'][ind,-Nprofs:])*1e6 for ind in indRng]
plt.plot(reff,[rslt['rEff'] for rslt in gDB.rslts],'.')
plt.plot(np.r_[0,maxReff],np.r_[0,maxReff],'k')
plt.xlabel('$r_{eff}$ True')
plt.ylabel('$τ_{eff}$ GRASP')
plt.xlim([0,maxReff])
plt.ylim([0,maxReff])


