#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", "GRASP_scripts"))
from runGRASP import graspDB, graspRun, pixel
from MADCAP_functions import readVILDORTnetCDF

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
#basePath = '/home/respinosa/ReedWorking/' # Uranus
dayStr = '20060901'
dirGRASPworking = False # use sytem temp directories as path to store GRASP SDATA and output files 
pathYAML = os.path.join(basePath, 'Local_Code_MacBook/MADCAP_Analysis/YAML_settingsFiles/settings_HARP_16bin_1lambda.yml') # path to GRASP YAML file
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_nosurface_PP_SSCORR_OUTGOING/calipso-g5nr.vlidort.vector.LAMBERTIAN.%dd00.nc4')
#binPathGRASP = '/usr/local/bin/grasp' # path to grasp binary
binPathGRASP = os.path.join(basePath, 'Local_Code_MacBook/grasp_open/build/bin/grasp')
savePath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_nosurface_PP_SSCORR_OUTGOING/rayleigh_bench_MS_azmth.pkl')

# Constants
wvls = [0.865] # wavelengths to read from levC files
lndPrct = 100; # land cover amount (%)
grspChnkSz = 25 # number of pixles in a single SDATA file
orbHghtKM = 700 # sensor height (km)
GRASP_MIN = 1e-6 # SDATA measurements smaller than GRASP_MIN will be replaced by GRASP_MIN
graspInputs = 'IQU' # 'Ionly' (intensity), 'DOLP' (I & DOLP), 'IQU' (1st 3 stokes), IQU_SURF (IQU for surface only)
maxCPUs = 3; # maximum number of simultaneous grasp run threads
solar_zenith = 30
solar_azimuth = 0

# Variable to read in from radiance netCDF file, note that many variables are hard coded below
# Also, this currently stores wavelength independent data Nwvlth times but the method is simple
#varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'toa_reflectance', 'sensor_zenith', 'sensor_azimuth']
varNames = ['I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'ROT','surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'toa_reflectance', 'sensor_zenith', 'sensor_azimuth']


# Read in radiances, solar spectral irradiance and find reflectances
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls, datSizeVar = 'sensor_azimuth')

# Read in levelB data to obtain pressure and then surface altitude
maslTmp = np.r_[0]
for i in range(len(wvls)): measData[i]['masl'] = maslTmp

# Generate GRASPruns from cases
graspObjs = []
Npix = measData[0]['I'].shape[1]
#Npix = 4 # HACK to test with only 6 pixels
strtInds = np.r_[0:Npix:grspChnkSz]
for strtInd in strtInds:
    gObj = graspRun(pathYAML, orbHghtKM, dirGRASPworking)
    endInd = min(strtInd+grspChnkSz, Npix)
    for ind in range(strtInd, endInd):
        dtNm = measData[0]['dtNm'][ind]
        lon = 0
        lat = 0
        masl = 1743.45 # ROT=0.01258532 @ 865nm; benchmark netCDF ROT=0.0125853186 => agreement to 0.000011%
        nowPix = pixel(dtNm, 1, 1, lon, lat, masl, lndPrct)
        sza = solar_zenith # assume instantaneous measurement
        for l,wl in enumerate(wvls): # LOOP OVER WAVELENGTHS
             thtv = np.abs(measData[l]['sensor_zenith'][:])
             nbvm = thtv.shape[0] 
             if graspInputs.upper()=='Ionly':
                 msTyp = np.r_[41]
                 msrmnts = np.r_[measData[l]['I'][:,ind]]
             elif graspInputs.upper()=='DOLP':
                 msTyp = np.r_[41, 44]
                 msrmnts = np.r_[measData[l]['I'][:,ind], measData[l]['DOLP'][:,ind]]
             elif graspInputs.upper()=='IQU':
                 msTyp = np.r_[41, 42, 43]
                 msrmnts = np.r_[measData[l]['I'][:,ind], measData[l]['Q'][:,ind], measData[l]['U'][:,ind]]
#                 msrmnts = np.r_[measData[l]['I'][ind,:], measData[l]['Q_scatplane'][ind,:], measData[l]['U_scatplane'][ind,:]]
             elif graspInputs.upper()=='IQU_SURF':
                 msTyp = np.r_[41, 42, 43]
                 msrmnts = np.r_[measData[l]['I_surf'][:,ind], measData[l]['Q_surf'][:,ind], measData[l]['U_surf'][:,ind]]
             else:
                 assert False, '%s is unrecognized value for graspInputs [Ionly,DOLP,IQU]' % graspInputs
             msrmnts[np.abs(msrmnts) < GRASP_MIN] = GRASP_MIN # HINT: could change Q or U sign but still small absolute shift
             nip = msTyp.shape[0]
             thtv = np.tile(thtv, nip) # ex. 11, 35, 55, 11, 35, 55...       
             phi = np.abs(np.tile(solar_azimuth - measData[l]['sensor_azimuth'][ind], nbvm*nip)) # HINT: might cause phi<-180 which GRASP technically doesn't like
             nowPix.addMeas(wl, msTyp, np.repeat(nbvm, nip), sza, thtv, phi, msrmnts)                 
        gObj.addPix(nowPix)
    graspObjs.append(gObj)

#plt I
#l=5; pltVar = 'I'
#azimth=measData[l]['sensor_azimuth']*np.pi/180
#zenith=measData[l]['sensor_zenith']
#data=measData[l][pltVar].T
#r, theta = np.meshgrid(zenith, azimth)
#fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
#c = ax.contourf(theta, r, data)
#cb = plt.colorbar(c)
#cb.ax.set_ylabel(pltVar)
#plt.tight_layout()

# Write SDATA, run GRASP and read in results
gDB = graspDB(graspObjs)
gDB.processData(maxCPUs, binPathGRASP, savePath)

