#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import re
from datetime import datetime as dt
sys.path.append(os.path.join("..", "GRASP_scripts"))
from runGRASP import graspDB, graspRun, pixel
from MADCAP_functions import readVILDORTnetCDF, hashFileSHA1, loadVARSnetCDF

# Paths to files
dayStr = '20060901'
basePath = '/Users/wrespino/Synced/' # NASA MacBook
rmtPrjctPath = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase_regenerated')
pathYAML = os.path.join(basePath, 'Local_Code_MacBook/MADCAP_Analysis/YAML_settingsFiles/settings_HARP_16bin_6lambda.yml') # path to GRASP YAML file
radianceFNfrmtStr = os.path.join(rmtPrjctPath, 'calipso-g5nr.vlidort.vector.MCD43C_noBPDF.'+dayStr+'_00z_%dd00nm.nc4')
binPathGRASP = os.path.join(basePath, 'Local_Code_MacBook/grasp_open/build/bin/grasp')
savePathTag = 'TEST' # preprend tag for save file, A-z and _ only
levBFN = os.path.join(rmtPrjctPath, 'calipso-g5nr.lb2.aer_Nv.'+dayStr+'_00z.nc4')
dirGRASPworking = False # Flase -> use sytem temp directories as path for GRASP SDATA & output files 

# Constants
#wvls = [0.410, 0.440, 0.470, 0.550, 0.670, 0.865, 1.020, 1.650, 2.100] # wavelengths to read from levC files
wvls = [0.440, 0.550, 0.670, 0.865, 1.020, 2.100] # wavelengths to read from levC files
dateRegex = '.*([0-9]{8})_[0-9]+z.nc4$' # regex to pull date string from levBFN, should give 'YYYYMMDD'
scaleHght = 8000 # scale height (meters) for presure to alt. conversion, 8km is consistent w/ GRASP
stndPres = 1.01e5 # standard pressure (Pa)
lndPrct = 100; # land cover amount (%), land only for now
graspInputs = 'IQU' # 'Ionly' (intensity), 'DOLP' (I & DOLP), 'IQU' (1st 3 stokes), IQU_SURF (IQU for surface only); must match YAML
orbHghtKM = 700 # sensor height (km)
GRASP_MIN = 1e-8 # measurements <GRASP_MIN will be replaced by GRASP_MIN in SDATA file
grspChnkSz = 2 # number of pixles in a single SDATA file
maxCPUs = 3; # maximum number of simultaneous grasp run threads

# Variable to read in from radiance netCDF file, note that many variables are hard coded below
varNames = ['ROT', 'I', 'Q', 'U', 'Q_scatplane', 'U_scatplane', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'toa_reflectance', 'solar_zenith', 'solar_azimuth', 'sensor_zenith', 'sensor_azimuth', 'time','trjLon','trjLat']

# append firt wavelength to savePath
waveTag = '_%dnm' % (wvls[0]*1000) if len(wvls)==1 else '_%dLambda' % len(wvls)
yamlTag = '_YAML%s' % hashFileSHA1(pathYAML)[0:8]
savePath = os.path.join(os.path.split(radianceFNfrmtStr)[0], savePathTag+waveTag+yamlTag+'.pkl')

# Read in radiances, solar spectral irradiance and find reflectances
datStr = re.match(dateRegex, levBFN).group(1)
dayDtNm = dt.strptime(datStr, "%Y%m%d").toordinal()
Nwvlth = len(wvls)
measData, invldInd = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls, datSizeVar = 'sensor_zenith')

# Read in levelB data to obtain pressure and then surface altitude
levB_data = loadVARSnetCDF(levBFN, varNames=['PS'])
surfPres = np.delete(levB_data['PS'], invldInd)
maslTmp = [scaleHght*np.log(stndPres/PS) for PS in surfPres]
for i in range(Nwvlth): measData[i]['masl'] = maslTmp

# Generate GRASPruns from cases
graspObjs = []
Npix = measData[0]['I'].shape[0]
#Npix = 6 # HACK to test with only 6 pixels
strtInds = np.r_[0:Npix:grspChnkSz]
for strtInd in strtInds:
    gObj = graspRun(pathYAML, orbHghtKM, dirGRASPworking)
    endInd = min(strtInd+grspChnkSz, Npix)
    for ind in range(strtInd, endInd):
        dtNm = measData[0]['dtNm'][ind]
        lon = measData[0]['trjLon'][ind]
        lat = measData[0]['trjLat'][ind]
        masl = max(measData[0]['masl'][ind], -100) # defualt GRASP build complains below -100m
        nowPix = pixel(dtNm, 1, 1, lon, lat, masl, lndPrct)
        sza = measData[0]['solar_zenith'][ind] # assume instantaneous measurement
        for l,wl in enumerate(wvls): # LOOP OVER WAVELENGTHS
             phi = measData[l]['solar_azimuth'][ind] - measData[l]['sensor_azimuth'][ind,:] 
             phi[phi<0] = phi[phi<0] + 360  # GRASP accuracy degrades when phi<0
             nbvm = phi.shape[0] 
             if graspInputs.upper()=='Ionly':
                 msTyp = np.r_[41]
                 msrmnts = np.r_[measData[l]['I'][ind,:]]
             elif graspInputs.upper()=='DOLP':
                 msTyp = np.r_[41, 44]
                 msrmnts = np.r_[measData[l]['I'][ind,:], measData[l]['DOLP'][ind,:]]
             elif graspInputs.upper()=='IQU':
                 msTyp = np.r_[41, 42, 43]
                 msrmnts = np.r_[measData[l]['I'][ind,:], measData[l]['Q'][ind,:], measData[l]['U'][ind,:]]
             elif graspInputs.upper()=='IQU_SURF':
                 msTyp = np.r_[41, 42, 43]
                 msrmnts = np.r_[measData[l]['I_surf'][ind,:], measData[l]['Q_surf'][ind,:], measData[l]['U_surf'][ind,:]]
             else:
                 assert False, '%s is unrecognized value for graspInputs [Ionly,DOLP,IQU,IQU_SURF]' % graspInputs
             smInd = np.abs(msrmnts)<GRASP_MIN
             msrmnts[smInd] = GRASP_MIN*(2*(msrmnts[smInd]>0).astype(int)-1) # this sets absolute minimum while preserving the sign
             nip = msTyp.shape[0]
             phi = np.tile(phi, nip) # ex. 11, 35, 55, 11, 35, 55...
             thtv = np.abs(np.tile(measData[l]['sensor_zenith'], nip))
             nowPix.addMeas(wl, msTyp, np.repeat(nbvm, nip), sza, thtv, phi, msrmnts)
        gObj.addPix(nowPix)
    graspObjs.append(gObj)

# Write SDATA, run GRASP and read in results
gDB = graspDB(graspObjs)
gDB.processData(maxCPUs, binPathGRASP, savePath)

