#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import re
from datetime import datetime as dt
import warnings
import os
import sys
sys.path.append(os.path.join("..", "GRASP_PythonUtils"))
from runGRASP import graspRun, pixel

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
#basePath = '/home/respinosa/ReedWorking/' # Uranus
dayStr = '20060901'
dirGRASPworking = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/graspWorking') # path to store GRASP SDATA and output files
pathYAML = os.path.join(basePath, 'Remote_Sensing_Analysis/GRASP_PythonUtils/settings_HARP_16bin_6lambda.yml') # path to GRASP YAML file
#radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.vlidort.vector.MCD43C.'+dayStr+'_00z_%dd00nm.nc4')
#radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase_regenerated/calipso-g5nr.vlidort.vector.MCD43C.'+dayStr+'_00z_%dd00nm.nc4')
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase_regenerated/calipso-g5nr.vlidort.vector.MCD43C_noBPDF.'+dayStr+'_00z_%dd00nm.nc4')
lidarFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lc2.ext.'+dayStr+'_00z_%dd00nm.nc4')
levBFN = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.aer_Nv.'+dayStr+'_00z.nc4')
lndCvrFN = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.land_cover.'+dayStr+'_00z.nc4')

# Constants
#wvls = [0.410, 0.440, 0.470, 0.550, 0.670, 0.865, 1.020, 1.650, 2.100] # wavelengths to read from levC files
#wvls = [0.440, 0.470, 0.550, 0.670, 0.865, 1.020] # wavelengths to read from levC files
wvls = [0.440, 0.550, 0.670, 0.865, 1.020, 2.100] # wavelengths to read from levC files
#wvls = [1.020, 2.100] # wavelengths to read from levC vlidort files
wvlsLidar = [0.532, 1.064] # wavelengths to read from levC lidar files
dateRegex = '.*([0-9]{8})_[0-9]+z.nc4$' # regex to pull date string from levBFN, should give 'YYYYMMDD'
scaleHght = 7640 # atmosphere scale height (meters)
stndPres = 1.01e5 # standard pressure (Pa)
lndPrct = 100; # land cover amount (%), ocean only for now
grspChnkSz = 300 # number of pixles in a single SDATA file
orbHghtKM = 700 # sensor height (km)

GRASP_MIN = 1e-6 # SDATA measurements smaller than GRASP_MIN will be replaced by GRASP_MIN

# Read in radiances, solar spectral irradiance and find reflectances
# Currently stores wavelength independent data Nwvlth times but this method is simple...
varNames = ['ROT', 'I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'toa_reflectance', 'solar_zenith', 'solar_azimuth', 'sensor_zenith', 'sensor_azimuth', 'time','trjLon','trjLat']
datStr = re.match(dateRegex, levBFN).group(1)
dayDtNm = dt.strptime(datStr, "%Y%m%d").toordinal()
Nwvlth = len(wvls)
measData = [{} for _ in range(Nwvlth)]
invldInd = np.array([])
warnings.simplefilter('ignore') # ignore missing_value not cast warning
for i,wvl in enumerate(wvls):
    radianceFN = radianceFNfrmtStr % int(wvl*1000)
    netCDFobj = Dataset(radianceFN)
    for varName in varNames:
        measData[i][varName] = np.array(netCDFobj.variables[varName])       
    invldInd = np.append(invldInd, np.nonzero((measData[i]['I']<0).any(axis=1))[0])
    netCDFobj.close()
invldInd = np.array(np.unique(invldInd), dtype='int') # only take points w/ I>0 at all wavelengths & angles    
warnings.simplefilter('always')
for i in range(Nwvlth):
    for varName in np.setdiff1d(varNames, 'sensor_zenith'):
        measData[i][varName] = np.delete(measData[i][varName], invldInd, axis=0)
    measData[i]['DOLP'] = np.sqrt(measData[i]['Q']**2+measData[i]['U']**2)/measData[i]['I']
    measData[i]['I'] = measData[i]['I']*np.pi # GRASP "I"=R=L/FO*pi 
    measData[i]['Q'] = measData[i]['Q']*np.pi 
    measData[i]['U'] = measData[i]['U']*np.pi 
#    measData[i]['I'] = measData[i]['surf_reflectance']*np.cos(measData[i]['solar_zenith']*np.pi/180).reshape(-1,1) # GRASP "I"=R=L/FO*pi
#    measData[i]['Q'] = measData[i]['surf_reflectance_Q']*np.cos(measData[i]['solar_zenith']*np.pi/180).reshape(-1,1) 
#    measData[i]['U'] = measData[i]['surf_reflectance_U']*np.cos(measData[i]['solar_zenith']*np.pi/180).reshape(-1,1)
#    measData[i]['DOLP'] = np.sqrt(measData[i]['Q']**2+measData[i]['U']**2)/measData[i]['I']
    measData[i]['dtNm'] = dayDtNm + measData[i]['time']/86400



# Read in levelB data to obtain pressure and then surface altitude
netCDFobj = Dataset(levBFN)
warnings.simplefilter('ignore') # ignore missing_value not cast warning
surfPres = np.array(netCDFobj.variables['PS'])
warnings.simplefilter('always')
surfPres = np.delete(surfPres, invldInd)
maslTmp = [scaleHght*np.log(stndPres/PS) for PS in surfPres]
for i in range(Nwvlth): measData[i]['masl'] = maslTmp

# Generate GRASPruns from cases
#msTyp = np.r_[41, 44] # reflectance, DOLP; currently assumes all msTyp's have same nbvm
msTyp = np.r_[41, 42, 43] # reflectance, Q, U; currently assumes all msTyp's have same nbvm
graspObjs = []
nip = msTyp.shape[0]
Npix = measData[0]['I'].shape[0]
strtInds = np.r_[0:Npix:grspChnkSz]
#strtInds = [0] # HACK!!!
for strtInd in strtInds:
    gObj = graspRun(pathYAML, orbHghtKM, dirGRASPworking)
    endInd = min(strtInd+grspChnkSz, Npix)
#    endInd = strtInd+10 # HACK!!!
    for ind in range(strtInd, endInd):
        dtNm = measData[0]['dtNm'][ind]
        lon = measData[0]['trjLon'][ind]
        lat = measData[0]['trjLat'][ind]
        masl = max(measData[0]['masl'][ind], -100) # defualt GRASP build complains below -100m  
#        masl = 20000 #HACK to remove most of Rayleigh signal
        nowPix = pixel(dtNm, 1, 1, lon, lat, masl, lndPrct)
        sza = measData[0]['solar_zenith'][ind] # assume instantaneous measurement
        for l,wl in enumerate(wvls): # LOOP OVER WAVELENGTHS
             phi = measData[l]['solar_azimuth'][ind] - measData[l]['sensor_azimuth'][ind,:] # HINT: might cause phi<-180 which GRASP technically doesn't like
             nbvm = phi.shape[0] 
             phi = np.tile(phi, nip) # ex. 11, 35, 55, 11, 35, 55...
             thtv = np.abs(np.tile(measData[l]['sensor_zenith'], nip))
#             msrmnts = np.r_[measData[l]['I'][ind,:], measData[l]['DOLP'][ind,:]]
             msrmnts = np.r_[measData[l]['I'][ind,:], measData[l]['Q'][ind,:], measData[l]['U'][ind,:]]
             msrmnts[np.abs(msrmnts) < GRASP_MIN] = GRASP_MIN # HINT: could change sign but doesn't matter for DoLP
             nowPix.addMeas(wl, msTyp, np.repeat(nbvm, nip), sza, thtv, phi, msrmnts)
        gObj.addPix(nowPix)
    graspObjs.append(gObj)

# Write SDATA, run GRASP and read in results
graspObjs[0].writeSDATA()
#graspObjs[0].runGRASP()
#rslts = graspObjs[0].readOutput()

# Read in model "truth" from levC lidar file
varNames = ['reff', 'refi', 'refr', 'ssa', 'tau']
Nwvlth = len(wvlsLidar)
trueData = [{} for _ in range(Nwvlth)]
warnings.simplefilter('ignore') # ignore missing_value not cast warning
for i,wvl in enumerate(wvlsLidar):
    lidarFN = lidarFNfrmtStr % int(wvl*1000)
    netCDFobj = Dataset(lidarFN)
    for varName in varNames:
        trueData[i][varName] = np.array(netCDFobj.variables[varName])      
    netCDFobj.close()
warnings.simplefilter('always')
for i in range(Nwvlth): 
    for varName in varNames:
        trueData[i][varName] = np.delete(trueData[i][varName], invldInd, axis=0)
    tauKrnl = trueData[i]['tau']
    trueData[i]['tau'] = np.sum(trueData[i]['tau'], axis=1)
    tauKrnl = tauKrnl/trueData[i]['tau'].reshape(Npix,1)
    for varName in np.setdiff1d(varNames, 'tau'):
        trueData[i][varName] = np.sum(tauKrnl*trueData[i][varName], axis=1)

warnings.simplefilter('ignore') # ignore missing_value not cast warning
netCDFobj = Dataset(lndCvrFN)
trueData[0]['BPDFcoef'] = np.array(netCDFobj.variables['BPDFcoef'])
netCDFobj.close()
warnings.simplefilter('always')
trueData[0]['BPDFcoef'] = np.delete(trueData[0]['BPDFcoef'], invldInd)
