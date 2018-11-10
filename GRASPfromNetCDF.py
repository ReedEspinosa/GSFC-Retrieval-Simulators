#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import warnings

radianceFNfrmtStr = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.vlidort.vector.MCD43C.20060801_00z_%dd00nm.nc4'
levBFN = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.aer_Nv.20060801_00z.nc4'

# Constants
wvls = [0.410, 0.440, 0.470, 0.550, 1.020, 1.650, 2.100]
varNames = ['I', 'Q', 'U', 'solar_zenith', 'solar_azimuth', 'sensor_zenith', 'sensor_azimuth', 'time','trjLon','trjLat']
scaleHght = 7640 # meters
stndPres = 1.01e5 # Pa

# Read in radiances, solar spectral irradiance and find reflectances
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
    measData[i]['R'] = measData[i]['I']*np.pi # GRASP "I"=R=L/FO*pi
    measData[i]['DOLP'] = np.sqrt(measData[i]['Q']**2+measData[i]['U']**2)/measData[i]['I']

# Read in levelB data to obtain pressure and then surface altitude
netCDFobj = Dataset(levBFN)
warnings.simplefilter('ignore') # ignore missing_value not cast warning
surfPres = np.array(netCDFobj.variables['PS'])
warnings.simplefilter('always')
surfPres = np.delete(surfPres, invldInd)
maslTmp = [scaleHght*np.log(stndPres/PS) for PS in surfPres]
for i in range(Nwvlth): measData[i]['masl'] = maslTmp




