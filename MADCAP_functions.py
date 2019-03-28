#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime as dt
import warnings
from netCDF4 import Dataset

def readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls, datStr = '20000101'):
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
        measData[i]['dtNm'] = dayDtNm + np.r_[0:measData[0]['sensor_zenith'].shape[0]]/24
        measData[i]['DOLP'] = np.sqrt(measData[i]['Q']**2+measData[i]['U']**2)/measData[i]['I']
        measData[i]['I'] = measData[i]['I']*np.pi # GRASP "I"=R=L/FO*pi 
        measData[i]['Q'] = measData[i]['Q']*np.pi 
        measData[i]['U'] = measData[i]['U']*np.pi
        if 'Q_scatplane' in varNames: measData[i]['Q_scatplane'] = measData[i]['Q_scatplane']*np.pi
        if 'U_scatplane' in varNames: measData[i]['U_scatplane'] = measData[i]['U_scatplane']*np.pi
    return measData
