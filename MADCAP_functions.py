#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime as dt
import warnings
from netCDF4 import Dataset
import hashlib
import fnmatch
import os

def loadVARSnetCDF(filePath, varNames=None):
    warnings.simplefilter('ignore') # ignore missing_value not cast warning
    measData = dict()
    netCDFobj = Dataset(filePath)
    if varNames is None: varNames = netCDFobj.variables.keys()
    for varName in varNames:
        if varName in netCDFobj.variables.keys():
            measData[varName] = np.array(netCDFobj.variables[varName])
        else:
            warnings.warn('Could not find %s variable in netCDF data' % varName)
    netCDFobj.close()
    warnings.simplefilter('always')
    return measData 

def readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls, datStr = '20000101', datSizeVar = 'sensor_zenith'):
    dayDtNm = dt.strptime(datStr, "%Y%m%d").toordinal()
    Nwvlth = len(wvls)
    measData = [{} for _ in range(Nwvlth)]
    invldInd = np.array([])
    varNames = np.union1d(varNames, datSizeVar)
    for i,wvl in enumerate(wvls):
        radianceFN = radianceFNfrmtStr % (wvl*1000)
        measData[i] = loadVARSnetCDF(radianceFN, varNames)
        invldInd = np.append(invldInd, np.nonzero((measData[i]['I']<0).any(axis=1))[0])
    invldInd = np.array(np.unique(invldInd), dtype='int') # only take points w/ I>0 at all wavelengths & angles  
    assert datSizeVar in measData[i].keys(), 'datSizeVar was not found in the netCDF file!'
    for i in range(Nwvlth):
        for varName in np.setdiff1d(list(measData[i].keys()), 'sensor_zenith'):
            measData[i][varName] = np.delete(measData[i][varName], invldInd, axis=0)
        measData[i]['dtNm'] = dayDtNm + np.r_[0:measData[0][datSizeVar].shape[0]]/24
        if 'I' in measData[i].keys():
            measData[i]['I'] = measData[i]['I']*np.pi # GRASP "I"=R=L/FO*pi 
            if 'Q' in measData[i].keys(): measData[i]['Q'] = measData[i]['Q']*np.pi 
            if 'U' in measData[i].keys(): measData[i]['U'] = measData[i]['U']*np.pi
            if 'Q' in measData[i].keys() and 'U' in measData[i].keys():
                measData[i]['DOLP'] = np.sqrt(measData[i]['Q']**2+measData[i]['U']**2)/measData[i]['I']
        if 'surf_reflectance' in measData[i].keys():
            measData[i]['I_surf'] = measData[i]['surf_reflectance']*np.cos(30*np.pi/180)
            if 'surf_reflectance_Q_scatplane' in measData[i].keys():
                measData[i]['Q_surf'] = measData[i]['surf_reflectance_Q_scatplane']*np.cos(30*np.pi/180)
                measData[i]['U_surf'] = measData[i]['surf_reflectance_U_scatplane']*np.cos(30*np.pi/180)
                print('Q[U]_surf derived from surf_reflectance_Q[U]_scatplane (scat. plane system)')
            else:
                measData[i]['Q_surf'] = measData[i]['surf_reflectance_Q']*np.cos(30*np.pi/180)
                measData[i]['U_surf'] = measData[i]['surf_reflectance_U']*np.cos(30*np.pi/180)
                print('Q[U]_surf derived from surf_reflectance_Q[U] (meridian system)')
            if (measData[i]['I_surf'] > 0).all(): # TODO: This will produce NaN in all DOLP if any I_surf<0
                measData[i]['DOLP_surf'] = np.sqrt(measData[i]['Q_surf']**2+measData[i]['U_surf']**2)/measData[i]['I_surf']
            else:
                measData[i]['DOLP_surf'] = np.full(measData[i]['I_surf'].shape, np.nan)
        if 'Q_scatplane' in varNames: measData[i]['Q_scatplane'] = measData[i]['Q_scatplane']*np.pi
        if 'U_scatplane' in varNames: measData[i]['U_scatplane'] = measData[i]['U_scatplane']*np.pi
    return measData, invldInd # measData has relevent netCDF data, invldInd has times that were deemed invalid

def hashFileSHA1(filePath):
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open(filePath, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

def findNewestMatch(directory, pattern='*'):
    nwstTime = 0
    for file in os.listdir(directory):
        filePath = os.path.join(directory, file)
        if fnmatch.fnmatch(file, pattern) and os.path.getmtime(filePath) > nwstTime:
            nwstTime = os.path.getmtime(filePath)
            newestFN = filePath 
    if nwstTime > 0:
        return newestFN
    else:
        return ''
            
        
        
        