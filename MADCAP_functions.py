#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime as dt
from datetime import timedelta
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
    
def ordinal2datetime(ordinal):
    dtObjDay = dt.fromordinal(np.int(np.floor(ordinal)))
    dtObjTime = timedelta(seconds=np.remainder(ordinal, 1)*86400)
    dtObj = dtObjDay + dtObjTime
    return dtObj