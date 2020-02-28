#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from datetime import datetime as dt
from datetime import timedelta
import warnings
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import hashlib
import fnmatch
import os

def loadVARSnetCDF(filePath, varNames=None, verbose=False):
    warnings.simplefilter('ignore') # ignore missing_value not cast warning
    measData = dict()
    netCDFobj = Dataset(filePath)
    if varNames is None: varNames = netCDFobj.variables.keys()
    for varName in varNames:
        if varName in netCDFobj.variables.keys():
            measData[varName] = np.array(netCDFobj.variables[varName])
        elif verbose:
            print("\x1b[1;35m Could not find \x1b[1;31m%s\x1b[1;35m variable in netCDF file: %s\x1b[0m" % (varName,filePath))
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

def KDEhist2D(x,y, axHnd=None, res=100, xrng=None, yrng=None, sclPow=1, cmap = 'BuGn', clbl='Probability Density (a.u.)'):
    # set plot range
    xmin = xrng[0] if xrng else x.min()
    xmax = xrng[1] if xrng else x.max()
    ymin = yrng[0] if yrng else y.min()
    ymax = yrng[1] if yrng else y.max()
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:complex(0,res), ymin:ymax:complex(0,res)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    # Plot results
    if not axHnd:
        fig = plt.figure()
        axHnd = fig.gca()
    axHnd.set_xlim(xmin, xmax)
    axHnd.set_ylim(ymin, ymax)
    # Contourf plot
    objHnd = axHnd.contourf(xx, yy, f**sclPow, 256, cmap=cmap)
    clrHnd = plt.colorbar(objHnd, ax=axHnd)
#    objHnd.set_clim(vmin=0)
    tckVals = clrHnd.get_ticks()**(1/sclPow)/np.max(clrHnd.get_ticks()**(1/sclPow))
    clrHnd.set_ticklabels(['%4.1f' % x for x in 100*tckVals])
    clrHnd.set_label(clbl)
    return axHnd