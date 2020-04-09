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
    badDataCuttoff = 1e12 # values larger than this will be replaced with NaNs
    if varNames: assert isinstance(varNames, (list, np.ndarray)), 'varNames must be a list or numpy array!'
    measData = dict()
    netCDFobj = Dataset(filePath)
    if varNames is None: varNames = netCDFobj.variables.keys()
    for varName in varNames:
        if varName in netCDFobj.variables.keys():
            with warnings.catch_warnings(): 
                warnings.simplefilter('ignore', category=UserWarning) # ignore missing_value not cast warning
                measData[varName] = np.array(netCDFobj.variables[varName])
            if 'float' in measData[varName].dtype.name and np.any(measData[varName] > badDataCuttoff):
                if np.issubdtype(measData[varName].dtype, np.integer): # numpy ints can't be NaN
                    measData[varName] = measData[varName].astype(np.float32)
                measData[varName][measData[varName] > badDataCuttoff] = np.nan
        elif verbose:
            print("\x1b[1;35m Could not find \x1b[1;31m%s\x1b[1;35m variable in netCDF file: %s\x1b[0m" % (varName,filePath))
    netCDFobj.close()
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
    
def MAIAC_BRDF_stats(fileName, normByIso=False, fltrPrct=100, λout=None):
    from pyhdf.SD import SD, SDC
    from sklearn.decomposition.pca import PCA
    print('--- MAIAC_BRDF_stats <> V9 ---')
    λHDF = np.r_[0.645, 0.8585, 0.469, 0.555, 1.24, 1.64, 2.13, 0.412]
    λOrder = λHDF.argsort()
    if λout is None: λout = np.r_[0.360	,0.380,0.410,0.550,0.670,0.870,1.550,1.650]
    hdf = SD(fileName, SDC.READ)
    vals = dict()
    badInd = []
    Nλ = len(λHDF)
    keys = ['Kiso', 'Kvol', 'Kgeo']
    for key in keys:
        hdfStrct = hdf.select(key)
        vals[key] = hdfStrct[:,:,:].reshape([Nλ,-1])
        badInd.append(np.any(vals[key]==hdfStrct.attributes()['_FillValue'], axis=0))
        vals[key] = vals[key][λOrder,:]*hdfStrct.attributes()['scale_factor']
        if normByIso and key is not 'Kiso':
            with np.errstate(divide='ignore'): # Kiso may be zero
                with np.errstate(invalid='ignore'): # apparently numpy gives two warnings for divide by zero? 
                    vals[key] = vals[key]/vals['Kiso'] 
    badInds = np.any(badInd, axis=0)
    for key in keys: vals[key] = vals[key][:, ~badInds]
    print('%d/%d remaining pixels had at least one fill value' % (np.sum(badInds), len(badInds)))
    if fltrPrct < 100:
        badInd = []
        for key in keys:
            upBnds = np.atleast_2d(np.percentile(vals[key],fltrPrct, axis=1)).T
            badInd.append(np.any(vals[key] > upBnds, axis=0))
            lowBnds = np.atleast_2d(np.percentile(vals[key],100-fltrPrct, axis=1)).T
            badInd.append(np.any(vals[key] < lowBnds, axis=0))
        badInds = np.any(badInd, axis=0)
        for key in keys: vals[key] = vals[key][:, ~badInds]
        print('%d/%d pixels had at least one outlier' % (np.sum(badInds), len(badInds)))
    shiftWave = lambda a: np.array([np.interp(λout, λHDF[λOrder], b) for b in a.T]).T
    x = np.vstack([shiftWave(y) for y in vals.values()])
    print('kiso(x%d), kvol(x%d), kgeo(x%d)' % (len(λout),len(λout),len(λout)))
    print('wavelengths:' + ','.join(map(str, λout)))
    print('means:')
    [print(y) for y in (map(str, x.mean(axis=1)))]
    print('std. dev.:')
    [print(y) for y in (map(str, x.std(axis=1)))]
    print('-----')
    print('NDVI [(Kiso[5]-Kiso[4])/(Kiso[5]+Kiso[4])]')
    NDVIs = (x[5,:]-x[4,:])/(x[5,:]+x[4,:])
    print('mean = %7.4f' % NDVIs.mean())
    print('std. dev. = %7.4f' % NDVIs.std())
    print('-----')
    #PCA
    pca = PCA(n_components=24)
    xStnrd = (x-x.mean(axis=1)[:,None])/x.std(axis=1)[:,None] # standardize x
    pca.fit(xStnrd.T)
    varExpln = np.sum(pca.explained_variance_ratio_[0:5]*100)
    print('Percent of variance explained by first five principle components %5.2f%%' % varExpln) 
    varExpln = np.sum(pca.explained_variance_ratio_[0:(len(λout)+2)]*100)
    print('Percent of variance explained by first Nλ+2=%d principle components %5.2f%%' % (len(λout)+2, varExpln)) 
    return x
    