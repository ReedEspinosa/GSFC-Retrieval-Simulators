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
import pickle
from glob import glob
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


def downsample1d(X,Y,Xnew):
    """ Returns 1D array with the mean of Y over the interval Xnew-ΔXnew to Xnew+ΔXnew."""
    from scipy import integrate, interpolate
    assert len(X)==len(Y), 'X and Y must have the same length.'
    assert np.all(Xnew==np.sort(Xnew)), 'The values of Xnew must be in ascending order.' 
    # edge treatment 1: mean(Y(X[0] to X[0]+ΔXnew))
#     XnewBnds = np.interp(np.r_[0.5:len(Xnew)-1], np.r_[0:len(Xnew)], Xnew)
#     XnewBnds = np.r_[Xnew[0], XnewBnds, Xnew[-1]]
    # edge treatment 2: {Y[0]+mean(Y(X[0] to X[0]+ΔXnew))}/2 – produces curves that _look_ more like original Y    
    f = interpolate.interp1d(np.r_[0:len(Xnew)], Xnew, fill_value="extrapolate")
    XnewBnds = f(np.r_[-0.5:len(Xnew)])
    Ynew = np.empty(len(Xnew))
    for i,(lb,ub) in enumerate(zip(XnewBnds[:-1], XnewBnds[1:])):
        bndsY = np.interp([lb, ub], X, Y)
        chnkInd = np.logical_and(X>lb, X<ub)
        chnkX = np.r_[lb, X[chnkInd], ub]
        chnkY = np.r_[bndsY[0], Y[chnkInd], bndsY[1]]
        Ynew[i] = integrate.simps(chnkY, chnkX)/(ub-lb)
    return Ynew

def hashFileSHA1(filePaths, quick=False):
    """
    filePaths -> string or list of string with file path(s) to hash; no NUMPY arrays!
    quick -> only read the first block (~65kB from files) 
    """
    BLOCKSIZE = 65536
    if type(filePaths) is np.ndarray: filePaths = filePaths.tolist()
    if not type(filePaths) is list: filePaths = [filePaths]
    hasher = hashlib.sha1()
    for fp in filePaths:
        with open(fp, 'rb') as afile:
            buf = afile.read(BLOCKSIZE)
            satisfied = False
            while not satisfied:
                hasher.update(buf)
                buf = afile.read(BLOCKSIZE)
                satisfied = len(buf) == 0 or quick
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
    
def readPetesAngleFiles(fileDirPath, nAng=10, verbose=False, saveData=True, loadData=True):
    """
    Read in the randomly sampled angle text files generated by Pete.
    fileDirPath - Full path directory from which glob will grab ALL text files (should be free of extraneous *.txt files) 
    nAng - The number of viewing angles per ground pixel
    saveData - Save the results to the directory in fileDirPath in the form of a pickle
    loadData - Load the results from a previous run, if they exist
    """
    fileNames = np.sort(glob(os.path.join(fileDirPath, '*.txt')))
    hashTag = hashFileSHA1(fileNames, quick=True)[0:16] # if files differ after first 65kB we will miss it
    pklPath = os.path.join(fileDirPath, 'readPetesAngleFiles_ALLdata_'+hashTag+'.pkl')
    if loadData == True:
        try:
            with open(pklPath, 'rb') as f:
                angDict = pickle.load(f)
            if verbose: print('Data loaded from %s' % pklPath)
            return angDict
        except EnvironmentError:
            if verbose: print('Could not load valid pickle data from %s. Processing raw text files...' % pklPath)    
    angDict = {k: [] for k in ['lon','lat','datetime','vis','sza','fis','sca']}
    for fn in fileNames:
        if verbose: print('Processing %s...' % os.path.basename(fn))
        peteData = np.genfromtxt(fn, delimiter=',', skip_header=1, dtype=None, encoding='UTF8')
        for ind in range(0, len(peteData), nAng):
            angDict['lon'].append(peteData[ind][0])
            angDict['lat'].append(peteData[ind][1])
            dtStr = peteData[ind][2] + '%02d' % np.floor(95/60) + '%02d' % (95 % 60)
            angDict['datetime'].append(dt.strptime(dtStr, '%Y%m%d_%H00z%M%S'))
            angDict['vis'].append([x[5] for x in peteData[ind:ind+nAng]])
            angDict['sza'].append(np.mean([x[6] for x in peteData[ind:ind+nAng]]))
            fis = np.array([x[8]-x[7] for x in peteData[ind:ind+nAng]])
            fis[fis<0] = fis[fis<0] + 360
            angDict['fis'].append(fis)
            angDict['sca'].append([x[9] for x in peteData[ind:ind+nAng]])
    for key in angDict.keys(): angDict[key] = np.array(angDict[key]) # convert everything to numpy
    if saveData:
        with open(pklPath, 'wb') as f:
            pickle.dump(angDict, f, pickle.HIGHEST_PROTOCOL)
        if verbose: print('Data saved to %s' % pklPath)
    return angDict