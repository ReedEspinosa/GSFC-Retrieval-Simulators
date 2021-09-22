#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run PCA on Level B land surface data from Patricia's Files """

# import some standard modules
import os
import sys
import functools
import numpy as np
import pickle
from readOSSEnetCDF import osseData

# define other paths not having to do with the python code itself
osseDataPath = '/discover/nobackup/projects/gmao/osse2/pub/c1440_NR/OBS/A-CCP/' # base path for A-CCP OSSE data (contains gpm and ss450 folders)

# True to use 10k randomly chosen samples for defined month, or False to use specific day and hour defined below
random = True

# point in time to pull OSSE data (if random==true then day and hour below can be ignored)
year = 2006
# month = 1 SET BELOW IN LOOP
day = 1
hour = 0

# simulated orbit to use – gpm OR ss450
orbit = 'ss450'

# filter out pixels with mean SZA above this value (degrees)
maxSZA = 60
	    
# true to skip retrievals on land pixels
oceanOnly = 'land'
		
# wavelengths (μm); if we only want specific λ set it here, otherwise use every λ found in the netCDF files
wvls = [0.355, 0.36, 0.38, 0.41, 0.532, 0.55, 0.67, 0.87, 1.064, 1.55, 1.65]

# specific pixels to run; set to None to run all pixels (likely very slow)
pixInd = None
# pixInd = [i*3 for i in range(252)] # Aug 1, 1400Z, stripe up atlantic w/ marine, BB, marine, dust, dustymarine(?)

savePath = 'allRandomData_%s_maxSZA%d_V1.pkl' % (orbit, maxSZA)

# load the data
if os.path.exists(savePath):
    with open(savePath, 'rb') as save_fid: fwdData = pickle.load(save_fid)
    print('fwdData from all runs loaded from %s' % savePath)
else:
    fwdData = []
    for i in range(12):
        month = i+1
        # create osseData instance w/ pixels from specified date/time (detail on these arguments in comment near top of osseData class's __init__ near readOSSEnetCDF.py:30)
        od = osseData(osseDataPath, orbit, year, month, day, hour, random=random, wvls=wvls, pixInd=pixInd,
                      lidarVersion=None, maxSZA=maxSZA, oceanOnly=oceanOnly, loadPSD=False, verbose=True)

        # extract the simulated observations and pack them in GRASP_scripts rslts dictionary format
        fwdData  = fwdData + od.osse2graspRslts()
    with open(savePath, 'wb') as save_fid: pickle.dump(fwdData, save_fid)
    print('fwdData from all runs concatenated and saved to %s' % savePath)


# run the PCA
data2D = np.asarray([fd['brdf'].reshape(-1) for fd in fwdData])
pca = PCA(n_components=24)
pca.fit(data2D)
compKeep = 5
print('First %d components explain %4.1f%% of the variance' % (compKeep, np.sum(pca.explained_variance_ratio_[0:compKeep]*100))























