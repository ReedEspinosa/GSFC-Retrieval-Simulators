#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run PCA on Level B land surface data from Patricia's Files """

# import some standard modules
import os
import sys
import functools
import numpy as np
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
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
dimTot = data2D.shape[1]
pca = PCA(n_components=dimTot)
pca.fit(data2D)

for compKeep in range(10):
    print('First %d components explain %5.2f%% of the variance' % (compKeep, np.sum(pca.explained_variance_ratio_[0:compKeep]*100)))

compKeep = 5
wvlng = fwdData[0]['lambda']
Nwvlng = len(wvlng)
plt.figure()
plt.plot(wvlng, pca.components_[0:compKeep,0:Nwvlng].T)
plt.gca().set_prop_cycle(None)
plt.plot(wvlng, pca.components_[0:compKeep,Nwvlng:(2*Nwvlng)].T,'--')
plt.gca().set_prop_cycle(None)
plt.plot(wvlng, pca.components_[0:compKeep,(2*Nwvlng):(3*Nwvlng)].T,'-.')
plt.xlabel('Wavelength')
plt.ylabel('Princple Components')
plt.ion()
plt.show()

print('The mean of each of the %d dimensions was:' % dimTot)
print(pca.mean_)

print('Values for the first %d components are:' % compKeep)
isoVals = ''
volVals = ''
geoVals = ''
numFormat = '%6E,' 
newLineStr = '  &\n' + ' '.join(['' for _ in range(27)])
for wvInd in range(Nwvlng):
    for pcInd in range(compKeep):
        isoVals += numFormat % np.around(pca.components_[pcInd, wvInd], 9)
        volVals += numFormat % np.around(pca.components_[pcInd, wvInd+Nwvlng], 9)
        geoVals += numFormat % np.around(pca.components_[pcInd, wvInd+2*Nwvlng], 9)
    isoVals += newLineStr
    volVals += newLineStr
    geoVals += newLineStr
print('      ISO_PC = reshape((/ %s /), shape(ISO_PC))' % (isoVals.replace('E','D')))
print('      VOL_PC = reshape((/ %s /), shape(VOL_PC))' % (volVals.replace('E','D')))
print('      GEO_PC = reshape((/ %s /), shape(GEO_PC))' % (geoVals.replace('E','D')))










