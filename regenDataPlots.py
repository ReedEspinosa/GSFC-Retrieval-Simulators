#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import re
from datetime import datetime as dt
import warnings
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
#basePath = '/home/respinosa/ReedWorking/' # Uranus
dayStr = '20060901'
dirGRASPworking = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/graspWorking') # path to store GRASP SDATA and output files
pathYAML = os.path.join(basePath, 'Remote_Sensing_Analysis/GRASP_PythonUtils/settings_HARP_16bin_6lambda.yml') # path to GRASP YAML file
#radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.vlidort.vector.MCD43C.'+dayStr+'_00z_%dd00nm.nc4')
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase_regenerated/calipso-g5nr.vlidort.vector.MCD43C.'+dayStr+'_00z_%dd00nm.nc4')
lidarFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lc2.ext.'+dayStr+'_00z_%dd00nm.nc4')
levBFN = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.aer_Nv.'+dayStr+'_00z.nc4')
lndCvrFN = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.land_cover.'+dayStr+'_00z.nc4')

# Constants
#wvls = [0.410, 0.440, 0.470, 0.550, 0.670, 0.865, 1.020, 1.650, 2.100] # wavelengths to read from levC files
#wvls = [0.440, 0.470, 0.550, 0.670, 0.865, 1.020] # wavelengths to read from levC files
wvls = [0.440, 0.550, 0.670, 0.865, 1.020, 2.100] # wavelengths to read from levC files
#wvls = [0.440, 0.550, 0.670, 0.865] # wavelengths to read from levC vlidort files
wvlsLidar = [0.532, 1.064] # wavelengths to read from levC lidar files
dateRegex = '.*([0-9]{8})_[0-9]+z.nc4$' # regex to pull date string from levBFN, should give 'YYYYMMDD'
scaleHght = 7640 # atmosphere scale height (meters)
stndPres = 1.01e5 # standard pressure (Pa)
lndPrct = 100; # land cover amount (%), ocean only for now
grspChnkSz = 120 # number of pixles in a single SDATA file
orbHghtKM = 700 # sensor height (km)


# Read in radiances, solar spectral irradiance and find reflectances
# Currently stores wavelength independent data Nwvlth times but this method is simple...
varNames = ['I', 'Q', 'U', 'surf_reflectance', 'surf_reflectance_Q', 'surf_reflectance_U', 'toa_reflectance', 'solar_zenith', 'solar_azimuth', 'sensor_zenith', 'sensor_azimuth', 'time','trjLon','trjLat']
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
    measData[i]['Q'] = measData[i]['U']*np.pi 
    measData[i]['U'] = measData[i]['Q']*np.pi 
    measData[i]['dtNm'] = dayDtNm + measData[i]['time']/86400


# Read in levelB data to obtain pressure and then surface altitude
netCDFobj = Dataset(levBFN)
warnings.simplefilter('ignore') # ignore missing_value not cast warning
surfPres = np.array(netCDFobj.variables['PS'])
warnings.simplefilter('always')
surfPres = np.delete(surfPres, invldInd)
maslTmp = [scaleHght*np.log(stndPres/PS) for PS in surfPres]
for i in range(Nwvlth): measData[i]['masl'] = maslTmp

# Read in model "truth" from levC lidar file
Npix = measData[0]['I'].shape[0]
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

# PLOTS
phi = np.reshape(measData[i]['solar_azimuth'], (-1,1)) - measData[i]['sensor_azimuth']
surfDmInd = trueData[1]['tau'] == trueData[1]['tau'][0]

#spctDOLP = np.array([np.mean(100*MD['toa_reflectance'][surfDmInd,:]*MD['DOLP'][surfDmInd,:], axis=0) for MD in measData])
spctDOLP = np.array([np.mean(100*np.sqrt(MD['surf_reflectance_Q'][surfDmInd,:]**2 + MD['surf_reflectance_U'][surfDmInd,:]**2), axis=0) for MD in measData])
plt.figure()
[plt.plot(wvls, DOLP,  c=plt.cm.jet(20*i)) for i,DOLP in enumerate(spctDOLP.T)]
plt.legend(['θ=%d' % val for val in measData[0]['sensor_zenith']], ncol=3)

rWvInd = 4;
#for i in range(np.sum(surfDmInd)):
for i in range(1):
    ind = np.nonzero(surfDmInd)[0][i]
    surf_DoLP = np.sqrt(measData[rWvInd]['surf_reflectance_Q'][ind,:]**2 + measData[rWvInd]['surf_reflectance_U'][ind,:]**2)
    phiN = phi[ind, :]
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='polar')
#    c = ax1.scatter(phiN*np.pi/180, np.abs(measData[0]['sensor_zenith']), c=100*measData[rWvInd]['toa_reflectance'][ind,:]*measData[rWvInd]['DOLP'][ind,:])
    c = ax1.scatter(phiN*np.pi/180, np.abs(measData[0]['sensor_zenith']), c=100*surf_DoLP)
    ax1.scatter(0, np.abs(measData[0]['solar_zenith'][ind]), s=100, facecolors='none', edgecolors='r')
    vmin,vmax = c.get_clim() #-- obtaining the colormap limits
    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) #-- Defining a normalised scale
    ax3 = fig.add_axes([0.8, 0.1, 0.03, 0.8]) #-- Creating a new axes at the right side
    cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm) #-- Plotting the colormap in the created axes
    fig.subplots_adjust(left=0.05,right=0.85)
    plt.ylabel("100 x Polarized Reflectance (%5.3f um)" % wvls[rWvInd])
    plt.title('T = %d sec' % measData[0]['time'][ind])


# OTHER CODE
I = np.array(netCDFobj2.variables['I'])
Q = np.array(netCDFobj2.variables['Q'])
U = np.array(netCDFobj2.variables['U'])
DoLP = np.sqrt(Q**2+U**2)/I
phi = np.array(netCDFobj2.variables['sensor_azimuth'])*np.pi/180
theta = np.array(netCDFobj2.variables['sensor_zenith'])
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='polar')
mshGrd = np.meshgrid(phi, theta)
c = ax1.scatter(mshGrd[0].reshape(-1), mshGrd[1].reshape(-1), c=I.reshape(-1))

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='polar')
mshGrd = np.meshgrid(phi, theta)
c = ax1.scatter(mshGrd[0].reshape(-1), mshGrd[1].reshape(-1), c=DoLP.reshape(-1))
vmin,vmax = c.get_clim() #-- obtaining the colormap limits
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) #-- Defining a normalised scale
ax3 = fig.add_axes([0.8, 0.1, 0.03, 0.8]) #-- Creating a new axes at the right side
cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm) #-- Plotting the colormap in the created axes
fig.subplots_adjust(left=0.05,right=0.85)





