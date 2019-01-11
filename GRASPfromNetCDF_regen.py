#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset
import re
from datetime import datetime as dt
import warnings
import os
import sys
sys.path.append(os.path.join("..", "GRASP_PythonUtils"))
from runGRASP import graspRun, pixel

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
#basePath = '/home/respinosa/ReedWorking/' # Uranus
dayStr = '20060901'
dirGRASPworking = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/graspWorking') # path to store GRASP SDATA and output files
#pathYAML = os.path.join(basePath, 'Remote_Sensing_Analysis/GRASP_PythonUtils/settings_HARP_logNorm_1lambda.yml') # path to GRASP YAML file
pathYAML = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/settings_HARP_logNorm_1lambda.yml' # path to GRASP YAML file
#radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.vlidort.vector.MCD43C.'+dayStr+'_00z_%dd00nm.nc4')
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase_regenerated/calipso-g5nr.vlidort.vector.MCD43C.'+dayStr+'_00z_%dd00nm.nc4')
lidarFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lc2.ext.'+dayStr+'_00z_%dd00nm.nc4')
levBFN = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.aer_Nv.'+dayStr+'_00z.nc4')
lndCvrFN = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase/calipso-g5nr.lb2.land_cover.'+dayStr+'_00z.nc4')

# Constants
wvls = [0.865] # wavelengths to read from levC vlidort files
wvlsLidar = [0.532, 1.064] # wavelengths to read from levC lidar files
dateRegex = '.*([0-9]{8})_[0-9]+z.nc4$' # regex to pull date string from levBFN, should give 'YYYYMMDD'
scaleHght = 7640 # atmosphere scale height (meters)
stndPres = 1.01e5 # standard pressure (Pa)
lndPrct = 100; # land cover amount (%), ocean only for now
grspChnkSz = 120 # number of pixles in a single SDATA file
orbHghtKM = 700 # sensor height (km)

#cstFile = '/Users/wrespino/Downloads/calipso-g5nr.vlidort.vector.nosurface.%dd00.nc4'
#cstFile = '/Users/wrespino/Downloads/calipso-g5nr.vlidort.vector.BRDF.%dd00.nc4'
cstFile = '/Users/wrespino/Downloads/calipso-g5nr.vlidort.vector.BRDF_BPDF.%dd00.nc4'

# Read in radiances, solar spectral irradiance and find reflectances
# Currently stores wavelength independent data Nwvlth times but this method is simple...
datStr = re.match(dateRegex, levBFN).group(1)
dayDtNm = dt.strptime(datStr, "%Y%m%d").toordinal()

# Generate GRASPruns from cases
sza = 30
trgtAzmth = 55; # must be multiple of 5

msTyp = np.r_[41, 42, 43] # reflectance, Q, U; currently assumes all msTyp's have same nbvm
graspObjs = []
nip = msTyp.shape[0]
gObj = graspRun(pathYAML, orbHghtKM, dirGRASPworking)
dtNm = np.float(dayDtNm)
lon = 0
lat = 0
#masl = 1662 # estimated from ROT vs ROT max in orginal files
masl = 1662
nowPix = pixel(dtNm, 1, 1, lon, lat, masl, lndPrct)

for wl in [0.865]: # LOOP OVER WAVELENGTHS HACK!
    warnings.simplefilter('ignore') # ignore missing_value not cast warning
    fileNow = cstFile % (1000*np.float(wl))
    netCDFobj2 = Dataset(fileNow)
    I = np.array(netCDFobj2.variables['I'])*np.pi
    Q = np.array(netCDFobj2.variables['Q'])*np.pi
    U = np.array(netCDFobj2.variables['U'])*np.pi
    ROT = np.sum(np.array(netCDFobj2.variables['ROT']))
    DoLP = np.sqrt(Q**2+U**2)/I
    phiSens = np.array(netCDFobj2.variables['sensor_azimuth'])
    theta = np.array(netCDFobj2.variables['sensor_zenith'])
    netCDFobj2.close()
    warnings.simplefilter('always')

    fwdInd = (phiSens == trgtAzmth).nonzero()[0][0]
    bckInd = (phiSens == trgtAzmth+180).nonzero()[0][0]
    phi =  360 - np.r_[np.tile(trgtAzmth, theta.shape[0]), np.tile(trgtAzmth+180, theta.shape[0])]
    nbvm = phi.shape[0] 
    phi = np.tile(phi, nip) # ex. 11, 35, 55, 11, 35, 55...
    thtv = np.abs(np.tile(np.tile(theta,2), nip))
    msrmnts = np.r_[I[:,fwdInd], I[:,bckInd], Q[:,fwdInd], Q[:,bckInd], U[:,fwdInd], U[:,bckInd]]
    nowPix.addMeas(wl, msTyp, np.repeat(nbvm, nip), sza, thtv, phi, msrmnts)
gObj.addPix(nowPix)
graspObjs.append(gObj)

# Write SDATA, run GRASP and read in results
graspObjs[0].writeSDATA()
#graspObjs[0].runGRASP()
#rslts = graspObjs[0].readOutput()
