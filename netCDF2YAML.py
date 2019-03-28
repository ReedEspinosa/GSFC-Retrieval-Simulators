#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import matplotlib.pyplot as plt
from runGRASP import graspDB
from MADCAP_functions import readVILDORTnetCDF
import numpy as np

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/benchmark_rayleigh_BRDF_BPDF_PP/calipso-g5nr.vlidort.vector.MODIS_BRDF_BPDF.%dd00.nc4')
#radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sulfateBenchmark/calipso-g5nr.vlidort.vector.MODIS_BRDF.%dd00.nc4')
yamlFile = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sulfateBenchmark/settings_HARP_16bin_6lambda.yml')
rsltsFile = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sulfateBenchmark/sulfate_bench_fit__PSDlt600nm_BRDF_FREE.pkl')

#varNames = ['I', 'Q', 'U', 'sensor_zenith', 'RGEO', 'RISO', 'RVOL', 'TAU', 'VOL', 'radius', 'TOTdist', 'REFR', 'REFI', 'ZE']
varNames = ['I', 'Q', 'U', 'sensor_zenith', 'RGEO', 'RISO', 'RVOL']
#wvls = [0.410, 0.440, 0.550, 0.670, 1.020, 2.100] # wavelengths to read from levC files
wvls = [0.865] # wavelengths to read from levC files

# Read in radiances, solar spectral irradiance and find reflectances
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls)

#gDB = graspDB()
#gDB.loadResults(rsltsFile)

# Read in model "truth" from levC lidar file
#varNamesLev = varNames[10:]
#Npix = measData[0]['I'].shape[0]
#Nwvlth = len(wvls)
#for i in range(Nwvlth): 
#    tauKrnl = measData[i]['TAU']
#    tauTot = np.sum(measData[i]['TAU'])
#    tauKrnl = tauKrnl/tauTot
#    measData[i]['ZE'] = (measData[i]['ZE'][0:-1]+measData[i]['ZE'][1:])/2
#    for varName in varNamesLev:
#        if measData[i][varName].ndim==2:
#            Nflds = measData[i][varName].shape[1]
#            measData[i][varName] = np.sum(np.tile(tauKrnl,(Nflds,1)).T*measData[i][varName], axis=0)
#        else:
#            measData[i][varName] = np.sum(tauKrnl*measData[i][varName])
#
#print('RRI:')
#print([md['REFR'] for md in measData]);
#print('IRI:')
#print([md['REFI'] for md in measData]);
#print('hieght:')
#print([md['ZE'] for md in measData]);
print('RTLS BRDF (ISO,VOL,GEOM):')
riso = np.array([md['RISO'] for md in measData])[:,0]
print(riso.tolist());
print((np.array([md['RVOL'] for md in measData])[:,0]/riso).tolist());
print((np.array([md['RGEO'] for md in measData])[:,0]/riso).tolist());

#plt.figure()
#l=2; t=0
#plt.plot(measData[l]['radius'], measData[l]['TOTdist']/measData[l]['TOTdist'].max())
#plt.plot(gDB.rslts[t]['r'], gDB.rslts[t]['dVdlnr']/gDB.rslts[t]['dVdlnr'].max())
#plt.xscale('log')
