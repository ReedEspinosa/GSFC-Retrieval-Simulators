#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import matplotlib.pyplot as plt
from runGRASP import graspDB
from MADCAP_functions import readVILDORTnetCDF
import numpy as np
from scipy import interpolate as intrp

# Paths to files
basePath = '/Users/wrespino/Synced/' # NASA MacBook
radianceFNfrmtStr = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/VLIDORTbench_graspConfig_12/benchmark_rayleigh+aerosol_BRDF/calipso-g5nr.vlidort.vector.MODIS_BRDF.%dd00.nc4')

rsltsFile = os.path.join(basePath, 'Remote_Sensing_Projects/MADCAP_CAPER/sulfateBenchmark/sulfate_bench_fit__PSDlt600nm_BRDF_FREE.pkl')

varNames = ['I', 'Q', 'U', 'sensor_zenith', 'RGEO', 'RISO', 'RVOL', 'TAU', 'VOL', 'radius', 'TOTdist', 'REFR', 'REFI', 'SSA', 'ZE', 'U10m', 'V10m', 'ROT', 'SUdist', 'AREA', 'REFF']
#wvls = [0.410, 0.440, 0.550, 0.670, 1.020, 2.100] # wavelengths to read from levC files
wvls = [0.470, 0.865] # wavelengths to read from levC files

extensiveVars = ['TAU', 'VOL', 'TOTdist', 'ROT', 'SUdist', 'AREA']

intrpRadii = np.array([0.05, 0.059009, 0.06964, 0.082188, 0.096996, 0.11447, 0.1351, 0.15944, 0.18816, 0.22206, 0.26207, 0.30929, 0.36502, 0.43078, 0.5084, 0.6])



# Read in radiances, solar spectral irradiance and find reflectances
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls)


# Read in model "truth" from levC lidar file
Nwvlth = len(wvls)
Nlayer = measData[0]['ZE'].shape[0]-1
volInt = np.sum(measData[0]['VOL']*np.diff(-measData[0]['ZE'])) # column concentration (m^3/m^2)
for i in range(Nwvlth):  # integrate over all layers
    tauKrnl = measData[i]['TAU']/np.sum(measData[i]['TAU']) # weight intensive parameters by layer optical depth
    measData[i]['ZE_edge'] = measData[i]['ZE']
    measData[i]['ZE'] = (measData[i]['ZE'][0:-1]+measData[i]['ZE'][1:])/2 # ZE now midpoint, ZE_all is AOD weighted average height 
    for varName in list(measData[i]):           
        if varName in extensiveVars:
            measData[i][varName+'_all'] = np.sum(measData[i][varName], axis=0)
        elif measData[i][varName].ndim==2 and measData[i][varName].shape[0]==Nlayer:
            Nflds = measData[i][varName].shape[1]
            measData[i][varName+'_all'] = np.sum(np.tile(tauKrnl,(Nflds,1)).T*measData[i][varName], axis=0)
        elif not np.isscalar(measData[i][varName]) and measData[i][varName].shape[0]==Nlayer:
            measData[i][varName+'_all'] = np.sum(tauKrnl*measData[i][varName])
    TOTdistIntegral = np.trapz(measData[i]['TOTdist_all'], x=np.log(measData[i]['radius'])) 
    measData[i]['TOT_COL_dvdlnr'] = volInt/TOTdistIntegral*measData[i]['TOTdist_all'] # dv/dlnr (m^3/m^2)


#print('<> Wavelengths (LAMBDA):')
#print(wvls)
#
#if 'RISO' in measData[0]:
#    print('<> RTLS BRDF (ISO;VOL;GEOM):')
#    riso = np.array([md['RISO'] for md in measData])[:,0]
#    print(riso.tolist());
#    print((np.array([md['RVOL'] for md in measData])[:,0]/riso).tolist());
#    print((np.array([md['RGEO'] for md in measData])[:,0]/riso).tolist());
#
#if 'U10m' in measData[0]:
#    print('<> Ocean Parameters (U10m;V10m):')
#    print(np.array([md['U10m'] for md in measData]).tolist());
#    print((np.array([md['V10m'] for md in measData])).tolist());
#    
#if 'SSA' in measData[0]:
#    print('<> Spectral Aerosol Parameters (RRI;IRI;SSA;TAU):')
#    print(np.array([md['REFR_all'] for md in measData]).tolist());
#    print(np.array([md['REFI_all'] for md in measData]).tolist());
#    print(np.array([md['SSA_all'] for md in measData]).tolist());
#    print(np.array([md['TAU_all'] for md in measData]).tolist());
#

intrpRadii = np.logspace(np.log10(0.05),np.log10(0.5),40) # this will match GRASP to 8 digits
Nradii = len(intrpRadii)
if 'radius' in measData[0]:
    dvdlnr = intrp.interp1d(measData[0]['radius'],measData[0]['TOT_COL_dvdlnr']*1e6) #dv/dlnr (um^3/um^2)
    print('<> Size Distribution (Radii;dvdlnr;min;max;wvlngInvlv):')
    print(intrpRadii.tolist())
    print(np.maximum(dvdlnr(intrpRadii),1.1e-13).tolist())
    print((1.0e-13*np.ones(Nradii)).tolist())
    print((5.0*np.ones(Nradii)).tolist())
    print((np.zeros(Nradii,dtype=int)).tolist())
#
#print('<> Rayleigh optical depth (ROT):')
#print((np.array([md['ROT_all'] for md in measData])).tolist());
#
#print('<> AOD weighted mean height [m]:')
#print((np.array([md['ZE_all'] for md in measData])).tolist());


# [outdated] sanity checks
#ind = 63            
#r = measData[0]['radius']
#volConc = measData[0]['VOL'][ind] # total aersol volume per m^3 of air volume?
#volConcCol = np.trapz(measData[0]['TOTdist'][ind], x=np.log(r)) # integration of absolute size distribution in layer per um^2 footprint?
#Reff = measData[0]['REFF'][ind] # effective radius?
#layWdth = (measData[0]['ZE_edge'][ind]-measData[i]['ZE_edge'][ind+1])*1e6 # layer thickness in um
#volConcFrmCol = volConcCol/layWdth # total volume concentration from integrated PSD per volume
#areaFrmCol = np.trapz(measData[0]['TOTdist'][ind]*(3/4)/(measData[0]['radius']**2),x=r)/layWdth # total cross sectional area from integrated PSD per volume
#area = measData[0]['AREA'][ind] # total cross sectional area per volume
#ReffConcInt = (3/4)*volConcFrmCol/areaFrmCol # effective radius from integrated PSD
#
#print('%15s=%E um' % ('layWdth',layWdth))
#print('%15s=%E um^3/um^2' % ('volConcCol',volConcCol))
#print('%15s=%E um' % ('Reff',Reff))
#print('%15s=%E um' % ('ReffConcInt',ReffConcInt))
#print('%15s=%E um^3/um^3' % ('volConcFrmCol',volConcFrmCol))
#print('%15s=%E m^3/m^3' % ('volConc',volConc))
#print('%15s=%E m^3/m^3' % ('area',area))
#print('%15s=%E um^2/um^3 (%E m^2/m^3)' % ('areaFrmCol',areaFrmCol,areaFrmCol/1e6))


#gDB = graspDB()
#gDB.loadResults(rsltsFile)

#plt.figure()
#l=2; t=0
#plt.plot(measData[l]['radius'], measData[l]['TOTdist']/measData[l]['TOTdist'].max())
#plt.plot(gDB.rslts[t]['r'], gDB.rslts[t]['dVdlnr']/gDB.rslts[t]['dVdlnr'].max())
#plt.xscale('log')
