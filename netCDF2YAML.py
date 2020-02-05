#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will read in data from a NetCDF file, print out the results and, if provided, write the result to a YAML file """

import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import matplotlib.pyplot as plt
from runGRASP import graspDB, graspYAML
from MADCAP_functions import loadVARSnetCDF, readVILDORTnetCDF
import numpy as np
from scipy import interpolate as intrp
import miscFunctions as mf

# Paths to files
baseYAML = None
#baseYAML = '/Users/wrespino/Synced/Working/GRASP_PMgenerationRun/settings_optics_SU.v1_5_gpmom_sizedist.yml'
#tauFctr = 1.0041052996873248 # netCDF vol will be scaled by this before being set in YAML, set to unity to skip this hack
tauFctr = 1.0 # netCDF vol will be scaled by this before being set in YAML, set to unity to skip this hack
#intrpRadii = np.array([0.05, 0.059009, 0.06964, 0.082188, 0.096996, 0.11447, 0.1351, 0.15944, 0.18816, 0.22206, 0.26207, 0.30929, 0.36502, 0.43078, 0.5084, 0.6])
lnNC2binYAML = True # take lorgnormal values from netCDF and convert them to a binned YAML PSD


## ---Single Scattering Optics Tables---
#rhInd = 0 #14->70%
#wvInds = [1,2,4,6,8,11,13,15,17] #1->300nm, 2->350nm, 4->450nm, 6->550nm, 8->650nm, 11->800nm, 13->1000nm, 15->1500nm, 17->2000nm, 18->2500nm
#varNames = ['r_dist', 'size_dist', 'refreal', 'refimag', 'lambda', 'sigma', 'rMode']
#radianceFNfrmtStr = '/Users/wrespino/Synced/Working/GRASP_PMgenerationRun/optics_SU.v5_7.GSFun.nc'
#rawData = loadVARSnetCDF(radianceFNfrmtStr, varNames)
#measData = np.array([dict() for _ in wvInds])
#wvls = np.zeros(len(wvInds))
#for i,wvInd in enumerate(wvInds):
#    measData[i]['REFR_all'] = np.array([rawData['refreal'][0,rhInd,wvInd]]).squeeze()
#    measData[i]['REFI_all'] = -np.array([rawData['refimag'][0,rhInd,wvInd]]).squeeze()
#    measData[i]['SSA_all'] = np.array([np.nan])
#    measData[i]['TAU_all'] = np.array([np.nan])
#    wvls[i] = rawData['lambda'][wvInd]*1e6
#if 'r_dist' in rawData: # files with full logdistribution
#    measData[0]['radius'] = np.array([rawData['r_dist'][:,0,rhInd]]).squeeze()
#    measData[0]['TOT_COL_dvdlnr'] = np.array([rawData['size_dist'][:,0,rhInd]]).squeeze()*(measData[0]['radius']**4)
#else: # files with just lognormal
#    measData[0]['rMode'] = rawData['rMode'][0,rhInd]
#    measData[0]['sigma'] = rawData['sigma'][0]    

# ---OSSE VILDORT Full RT Outputs---
rmtPrjctPath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/VLIDORTbench_graspConfig_12/benchmark_rayleigh+simple_aerosol_BRDF'
radianceFNfrmtStr = os.path.join(rmtPrjctPath, 'calipso-g5nr.vlidort.vector.MODIS_BRDF.%dd00.nc4')
varNames = ['I', 'Q', 'U', 'sensor_zenith', 'RGEO', 'RISO', 'RVOL', 'TAU', 'VOL', 'radius', 'TOTdist', 'colTOTdist', 'REFR', 'REFI', 'SSA', 'ZE', 'U10m', 'V10m', 'ROT', 'SUdist', 'AREA', 'REFF']
extensiveVars = ['TAU', 'TOTdist', 'ROT', 'SUdist', 'AREA']
#wvls = [0.410, 0.440, 0.550, 0.670, 1.020, 2.100] # wavelengths to read from levC files
wvls = [0.400] # wavelengths to read from levC files
measData = readVILDORTnetCDF(varNames, radianceFNfrmtStr, wvls)
Nwvlth = len(wvls)
Nlayer = measData[0]['ZE'].shape[0]-1
for i in range(Nwvlth):  # integrate over all layers
    measData[i]['ZE_edge'] = measData[i]['ZE']
    measData[i]['ZE'] = (measData[i]['ZE'][0:-1]+measData[i]['ZE'][1:])/2 # ZE now midpoint, ZE_all is AOD weighted average height 
    if 'TAU' in measData[i]: # Else there is no aerosol
        if 'TOTdist' in measData[i]:
            measData[i]['TOT_COL_dvdlnr'] = np.sum(measData[i]['TOTdist'].T*np.diff(-measData[i]['ZE_edge']), axis=1) # dv/dlnr (um^3/um^2)
        tauKrnl = measData[i]['TAU']/np.sum(measData[i]['TAU']) # weight intensive parameters by layer optical depth
        for varName in list(measData[i]):           
            if varName in extensiveVars:
                measData[i][varName+'_all'] = np.sum(measData[i][varName], axis=0)
            elif measData[i][varName].ndim==2 and measData[i][varName].shape[0]==Nlayer and 'TAU' in measData[i]:
                Nflds = measData[i][varName].shape[1]
                measData[i][varName+'_all'] = np.sum(np.tile(tauKrnl,(Nflds,1)).T*measData[i][varName], axis=0)
            elif not np.isscalar(measData[i][varName]) and measData[i][varName].shape[0]==Nlayer and 'TAU' in measData[i]:
                measData[i][varName+'_all'] = np.sum(tauKrnl*measData[i][varName])

print('<> Wavelengths (LAMBDA):')
print(wvls)
if baseYAML:
    apndStr = '_%s_RHind%d' % (os.path.basename(radianceFNfrmtStr),rhInd)
    if lnNC2binYAML: apndStr = apndStr + "_BINNED"
    newYamlPath = radianceFNfrmtStr[:-3]+apndStr+'.yml'
    newYamlPath = newYamlPath.replace('%d','%dWvls' % len(wvls))
    gy = graspYAML(baseYAML, newYamlPath)
    gy.adjustLambda(len(wvls))
    gy.access('stop_before_performing_retrieval', True)
    gy.access('output.segment.stream', 'bench_inversionRslts'+apndStr+'.txt')

#
if 'RISO' in measData[0]:
    print('<> RTLS BRDF (ISO;VOL;GEOM):')
    riso = np.array([md['RISO'] for md in measData])[:,0]
    print(riso.tolist());
    rvol = np.array([md['RVOL'] for md in measData])[:,0]/riso
    print(rvol.tolist());
    rgeo = np.array([md['RGEO'] for md in measData])[:,0]/riso
    print(rgeo.tolist());
    if baseYAML:
        gy.access('surface_land_brdf_ross_li.1', newVal=riso)
        gy.access('surface_land_brdf_ross_li.2', newVal=rvol)
        gy.access('surface_land_brdf_ross_li.3', newVal=rgeo)
#
#if 'U10m' in measData[0]:
#    print('<> Ocean Parameters (U10m;V10m):')
#    print(np.array([md['U10m'] for md in measData]).tolist());
#    print((np.array([md['V10m'] for md in measData])).tolist());
#    
if 'SSA_all' in measData[0]:
    print('<> Spectral Aerosol Parameters (SSA;TAU):')
    print(np.array([md['SSA_all'] for md in measData]).tolist());
    print(np.array([md['TAU_all'] for md in measData]).tolist());
    if baseYAML: print('NOTE: THIS SCRIPT DOES NOT ADJUST VOL/TAU')

if 'REFR_all' in measData[0]:
    print('<> Spectral Aerosol Parameters (RRI;IRI):')
    REFR_all = np.array([md['REFR_all'] for md in measData])
    print(REFR_all.tolist());
    REFI_all = np.array([md['REFI_all'] for md in measData])
    print(REFI_all.tolist());
    if baseYAML:
        gy.access('real_part_of_refractive_index_spectral_dependent', newVal=REFR_all)
        gy.access('imaginary_part_of_refractive_index_spectral_dependent', newVal=REFI_all)

if 'rMode' in measData[0]:
    print('<> lognormal psd (rv;ln(σ)):')
    sigma = np.log(measData[0]['sigma'])
    rv = 1e6*measData[0]['rMode']*np.exp(3*sigma**2)
    print('rv=%5.3f μm; ln(σ)=%5.3f' % (rv,sigma))
    if baseYAML and not lnNC2binYAML: gy.access('size_distribution_lognormal.1.value', [rv, sigma])
        
Nbins = 40
minR = 0.03
maxR = 7.0
intrpRadii = np.logspace(np.log10(minR),np.log10(maxR),Nbins) # this will match GRASP to 8 digits
Nradii = len(intrpRadii)
if lnNC2binYAML: 
    print('WARNING: PSD not contructed from rv and σ because rMode was not found in NetCDF.')
    lnNC2binYAML = False
if 'radius' in measData[0] or lnNC2binYAML:
    print('<> Size Distribution (Radii;dvdlnr;min;max;wvlngInvlv):')
    print('radiusMin = %6.4f, radiusMax = %6.4f' % (minR,maxR) )
    print(intrpRadii.tolist())
    if lnNC2binYAML and 'rMode' in measData[0]:
        psd = mf.logNormal(rv*np.exp(sigma**2), sigma, r=intrpRadii)[0]
    else:
        dvdlnr = intrp.interp1d(measData[0]['radius'],measData[0]['TOT_COL_dvdlnr'], bounds_error=False, fill_value=0) #dv/dlnr (um^3/um^2)
        psd = np.maximum(tauFctr*dvdlnr(intrpRadii),1.1e-13)
        plt.figure()
        plt.plot(intrpRadii,3*dvdlnr(intrpRadii)/intrpRadii, '-o')
        plt.plot(measData[0]['radius'],3*measData[0]['TOT_COL_dvdlnr']/measData[0]['radius'])
        plt.plot([minR, minR], [0,(3*dvdlnr(intrpRadii)/intrpRadii).max()], 'k--')
        plt.plot([maxR, maxR], [0,(3*dvdlnr(intrpRadii)/intrpRadii).max()], 'k--')
        plt.xscale('log')
        plt.xlabel('radius (μm)')
        plt.ylabel('dS/dlnr (μm^2/μm^2)')
        plt.legend(['interpolated\n(GRASP)', 'netCDF'])        
    print(psd.tolist())
    psdMin = 1.0e-13*np.ones(Nradii)
    print(psdMin.tolist())
    psdMax = 5.0*np.ones(Nradii)
    print(psdMax.tolist())
    psdWv = np.zeros(Nradii,dtype=int)
    print(psdWv.tolist())
    if baseYAML:
        gy.access('size_distribution_triangle_bins.1.value', psd)
        gy.access('size_distribution_triangle_bins.1.min', psdMin)
        gy.access('size_distribution_triangle_bins.1.max', psdMax)
        gy.access('size_distribution_triangle_bins.1.index_of_wavelength_involved', psdWv)
        gy.access('retrieval.phase_matrix.radius.mode[1].min', newVal=minR)
        gy.access('retrieval.phase_matrix.radius.mode[1].max', newVal=maxR)       
    

if 'ROT_all' in measData[0]:
    print('<> Rayleigh optical depth (ROT):')
    print((np.array([md['ROT_all'] for md in measData])).tolist());
    if np.isclose(measData[0]['ROT_all'], 0) and baseYAML:
        gy.access('retrieval.radiative_transfer.molecular_profile_vertical_type', 'no_rayleigh')
    elif baseYAML:
        gy.access('retrieval.radiative_transfer.molecular_profile_vertical_type', 'exponential')
        print('NOTE: THIS SCRIPT CAN NOT SET ROT MAGNITUDE (use the alt. value within SDATA file)')

if 'ZE_all' in measData[0]:
    print('<> AOD weighted mean height [m]:')
    vrtHght = np.array([md['ZE_all'] for md in measData]) 
    print(vrtHght .tolist());
    if baseYAML:
        gy.access('vertical_profile_parameter_height.1', vrtHght)


#plt.figure()
#plt.plot(measData[0]['radius'],measData[0]['TOT_COL_dvdlnr'],gDB.rslts[0]['r'],gDB.rslts[0]['vol']*gDB.rslts[0]['dVdlnr'],'x')
#plt.xlim(0.04,0.6)
#plt.xscale('log')
#plt.xlabel('radius (μm)')
#plt.ylabel('dv/dlnr ($μm^3/μm^3$)')
#plt.legend(['netCDF', 'GRASP'])

#totPSD = np.zeros(measData[0]['TOTdist'].shape[1])
#for h in range(measData[0]['VOL'].shape[0]):
#    TOTdistIntegral = np.trapz(measData[0]['TOTdist'][h,:], x=np.log(measData[0]['radius']))
#    nrmFactor = measData[0]['VOL'][h]/TOTdistIntegral*np.diff(-measData[0]['ZE_edge'])[h]*1e6
#    totPSD = totPSD + nrmFactor*measData[0]['TOTdist'][h,:]
#plt.plot(measData[0]['radius'],totPSD)


#plt.plot(measData[0]['radius'],gDB.rslts[0]['dVdlnr'].max()/measData[0]['TOTdist'].max()*measData[0]['TOTdist'].T)

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

