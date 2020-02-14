#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import MADCAP_functions as mf

aeroFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/AERONET_stats/puredata-combi-Level1_5-alm-SU-0.70pureness-clean2.nc4'
"""
'Refractive_Index_Real_440', 'Refractive_Index_Real_675', 'Refractive_Index_Real_870', 'Refractive_Index_Real_1020', 
'Refractive_Index_Imag_440', 'Refractive_Index_Imag_675', 'Refractive_Index_Imag_870', 'Refractive_Index_Imag_1020', 
'VMR_T', 'Std_T', 'VolC_T', 'VMR_C', 'Std_C', 'VolC_C', 'VMR_F', 'Std_F', 'VolC_F', 
'Sphericity_Factor', 'Extinction_Angstrom_Exponent_440_870_Total', 
'Single_Scattering_Albedo_440', 'Single_Scattering_Albedo_675', 'Single_Scattering_Albedo_870', 'Single_Scattering_Albedo_1020', 
'AERONET_Site_id', 'pureness', 'hour', 'year', 'month', 'day'
"""
ttlTxt = 'AERONET Level 1.5 Retrievals - Sulfate'
figHnd, axHnd = plt.subplots(1,2,figsize=(12,6))

axInd = 0
varX = 'Refractive_Index_Real_440'
varY = 'VMR_T'  
aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[0], res=50, \
                      xrng=None, yrng=[0.05, 1.5], sclPow=0.5, \
                      cmap='YlOrRd', clbl='')
axHnd[0].set_xlabel(r'$RRI_{440}$')
axHnd[0].set_ylabel(r'$r_v\ (μm)$')

axInd = 1
varX = 'Refractive_Index_Real_440'
varY = 'Refractive_Index_Imag_440'     
aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
                      xrng=None, yrng=[0.0005, 0.01], sclPow=0.5, \
                      cmap='YlOrRd')
axHnd[axInd].set_xlabel(r'$RRI_{440}$')
axHnd[axInd].set_ylabel(r'$IRI_{440}$')

figHnd.suptitle(ttlTxt)
figHnd.tight_layout(rect=[0, 0.03, 1, 0.95])

