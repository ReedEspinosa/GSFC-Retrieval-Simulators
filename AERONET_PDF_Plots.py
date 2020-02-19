#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import matplotlib
import MADCAP_functions as mf

aeroFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/AERONET_stats/puredata-combi-Level1_5-alm-DU-0.70pureness-clean2.nc4'
"""
'Refractive_Index_Real_440', 'Refractive_Index_Real_675', 'Refractive_Index_Real_870', 'Refractive_Index_Real_1020', 
'Refractive_Index_Imag_440', 'Refractive_Index_Imag_675', 'Refractive_Index_Imag_870', 'Refractive_Index_Imag_1020', 
'VMR_T', 'Std_T', 'VolC_T', 'VMR_C', 'Std_C', 'VolC_C', 'VMR_F', 'Std_F', 'VolC_F', 
'Sphericity_Factor', 'Extinction_Angstrom_Exponent_440_870_Total', 
'Single_Scattering_Albedo_440', 'Single_Scattering_Albedo_675', 'Single_Scattering_Albedo_870', 'Single_Scattering_Albedo_1020', 
'AERONET_Site_id', 'pureness', 'hour', 'year', 'month', 'day'
"""
figHnd, axHnd = plt.subplots(1,3,figsize=(12,3.4))

font = {'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

# --- things only in lev1.5---
axInd = 0
varX = 'Sphericity_Factor'
varY = 'VMR_T'  
aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
                      yrng=[0.12, 2.7], xrng=[0.1, 100.0], sclPow=0.5, \
                      cmap='YlOrRd', clbl='')
axHnd[axInd].set_xlabel(r'$spherical\ fraction$')
axHnd[axInd].set_ylabel(r'$r_{vt}\ (μm)$')
ttlTxt = 'AERONET Level 1.5 Retrievals - Dust (N_DU=%d)' % len(aero[varX])

axInd = 1
varX = 'Refractive_Index_Real_675'
varY = 'VMR_T'  
aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
                      yrng=[0.12, 2.7], xrng=[1.33, 1.56], sclPow=0.5, \
                      cmap='YlOrRd', clbl='')
axHnd[axInd].set_xlabel(r'$RRI_{675}$')
axHnd[axInd].set_ylabel(r'$r_{vt}\ (μm)$')

axInd = 2
varX = 'Single_Scattering_Albedo_675'
varY = 'VMR_T'  
aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
                      yrng=[0.12, 2.7], xrng=[0.85, 1.0], sclPow=0.5, \
                      cmap='YlOrRd', clbl='')
axHnd[axInd].set_xlabel(r'$SSA_{675}$')
axHnd[axInd].set_ylabel(r'$r_{vt}\ (μm)$')

figHnd.suptitle(ttlTxt)
figHnd.tight_layout(rect=[0, 0.03, 1, 0.95])


#--- size vs Vol ---
#axInd = 0
#varX = 'VolC_T'
#varY = 'VMR_T'  
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.12, 2.1], xrng=[0.001, 0.7], sclPow=0.5, \
#                      cmap='YlOrRd', clbl='')
#axHnd[axInd].set_xlabel(r'$vol \ (μm^3/μm^2)$')
#axHnd[axInd].set_ylabel(r'$r_{vt}\ (μm)$')
#ttlTxt = 'AERONET Level 2.0 Retrievals - Sulfate (N_SU=%d)' % len(aero[varX])
#
#axInd = 1
#varX = 'VolC_F'
#varY = 'VMR_F'  
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.09, 0.41], xrng=[0.001, 0.3], sclPow=0.5, \
#                      cmap='YlOrRd', clbl='')
#axHnd[axInd].set_xlabel(r'$vol \ (μm^3/μm^2)$')
#axHnd[axInd].set_ylabel(r'$r_{vf}\ (μm)$')
#
#axInd = 2
#varX = 'VolC_C'
#varY = 'VMR_C'     
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.9, 4.4], xrng=[0.001, 0.7], sclPow=0.5, \
#                      cmap='YlOrRd')
#axHnd[axInd].set_xlabel(r'$vol \ (μm^3/μm^2)$')
#axHnd[axInd].set_ylabel(r'$r_{vc}\ (μm)$')
#
#figHnd.suptitle(ttlTxt)
#figHnd.tight_layout(rect=[0, 0.03, 1, 0.95])


# ---size VS SSA ---
#axInd = 0
#varX = 'Single_Scattering_Albedo_675'
#varY = 'VMR_T'  
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.12, 0.96], xrng=[0.898, 1.0], sclPow=0.5, \
#                      cmap='YlOrRd', clbl='')
#axHnd[axInd].set_xlabel(r'$SSA_{675}$')
#axHnd[axInd].set_ylabel(r'$r_{vt}\ (μm)$')
#ttlTxt = 'AERONET Level 2.0 Hyb. Retrievals - Sulfate (N_SU=%d)' % len(aero[varX])
#
#axInd = 1
#varX = 'Single_Scattering_Albedo_675'
#varY = 'VMR_F'  
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.09, 0.41], xrng=[0.898, 1.0], sclPow=0.5, \
#                      cmap='YlOrRd', clbl='')
#axHnd[axInd].set_xlabel(r'$SSA_{675}$')
#axHnd[axInd].set_ylabel(r'$r_{vf}\ (μm)$')
#
#axInd = 2
#varX = 'Single_Scattering_Albedo_675'
#varY = 'VMR_C'     
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[2.2, 4.4], xrng=[0.898, 1.0], sclPow=0.5, \
#                      cmap='YlOrRd')
#axHnd[axInd].set_xlabel(r'$SSA_{675}$')
#axHnd[axInd].set_ylabel(r'$r_{vc}\ (μm)$')
#
#figHnd.suptitle(ttlTxt)
#figHnd.tight_layout(rect=[0, 0.03, 1, 0.95])



# --- SIZE VS RRI ---
#axInd = 0
#varX = 'Refractive_Index_Real_675'
#varY = 'VMR_T'  
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.12, 0.96], xrng=[1.33, 1.53], sclPow=0.5, \
#                      cmap='YlOrRd', clbl='')
#axHnd[axInd].set_xlabel(r'$RRI_{675}$')
#axHnd[axInd].set_ylabel(r'$r_{vt}\ (μm)$')
#ttlTxt = 'AERONET Level 2.0 Retrievals - Sulfate (N_SU=%d)' % len(aero[varX])
#
#axInd = 1
#varX = 'Refractive_Index_Real_675'
#varY = 'VMR_F'  
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[0.09, 0.41], xrng=[1.33, 1.53], sclPow=0.5, \
#                      cmap='YlOrRd', clbl='')
#axHnd[axInd].set_xlabel(r'$RRI_{675}$')
#axHnd[axInd].set_ylabel(r'$r_{vf}\ (μm)$')
#
#axInd = 2
#varX = 'Refractive_Index_Real_675'
#varY = 'VMR_C'     
#aero = mf.loadVARSnetCDF(aeroFile, [varX, varY])
#objHnd = mf.KDEhist2D(aero[varX], aero[varY], axHnd=axHnd[axInd], res=50, \
#                      yrng=[2.2, 4.4], xrng=[1.33, 1.53], sclPow=0.5, \
#                      cmap='YlOrRd')
#axHnd[axInd].set_xlabel(r'$RRI_{675}$')
#axHnd[axInd].set_ylabel(r'$r_{vc}\ (μm)$')
#
#figHnd.suptitle(ttlTxt)
#figHnd.tight_layout(rect=[0, 0.03, 1, 0.95])
#
