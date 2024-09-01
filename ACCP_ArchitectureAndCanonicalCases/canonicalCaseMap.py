#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rnd
# to use the same seed for random number generator
# rnd.default_rng(seed=33)
import tempfile
import os
import sys
import pickle
import re
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
from miscFunctions import logNormal, loguniform, slope4RRI
import runGRASP as rg

def conCaseDefinitions(caseStr, nowPix, defineRandom = None):
    """ '+' is used to seperate multiple cases (implemented in splitMultipleCases below)
        This function should insensitive to trailing characters in caseStr
            (e.g. 'smokeDesert' and 'smokeDeserta2' should produce same result)
    """
    vals = dict()
    wvls = np.unique([mv['wl'] for mv in nowPix.measVals])
    nwl = len(wvls)
    
    # variable type cases
    # variable type appended options: 'fine'/'coarse', 'nonsph' and 'lofted' """
    if 'variable' in caseStr.lower(): # dimensions are [mode, λ or (rv,sigma)];
        σ = 0.35+rnd.random()*0.3
        if 'fine' in caseStr.lower():
            rv = 0.145+rnd.random()*0.105
            vals['vol'] = np.array([[0.5+rnd.random()*0.5]])/3 # (currently gives AOD≈1 but changes w/ intensive props.)
        elif 'coarse' in caseStr.lower():
            rv = 0.8+rnd.random()*3.2
            vals['vol'] = np.array([[1.5+rnd.random()*1.5]])/3 # (currently gives AOD≈1 but changes w/ intensive props.)
        else:
            assert False, 'variable aerosol case must be appended with either fine or coarse'
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.0001]] if 'nonsph' in caseStr.lower() else [[0.99999]] # mode 1, 2,...
        vals['vrtHght'] = [[3010]] if 'lofted' in caseStr.lower() else  [[1010]]  # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500]] # Gaussian sigma in meters
        vals['n'] = np.interp(wvls, [wvls[0],wvls[-1]],   1.36+rnd.random(2)*0.15)[None,:] # mode 1 # linear w/ λ
        vals['k'] = np.interp(wvls, [wvls[0],wvls[-1]], 0.0001+rnd.random(2)*0.015)[None,:] # mode 1 # linear w/ λ
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    
    # cloud-like cases for Dan's RST proposal
    elif 'lwcloud' in caseStr.lower():
        σ = [0.5] # mode 1,...
        rv = [20.0]
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999]] # mode 1
        vals['vol'] = np.array([[200.0]])
        vals['vrtHght'] = [[2010]] # mode 1,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500]] # mode 1,,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.33, nwl)[None,:] # mode 1
        vals['k'] = np.repeat(1e-8, nwl)[None,:] # mode 1
        landPrct = 0 # ocean
        
    # Yingxi's smoke model cases
    elif 'huambo' in caseStr.lower():
        vals = yingxiProposalSmokeModels('Huambo', wvls)
        landPrct = 0 if 'ocean' in caseStr.lower() else 100
    elif 'nasaames' in caseStr.lower():
        vals = yingxiProposalSmokeModels('NASA_Ames', wvls)
        landPrct = 0 if 'ocean' in caseStr.lower() else 100
    
    # New AOS cases
    elif 'clean' in caseStr.lower():
        σ = [0.4, 0.68] # mode 1, 2,...
        rv = [0.1, 0.84]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.00000001], [0.00000001]])
        vals['vrtHght'] = [[3010],  [3010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.39, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.51, nwl)]) # mode 2
        vals['k'] = np.repeat(1e-8, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.002, nwl)]) # mode 2 # THIS HAS A SPECTRAL DEPENDENCE IN THE SPREADSHEET
        landPrct = 0 if 'ocean' in caseStr.lower() else 100
    elif 'smoke' in caseStr.lower(): # ALL VARIABLES WITH MODES MUST BE 2D (ie. var[mode,wl]) or [] (will not change these values)
        σ = [0.4, 0.45] # mode 1, 2,...
        rv = [0.12, 0.36]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.1094946], [0.03520468]]) # gives AOD=4*[0.2165, 0.033499]=1.0
        vals['vrtHght'] = [[3010],  [3010]] # mode 1, 2,... # Gaussian mean in meters #HACK: should be 3k
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.54, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.47, nwl)]) # mode 2
        vals['k'] = np.repeat(0.01, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.0001, nwl)]) # mode 2
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    elif 'marine' in caseStr.lower():
        σ = [0.45, 0.70] # mode 1, 2,...
        rv = [0.18, 0.6]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
#         rv = [0.12, 0.6]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.0477583], [0.7941207]]) # gives AOD=10*[0.0287, 0.0713]=1.0 total
        vals['vrtHght'] = [[1010],  [1010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.415, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.363, nwl)]) # mode 2
        vals['k'] = np.repeat(0.002, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(1e-5, nwl)]) # mode 2
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    elif 'plltdmrn' in caseStr.lower(): # Polluted Marine
        σ = [0.36, 0.70] # mode 1, 2,...
        rv = [0.11, 0.6]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.13965681],[0.31480467]]) # gives AOD=9.89*[0.0287, 0.0713]==1.0 total
        vals['vrtHght'] = [[1010],  [1010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.45, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.363, nwl)]) # mode 2
        vals['k'] = np.repeat(0.001, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(1e-5, nwl)]) # mode 2
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    elif 'pollution' in caseStr.lower():
        σ = [0.36, 0.64] # mode 1, 2,...
        rv = [0.11, 0.4]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.1787314], [0.0465671]]) # gives AOD=10*[0.091801,0.0082001]=1.0
        vals['vrtHght'] = [[1010],  [1010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.45, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.5, nwl)]) # mode 2
        vals['k'] = np.repeat(0.001, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.01, nwl)]) # mode 2 # NOTE: we cut this in half from XLSX
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    elif 'dust' in caseStr.lower(): # - Updated to match canonical case spreadsheet V25 -
        σ = [0.5, 0.75] # mode 1, 2,...
        rv = [0.1, 1.10]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['vol'] = np.array([[0.08656077541], [1.2667183842]]) # gives AOD=4*[0.13279, 0.11721]=1.0
        if 'onlysph' in caseStr.lower():
            vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        else:
            vals['sph'] = [[0.99999], [0.00001]] # mode fine sphere, coarse spheroid
            vals['vol'][1,0] = vals['vol'][1,0]*0.8864307902113797 # spheroids require scaling to maintain AOD
        vals['vrtHght'] = [[3010],  [3010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.46, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.51, nwl)]) # mode 2
        vals['k'] = np.repeat(1e-8, nwl) # mode 1
        mode2λ = [0.355, 0.380, 0.440, 0.532, 0.550, 0.870, 1.064, 2.100]
        mode2k = [0.0025, 0.0025, 0.0024, 0.0021, 0.0019, 0.0011, 0.0010, 0.0010]
        mode2Intrp = np.interp(wvls, mode2λ, mode2k)
        vals['k'] = np.vstack([vals['k'], mode2Intrp]) # mode 2 # THIS HAS A SPECTRAL DEPENDENCE IN THE SPREADSHEET
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    
    # Greema's TAMU cases
    elif 'd_tamu' in caseStr.lower(): # - Updated to match canonical case spreadsheet V25 -
        if defineRandom is not None: random_r = defineRandom # array of random numbers at least 4 element for this case        
        rv = [0.8+random_r[0]*3.2,0.8+random_r[1]*3.2] #coarse mode
        vals['vol'] = np.array([[1.5+random_r[2]*1.5]])/3 
        σ = [0.5, 0.75] # mode 1, 2,...
        rv = [0.1, 1.10]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['vol'] = np.array([[0.08656077541], [1.2667183842]]) # gives AOD=4*[0.13279, 0.11721]=1.0
        if 'nonsph' in caseStr.lower():
            vals['sph'] = [[0.00001], [0.00001]] # mode fine sphere, coarse spheroid
            vals['vol'][1,0] = vals['vol'][1,0]*0.8864307902113797 # spheroids require scaling to maintain AOD
        else:
            vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vrtHght'] = [[3010],  [3010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.46, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.51, nwl)]) # mode 2
        vals['k'] = np.repeat(1e-8, nwl) # mode 1
        mode2λ = [0.355, 0.380, 0.440, 0.532, 0.550, 0.870, 1.064, 2.100]
        mode2k = [0.0025, 0.0025, 0.0024, 0.0021, 0.0019, 0.0011, 0.0010, 0.0010]
        mode2Intrp = np.interp(wvls, mode2λ, mode2k)
        vals['k'] = np.vstack([vals['k'], mode2Intrp]) # mode 2 # THIS HAS A SPECTRAL DEPENDENCE IN THE SPREADSHEET
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0    
    elif 'var_tamu' in caseStr.lower(): # - Updated to match canonical case spreadsheet V25 -
        if defineRandom is not None: random_r  = defineRandom # array of random numbers       
        σ = 0.35+random_r[0]*0.3
        if 'fine' in caseStr.lower():
            rv = 0.145+random_r[1]*0.105
            vals['vol'] = np.array([[0.5+random_r[2]*0.5]])/3 # (currently gives AOD≈1 but changes w/ intensive props.)
        elif 'coarse' in caseStr.lower():
            rv = 0.8+random_r[4]*3.2
            vals['vol'] = np.array([[1.5+random_r[3]*1.5]])/3 # (currently gives AOD≈1 but changes w/ intensive props.)
        else:
            assert False, 'variable aerosol case must be appended with either fine or coarse'
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.0001]] if 'nonsph' in caseStr.lower() else [[0.99999]] # mode 1, 2,...
        vals['vrtHght'] = [[3010]] if 'lofted' in caseStr.lower() else  [[1010]]  # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500]] # Gaussian sigma in meters
        vals['n'] = np.interp(wvls, [wvls[0],wvls[-1]],   1.36+np.array([random_r[5],random_r[6]])*0.15)[None,:] # mode 1 # linear w/ λ
        vals['k'] =  np.repeat(0.0001+np.array(random_r[7])*0.015, len(wvls))# mode 1 # linear w/ λ
        # vals['k'] = np.interp(wvls, [wvls[0],wvls[-1]], 0.0001+np.array([random_r[7],random_r[8]])*0.015)[None,:] # mode 1 # linear w/ λ
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0    
    
    # Old AOS Cases
    # case01 is blank in V22 of the canoncial case spreadsheet...
    elif 'case02' in caseStr.lower(): # VERSION 22 (except vol & 2.1μm RI)
        σ = [0.4, 0.4] # mode 1, 2,...
        rv = [0.07, 0.25]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.07921839], [0.03682901]]) # gives AOD = [0.3046, 0.1954]
        if 'case02b' in caseStr.lower() or 'case02c' in caseStr.lower():
            vals['vol'] = vals['vol']/2.0
        vals['vrtHght'] = [[3500],  [3500]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[750],  [750]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.45, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.35, nwl)]) # mode 2
        vals['k'] = np.repeat(1e-8, nwl) if 'case02c' in caseStr.lower() else np.repeat(0.035, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(1e-8, nwl)]) # mode 2
        landPrct = 0
    elif caseStr.lower()=='case03': # VERSION 22 (2.1μm RRI)
        σ = [0.6, 0.6] # mode 1, 2,...
        rv = [0.1, 0.4]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.01387783], [0.01277042]]) # gives AOD = [0.0732, 0.026801]
        vals['vrtHght'] = [[750],  [750]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[250],  [250]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.4, nwl) # mode 1
        vals['n'] = np.vstack([vals['n'], np.repeat(1.35, nwl)]) # mode 2
        vals['k'] = np.repeat(0.002, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(1e-8, nwl)]) # mode 2
        landPrct = 0
    # case 04 is over land
    # case 05 has a water cloud in the scene
    elif 'case07' in caseStr.lower() or 'case08' in caseStr.lower(): # VERSION 22 (except spectral dep. of imag. in case 8)
        σ = [0.5, 0.7] # mode 1, 2,...
        rv = [0.1, 0.55]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vrtHght'] = [[750],  [750]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[250],  [250]] # mode 1, 2,... # Gaussian sigma in meters
        vals['k'] = np.repeat(0.002, nwl) # mode 1
        if 'case07' in caseStr.lower():
            vals['vol'] = np.array([[0.00580439], [0.00916563]]) # gives AOD = [0.0309, 0.0091]
            vals['n'] = np.repeat(1.415, nwl) # mode 1
            vals['n'] = np.vstack([vals['n'], np.repeat(1.363, nwl)]) # mode 2
            vals['k'] = np.vstack([vals['k'], np.repeat(1e-8, nwl)]) # mode 2
        else:
            vals['vol'] = np.array([[0.01191013], [0.0159524]]) # gives AOD = [0.064499, 0.0155  ]
            vals['n'] = np.repeat(1.42, nwl) # mode 1
            vals['n'] = np.vstack([vals['n'], np.repeat(1.52, nwl)]) # mode 2
            vals['k'] = np.vstack([vals['k'], np.repeat(0.002, nwl)]) # mode 2
        landPrct = 0 if 'case07' in caseStr.lower() else 100
    
    # Added by Anin to account for the aerosol models, PSD for the CAMP2Ex based simulation study
    elif 'coarse_mode_campex' in caseStr.lower(): # 
        try:
            nbins = 36
            radiusBin = np.logspace(np.log10(0.005), np.log10(15), nbins)
            
            # # Multiple mode in coarse mode will crash            
            σ = [0.45, 0.70] # mode 1, 2,...
            rv = [0.1, 0.6]*np.exp(3*np.power(σ,2))
            dvdr = logNormal(rv[0], σ[0], radiusBin)
            dvdr2 = logNormal(rv[1], σ[1], radiusBin)
            dvdlnr1 = dvdr[0]*radiusBin
            dvdlnr2 = dvdr2[0]*radiusBin
            vals['triaPSD'] = [np.around(dvdlnr1*[0.18], decimals=6),
                               np.around(dvdlnr2*[0.20], decimals=6)]
            vals['triaPSD'] = np.vstack(vals['triaPSD'])
            vals['sph'] = [[0.99999], [0.99999]]  # mode 1, 2,...
            # removed to avoid the descrepency in printing the aero vol conc in the output
            #vals['vol'] = np.array([[0.0477583], [0.7941207]]) # gives AOD=10*[0.0287, 0.0713]=1.0 total
            vals['vrtHght'] = [[2010],  [3010]] # mode 1, 2,... # Gaussian mean in meters
            vals['vrtHghtStd'] = [[500],  [500]]  # mode 1, 2,... # Gaussian sigma in meters
            vals['n'] = np.repeat(1.415, nwl)  # mode 1
            vals['n'] = np.vstack([vals['n'], np.repeat(1.363, nwl)]) # mode 2
            vals['k'] = np.repeat(0.002, nwl) # mode 1
            vals['k'] = np.vstack([vals['k'], np.repeat(1e-5, nwl)]) # mode 2
            landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
        except Exception as e:
            print('File loading error: check if the PSD file path is correct'\
                  ' or not\n %s' %e)
    elif 'aerosol_campex' in caseStr.lower(): # ALL VARIABLES WITH MODES MUST BE 2D (ie. var[mode,wl]) or [] (will not change these values)

        # A function to read the PSD from Jeff's file or ASCII and pass it to
        # this definition

        # Defining PSD of fine and coarse mode layers for a particular flight (fine mode for the case of CAMP2Ex
        kFMin = 0.001                   # minimum k value for fine mode
        kFMax = 0.03                    # maximum k value for fine mode
        kCMin = 0.0001                  # minimum k value for coarse mode
        kCMax = 0.0005                  # maximum k value for coarse mode
        ALH = [500, 1000, 1500, 2000]   # altitude of the fine mode layers in meters
        ALH_C = 1000                    # altitude of the coarse mode layer in meters
        ALHStd = 500                    # standard deviation of the altitude of the fine mode layers in meters
        nFM = 1.5                       # refractive index of the fine mode
        nCM = 1.4                       # refractive index of the coarse mode
        nFMStd = 0.15                   # standard deviation of the refractive index of the fine mode
        nCMStd = 0.05                   # standard deviation of the refractive index of the coarse mode
        sigma_ = 0.70                   # sigma of the coarse mode lognormal distribution
        mu_ = 0.6                       # mu of the coarse mode lognormal distribution
        sigma_Std = 0.05                # standard deviation of the sigma of the coarse mode lognormal distribution
        mu_Std = 0.035                  # standard deviation of the mu of the coarse mode lognormal distribution
        rMin = 0.005                    # minimum radius of the coarse mode lognormal distribution
        rMax = 15                       # maximum radius of the coarse mode lognormal distribution
        
        λ_LeiBi = [0.360, 0.380, 0.440, 0.550, 0.670, 0.870, 1.000, 1.570, 2.100]
        k_LeiBi = [3.e-7, 2.e-7, 1.e-7, 1.e-7, 1.e-7, 2.e-7, 3.e-7, 4.e-7, 5.e-7]   # dependency from Lei Bi et al 2019 [modified]
        lamb_k = np.interp(wvls, λ_LeiBi, k_LeiBi)*1e7                              # multiplied by 1e7 to make the k at 440 nm to be unity

        # read PSD bins
        try:
            # load the PSD from the file
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_dVDlnr72.pkl", 'rb') 
            dVdlnr = pickle.load(file)
            file.close()

            # load the radius bins from the file
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_r72.pkl", 'rb')
            radiusBin = pickle.load(file)
            file.close()

            # load the ratio of concentration of layers in the fine mode
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_dVDlnr72_V0Norm.pkl", 'rb')
            dVdlnrV0 = pickle.load(file)
            # file.close()
                
        except Exception as e:
            print('File loading error: check if the PSD file path is correct'
                  ' or not\n %s' % e)
        # modifying PSD based on the flight and layer. This will be updated to
        # include multiple layer information
       
        # flight
        if 'flight#' in caseStr.lower():
            try:
                matchRe = re.compile(r'flight#\d{2}')
                flight_numtemp = matchRe.search(caseStr.lower())
                flight_num = int(flight_numtemp.group(0)[7:])
            
            except Exception as e:
                print('Could not find a matching string pattern: %s' % e)
                flight_num = 1
    
            nlayer = 1
            flight_vrtHght = ALH[0]
            flight_vrtHghtStd = ALHStd
            multiMode = False
            # layer
            if 'layer#' in caseStr.lower():
                try:
                    matchRe = re.compile(r'layer#\d{2}')
                    layer_numtemp = matchRe.search(caseStr.lower())
                    nlayer = int(layer_numtemp.group(0)[6:])
                    # to use all layers
                    if not nlayer:
                        multiMode = True
                        print('Using all layers')
                    # to use only one layer
                    else:
                        flight_vrtHght = ALH[0]*nlayer
                        flight_vrtHghtStd = ALHStd   
                except Exception as e:
                    print('Could not find a matching string pattern: %s' % e)
                    flight_vrtHght = ALH[0]
                    flight_vrtHghtStd = ALHStd

            print('Using the PSD from the flight# %d and layer#'
                  ' %d' % (flight_num, nlayer))
            
            # update the PSD based on flight and layer information
            # This needs modification to use multiple layers
            if not nlayer == 0:
                vals['triaPSD'] = [np.around(dVdlnr[flight_num-1, nlayer-1, :], decimals=6)]
                vals['triaPSD'] = np.vstack(vals['triaPSD'])
        else:
            # using the first flight PSD
            print('Using the PSD from the first flight and first layer')
            vals['triaPSD'] = [np.around(dVdlnr[0, 0, :], decimals=6)]
            vals['triaPSD'] = np.vstack(vals['triaPSD'])

        # parameters above this line has to be modified [AP]
        if multiMode:
            # HACK: this is a hack to make the code work for the case of of one layer, basically forcing the concentration to be zero
            #vals['triaPSD'] = [np.around(dVdlnr[flight_num-1,layer-1,:]*[0.1], decimals=6)*0.000001]
            oneLayerHack = 0  # True if only one layer is used
            if 'olh' in caseStr.lower():
                zeroAeroConc = [0.000001]
                oneLayerHack = 1  # True if only one layer is used
                if 'wlayer' in caseStr.lower():
                    whichLayer = int(caseStr.lower()[caseStr.lower().find('wlayer')+6:])# layer number to be used
                else:
                    whichLayer = 0 # default layer number to be used
            else:
                zeroAeroConc = [0.000001]
                whichLayer = 0
            
            # Defining PSD of four layers for a particular flight (fine mode for the case of CAMP2Ex
            vals['triaPSD'] = [np.around(dVdlnr[flight_num-1,0,:]*[0.1], decimals=6),
                               np.around(dVdlnr[flight_num-1,1,:]*[0.1], decimals=6),
                               np.around(dVdlnr[flight_num-1,2,:]*[0.1], decimals=6),
                               np.around(dVdlnr[flight_num-1,3,:]*[0.1], decimals=6)]
            
            # change the fine mode concentration based on the flight and layer
            for i in range(len(vals['triaPSD'])):
                if i != whichLayer:
                    # Run this if one layer is used
                    if oneLayerHack:
                        vals['triaPSD'][i] = zeroAeroConc * vals['triaPSD'][i]
                    else:
                        vals['triaPSD'][i] = [dVdlnrV0[flight_num-1,i]] * vals['triaPSD'][i]
                        # Using the volume concentration ratio to keep the volume concentration profile same as the measurement
                    # print('Not using the PSD from the layer#%0.2d' % i)
            vals['triaPSD'] = np.array(vals['triaPSD'])
            sphFrac = 0.999 - round(rnd.uniform(0, 1))*0.99
            vals['sph'] = [sphFrac,
                           sphFrac,
                           sphFrac,
                           sphFrac] # mode 1, 2,...
            vals['vrtHght'] = [[ALH[0]], [ALH[1]], [ALH[2]], [ALH[3]]]      # mode 1, 2,... # Gaussian mean in meters #HACK: should be 3k
            vals['vrtHghtStd'] = [[ALHStd], [ALHStd], [ALHStd], [ALHStd]]   # mode 1, 2,... # Gaussian sigma in meters
            nAero_ = np.repeat(nFM + (rnd.uniform(-nFMStd, nFMStd)),nwl)    # fine mode aerosols
			
			# redefining the spectral refractive index based on the runtype (options are either 
            if 'flatfine' in caseStr.lower():
                nAero = slope4RRI(nAero_, wvls, slope=0)
                kAero = loguniform(kFMin, kFMax)*np.repeat(1, nwl)
            elif 'urban' in caseStr.lower():
                nAero = slope4RRI(nAero_, wvls, slope=-0.01)
                kAero = loguniform(kFMin, kFMax)*np.repeat(1, nwl)
            else:
                nAero = slope4RRI(nAero_, wvls)
                kAero = loguniform(kFMin, kFMax)*lamb_k
            
            vals['n'] = [list(nAero),
                         list(nAero),
                         list(nAero),
                         list(nAero)] # mode 1
            
            vals['k'] = [list(kAero),
                         list(kAero),
                         list(kAero),
                         list(kAero)]# mode 1
            # Adding the coarse mode in the simulation
            if 'coarse' in caseStr.lower():
                # This is hard-coded, which limits its applicability for generalization
                # GRASP needs to be modified to make use of triangle and lognormal bins
                # together
                σ = [sigma_ + rnd.uniform(-sigma_Std, sigma_Std)]  # Coarse mode
                rv = [mu_ + rnd.uniform(-mu_Std, mu_Std)]*np.exp(3*np.power(σ,2)) # Coarse mode (rv = rn*e^3σ)
                nbins = np.size(radiusBin)
                radiusBin_ = np.logspace(np.log10(rMin), np.log10(rMax), nbins)
                dvdr = logNormal(rv[0], σ[0], radiusBin_)
                dvdlnr = dvdr[0]*radiusBin_
                
                if 'nocoarse' in caseStr.lower(): multFac = 0.0001
                else: multFac = 0.77
                vals['triaPSD'] = np.vstack([vals['triaPSD'],
                                            [dvdlnr*multFac]])
                vals['sph'] = vals['sph'] + [0.999 - round(rnd.uniform(0, 1))*0.99]
                # removed to avoid the discrepancy in printing the aero vol conc in the output
                vals['vrtHght'] = vals['vrtHght'] + [[ALH_C]]
                vals['vrtHghtStd'] = vals['vrtHghtStd'] + [[ALHStd]]
                nAero_ = np.repeat(nCM + (rnd.uniform(-nCMStd, nCMStd)), nwl)
                # redefining the spectral refractive index based on the runtype (options are either 
                if 'flatcoarse' in caseStr.lower():
                    nAero = slope4RRI(nAero_, wvls, slope=0)
                    kAero = loguniform(kCMin, kCMax)*np.repeat(1, nwl)
                else:
                    nAero = slope4RRI(nAero_, wvls)
                    # k
                    
                    kAero = loguniform(kCMin, kCMax)*lamb_k

                vals['n'] = vals['n'] + [list(nAero)]
                vals['k'] = vals['k'] + [list(kAero)]
        else:
            vals['sph'] = [0.999 - round(rnd.uniform(0, 1))*0.99] # mode 1, 2,...
            vals['vol'] = np.array([0.0017]) # gives AOD=4*[0.2165, 0.033499]=1.0
            vals['vrtHght'] =[flight_vrtHght] # mode 1, 2,... # Gaussian mean in meters #HACK: should be 3k
            vals['vrtHghtStd'] = [flight_vrtHghtStd] # mode 1, 2,... # Gaussian sigma in meters
            nAero_ = np.repeat(nFM + (rnd.uniform(-nFMStd, nFMStd)),nwl)
            if 'flat' in caseStr.lower():
                nAero = slope4RRI(nAero_, wvls, slope=0)
                kAero = loguniform(kFMin,kFMax)*np.repeat(1, nwl)
            elif 'urban' in caseStr.lower():
                nAero = slope4RRI(nAero_, wvls, slope=0)
                kAero = loguniform(kFMin,kFMax)*np.repeat(1, nwl)
            else:
                nAero = slope4RRI(nAero_, wvls)
                # k
                kAero = loguniform(kCMin, kCMax)*lamb_k

            # may have bug here!!!
            vals['n'] = np.vstack([vals['n'], nAero]) # mode 2 
            vals['k'] = [kAero] # mode 1
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    elif 'fit_campex' in caseStr.lower(): #Fix: This has to be modified to make it work with latest simulation setup
        '''
        This is a special case where the PSD is read from the Jeff's file and fitted a lognormal distribution to the fine mode
        and some modifications has been done to the >500nm diameter bin to make it a lognormal distribution. The documentation 
        regarding this technique is available in the GSFC_ESI_Scripts/Jeff-Project/README.md file
        '''
        
        # find the params based on the flight and layer #
        # A function to read the PSD from Jeff's file or ASCII and pass it to
        # this definition

        # Defining PSD of four layers for a particular flight (fine mode for the case of CAMP2Ex
        kFMin = 0.001                   # minimum k value for fine mode
        kFMax = 0.03                    # maximum k value for fine mode
        kCMin = 0.0001                  # minimum k value for coarse mode
        kCMax = 0.0005                  # maximum k value for coarse mode
        ALH = [500, 1000, 1500, 2000]   # altitude of the fine mode layers in meters
        ALH_C = 1000                    # altitude of the coarse mode layer in meters
        ALHStd = 500                    # standard deviation of the altitude of the fine mode layers in meters
        nFM = 1.5                       # refractive index of the fine mode
        nCM = 1.4                       # refractive index of the coarse mode
        nFMStd = 0.15                   # standard deviation of the refractive index of the fine mode
        nCMStd = 0.05                   # standard deviation of the refractive index of the coarse mode
        sigma_ = 0.70                   # sigma of the coarse mode lognormal distribution
        mu_ = 0.6                       # mu of the coarse mode lognormal distribution
        sigma_Std = 0.05                # standard deviation of the sigma of the coarse mode lognormal distribution
        mu_Std = 0.035                  # standard deviation of the mu of the coarse mode lognormal distribution
        rMin = 0.005                    # minimum radius of the coarse mode lognormal distribution
        rMax = 15                       # maximum radius of the coarse mode lognormal distribution

        λ_LeiBi = [0.360, 0.380, 0.440, 0.550, 0.670, 0.870, 1.000, 1.570, 2.100]
        k_LeiBi = [3.e-7, 2.e-7, 1.e-7, 1.e-7, 1.e-7, 2.e-7, 3.e-7, 4.e-7, 5.e-7]   # dependency from Lei Bi et al 2019 [modified]
        lamb_k = np.interp(wvls, λ_LeiBi, k_LeiBi)*1e7                              # multiplied by 1e7 to make the k at 440 nm to be unity

        # read PSD bins
        try:
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_dVDlnr72_fitParams.pkl", 'rb')
            dVdlnrParams = pickle.load(file)
            file.close()
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_r72.pkl", 'rb')
            radiusBin = pickle.load(file)
            file.close()
        except Exception as e:
            print('File loading error: check if the PSD file path is correct'\
                  ' or not\n %s' %e)
        # flight
        if 'flight#' in caseStr.lower():
            try:
                matchRe = re.compile(r'flight#\d{2}')
                flight_numtemp = matchRe.search(caseStr.lower())
                flight_num = int(flight_numtemp.group(0)[7:])
                
            except Exception as e:
                print('Could not find a matching string pattern: %s' %e)
                flight_num = 1
        
            nlayer = 1
            flight_vrtHght = ALH[0]
            flight_vrtHghtStd = ALHStd
            multiMode = False
            # layer
            if 'layer#' in caseStr.lower():
                try:
                    matchRe = re.compile(r'layer#\d{2}')
                    layer_numtemp = matchRe.search(caseStr.lower())
                    nlayer = int(layer_numtemp.group(0)[6:]) 
                    # to use all layers
                    if not nlayer:
                        multiMode = True
                    # to use only one layer
                    else:
                        flight_vrtHght = ALH[0]*nlayer # FIXME: this is hard-coded, it needs to changed to make it work with other cases
                        flight_vrtHghtStd = ALHStd   
                except Exception as e:
                    print('Could not find a matching string pattern: %s' %e)
                    flight_vrtHght = ALH[0]
                    flight_vrtHghtStd = ALHStd

            print('Using the PSD from the flight# %d and layer#'\
                  ' %d' %(flight_num, nlayer))
        else:
            # using the first flight PSD
            print('Using the PSD from the first flight and first layer')
            vals['triaPSD'] = [np.around(dVdlnr[0,0,:], decimals=6)]
            vals['triaPSD'] = np.vstack(vals['triaPSD'])
        flight_num = flight_num-1
        # Defining a random PSD for coarse mode 
        rndmSigma = sigma_ +rnd.uniform(-sigma_Std, sigma_Std) # mode 1, 2,...
        rndmMuFit = (mu_ +rnd.uniform(-mu_Std, mu_Std))*np.exp(3*np.power(rndmSigma,2))

        # Stacking up PSD info
        sigmaFit = [dVdlnrParams['sigma'][flight_num, 0],
                    dVdlnrParams['sigma'][flight_num, 1],
                    dVdlnrParams['sigma'][flight_num, 2],
                    dVdlnrParams['sigma'][flight_num, 3],
                    rndmSigma]               
        muFit = [dVdlnrParams['mu'][flight_num, 0],
                dVdlnrParams['mu'][flight_num, 1],
                dVdlnrParams['mu'][flight_num, 2],
                dVdlnrParams['mu'][flight_num, 3],
                rndmMuFit] # The values based on the rndm variation limit in the previous model 
        # Not using this at the moment (but if we randomize the fine mode fraction 
        # we wil have to play with/adjust this)
        V0Fit =[dVdlnrParams['V0'][flight_num, 0],
                dVdlnrParams['V0'][flight_num, 1],
                dVdlnrParams['V0'][flight_num, 2],
                dVdlnrParams['V0'][flight_num, 3],
                1]
        
        σ = sigmaFit # mode 1, 2,...
        rv = muFit
        vals['lgrnm'] = np.vstack([rv, σ]).T
        sphFrac = 0.999 - round(rnd.uniform(0, 1))*0.99
        sphFrac2= 0.999 - round(rnd.uniform(0, 1))*0.99                     # Coarse mode
        vals['sph'] = [sphFrac, sphFrac, sphFrac, sphFrac, sphFrac2]        # mode 1, 2,...
        vals['vol'] = np.array([[0.35], [0.35],[0.35], [0.35], [4.1]])      # gives AOD=10*[0.0287, 0.0713]=1.0 total
        # ----------------------------------------------------------------------#
        # if using only one layer in fine mode
        # ----------------------------------------------------------------------#
        #HACK: this is a hack to make the code work for the case of of one layer, basically forcing the concentration to be zero
        oneLayerHack = 0  # True if only one layer is used
        if 'olh' in caseStr.lower():
            zeroAeroConc = [0.0001]
            oneLayerHack = 1  # True if only one layer is used
            if 'wlayer' in caseStr.lower(): # Currently not used, but can be used to select a layer
                whichLayer = int(caseStr.lower()[caseStr.lower().find('wlayer')+6:])# layer number to be used
            else:
                whichLayer = 0   
        else:
            zeroAeroConc = [0.0001]
            whichLayer = 0  
            # Defining PSD of four layers for a particular flight (fine mode for the case of CAMP2Ex)
        for i in range(len(vals['vol'])):
            if i != whichLayer:
                if i != 4:
                    # print(vals['vol'][i])
                    # Run this if one layer is used
                    if oneLayerHack:
                        vals['vol'][i] = zeroAeroConc * vals['vol'][i]
                    else:
                        vals['vol'][i] = [dVdlnrParams['dVdLog10D_GRASP_SumNorm'][flight_num,i]] * vals['vol'][i]
                        # Using the volume concentration ratio to keep the volume concentration profile same as the measurement
                    # print('Not using the PSD from the layer#%0.2d' % i)

        vals['vrtHght'] = [[ALH[0]], [ALH[1]], [ALH[2]], [ALH[3]], [ALH_C]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [ALHStd, ALHStd, ALHStd, ALHStd, ALHStd]       # mode 1, 2,... # Gaussian sigma in meters
        # n
        nAero_ = np.repeat(nFM + (rnd.uniform(-nFMStd, nFMStd)), nwl)
        if 'flatfine' in caseStr.lower():
            nAero = slope4RRI(nAero_, wvls, slope=0)
            kAero = loguniform(kFMin, kFMax)*np.repeat(1, nwl)
        elif 'urban' in caseStr.lower():
            nAero = slope4RRI(nAero_, wvls, slope=-0.01)
            kAero = loguniform(kFMin, kFMax)*np.repeat(1, nwl)
        else:
            nAero = slope4RRI(nAero_, wvls)
            # k
            kAero = loguniform(kFMin, kFMax)*lamb_k
        vals['n'] = np.vstack([nAero,
                               nAero,
                               nAero,
                               nAero]) # mode 1,2,...
        vals['k'] = np.vstack([ kAero,
                                kAero,
                                kAero,
                                kAero])# mode 1,2,...
        #-----------------------------------------------------------------------#
        # Coarse mode
        #-----------------------------------------------------------------------#
        # Adding the coarse mode in the simulation
        if 'coarse' in caseStr.lower():
            nAero_ = np.repeat(nCM + rnd.uniform(-nCMStd, nCMStd), nwl)         # Coarse mode aerosols real refractive index
            if 'flatcoarse' in caseStr.lower():
                nAero = slope4RRI(nAero_, wvls, slope=0)
                kAero = loguniform(kCMin, kCMax)**np.repeat(1, nwl)
            else:
                nAero = slope4RRI(nAero_, wvls)
                kAero = loguniform(kCMin, kCMax)*lamb_k
            vals['n'] = np.vstack([vals['n'], nAero]) # mode 5
            
            vals['k'] = np.vstack([vals['k'], kAero]) # mode 5
        landPrct = 100 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    else:
        assert False, 'No match for caseStr: '+caseStr+'!'
    # MONOMDE [keep only the large of the two (or more) modes]
    if 'monomode' in caseStr.lower():
        bigMode = np.argmax(vals['vol'])
        for key in ['vol','n','k','sph','lgrnm','vrtHght','vrtHghtStd']:
            vals[key] = np.atleast_2d(np.array(vals[key])[bigMode,:])
    # OCEAN MODEL
    if landPrct<100:
        λ=[0.355, 0.380, 0.440, 0.532, 0.550, 0.870, 1.064, 2.100]
        if 'chl' in caseStr.lower():
            #R=[0.0046195003, 0.0050949964, 0.0060459884, 0.0024910956,	0.0016951599, 0.00000002, 0.00000002, 0.00000002] # SIT-A canonical values, TODO: need to double check these units
            R=[0.02, 0.02, 0.02, 0.02,  0.01, 0.0005, 0.00000002, 0.00000002] # Figure 8, Chowdhary et al, APPLIED OPTICS Vol. 45, No. 22 (2006), also need to check units...
        elif 'open_ocean' in caseStr.lower():
            R=[0.012, 0.014, 0.010, 0.00249109,	0.00169515, 0.00000002, 0.00000002, 0.00000002]
        else: 
            R=[0.00000002, 0.00000002, 0.00000002, 0.00000002,	0.00000002, 0.00000002, 0.00000002, 0.00000002]
        lambR = np.interp(wvls, λ, R)
        FresFrac = 0.999999*np.ones(nwl)
        cxMnk = (7*0.00512+0.003)/2*np.ones(nwl) # 7 m/s
        vals['cxMnk'] = np.vstack([lambR, FresFrac, cxMnk])
    # LAND SURFACE BRDF [Polar07_reflectanceTOA_cleanAtmosphere_landSurface_V7.xlsx]
    if landPrct>0: # we havn't programed these yet
        λ=[0.415, 0.470, 0.555, 0.659, 0.865, 1.24, 1.64, 2.13] # this should be ordered (interp behavoir is unexpected otherwise)
        if 'desert' in caseStr.lower(): # mean of July 1st 2019 Sahara MIAIC MODIS RTLS (MCD19A3.A2019177.h18v06.006.2019186034811.hdf)
            iso = [0.0859, 0.1453, 0.2394, 0.3838, 0.4619, 0.5762, 0.6283, 0.6126]
            vol = [0.4157, 0.4157, 0.4157, 0.4157, 0.4157, 0.4157, 0.4157, 0.4157] # MAIAC_vol/MAIAC_iso
            geo = [0.0262, 0.0262, 0.0262, 0.0262, 0.0262, 0.0262, 0.0262, 0.0262] # MAIAC_geo/MAIAC_iso
        elif 'vegetation' in caseStr.lower(): # mean of July 1st 2019 SEUS MIAIC MODIS RTLS (MCD19A3.A2019177.h11v05.006.2019186033524.hdf)
            iso = [0.0237, 0.0368, 0.0745, 0.0560, 0.4225, 0.4104, 0.2457, 0.1128]
            vol = [0.6073, 0.6073, 0.6073, 0.6073, 0.6073, 0.6073, 0.6073, 0.6073] # MAIAC_vol/MAIAC_iso
            geo = [0.1411, 0.1411, 0.1411, 0.1411, 0.1411, 0.1411, 0.1411, 0.1411] # MAIAC_geo/MAIAC_iso
        else:
            assert False, 'Land surface type not recognized!'
        lambISO = np.interp(wvls, λ, iso)
        lambVOL = np.interp(wvls, λ, vol)
        lambGEO = np.interp(wvls, λ, geo)
        vals['brdf'] = np.vstack([lambISO, lambVOL, lambGEO])
    # LAND BPDF
    if landPrct>0:
        if 'desert' in caseStr.lower(): # OSSE original sept. 1st test case over Sahara, BPDFCoef=7.3, NDVI=0.1
            vals['bpdf'] = 6.6564*np.ones([1,nwl]) # exp(-VLIDORT_NDVI)*VLIDORT_C)
        elif 'vegetation' in caseStr.lower(): # OSSE original sept. 1st test case over SEUS, BPDFCoef=6.9, NDVI=0.9
            vals['bpdf'] = 2.6145*np.ones([1,nwl]) # exp(-VLIDORT_NDVI)*VLIDORT_C)
        else:
            assert False, 'Land surface type not recognized!'
    # LIDAR PROFILE SHAPE
    lidarMeasLogical = np.isclose(34.5, [mv['meas_type'][0] for mv in nowPix.measVals], atol=5) # measurement types 30-39 reserved for lidar; if first meas_type is LIDAR, they all should be
    if lidarMeasLogical.any():
        lidarInd = lidarMeasLogical.nonzero()[0][0]
        hValTrgt = np.array(nowPix.measVals[lidarInd]['thetav'][0:nowPix.measVals[lidarInd]['nbvm'][0]]) # HINT: this assumes all LIDAR measurement types have the same vertical range values
        vals['vrtProf'] = np.empty([len(vals['vrtHght']), len(hValTrgt)])
        for i, (mid, rng) in enumerate(zip(vals['vrtHght'], vals['vrtHghtStd'])):
            bot = max(mid[0]-2*rng[0],0)
            top = mid[0]+2*rng[0]
            vals['vrtProf'][i,:] = np.logical_and(np.array(hValTrgt) > bot, np.array(hValTrgt) <= top)*1+0.000001
            if vals['vrtProf'][i,1]>1: vals['vrtProf'][i,0]=0.01 # keep very small amount in top bin if upper layer
            if vals['vrtProf'][i,-2]>1: vals['vrtProf'][i,-1]=1.0 # fill bottom bin if lowwer layer
        del vals['vrtHght']
        del vals['vrtHghtStd']
    return vals, landPrct

def setupConCaseYAML(caseStrs, nowPix, baseYAML, caseLoadFctr=1, caseHeightKM=None, simBldProfs=None, defineRandom = None): # equal volume weighted marine at 1km & smoke at 4km -> caseStrs='marine+smoke', caseLoadFctr=[1,1], caseHeightKM=[1,4]
    """ nowPix needed to: 1) set land percentage of nowPix and 2) get number of wavelengths """
    aeroKeys = ['traiPSD','lgrnm','sph','vol','vrtHght','vrtHghtStd','vrtProf','n','k', 'landPrct']
    vals = dict()
    if type(caseLoadFctr)==str and 'randLogNrm' in caseLoadFctr: # e.g., 'randLogNrm0.2' for aod_median=0.2
        sigma = np.log(2) # hardcode σg=2 for lognormal 
        medianAOD = float(re.match('[A-z]+([0-9\.]+)', caseLoadFctr).group(1))
        loading = [l for _,l in  splitMultipleCases(caseStrs)]
        normFact = medianAOD/sum(loading) # cases already have loading, we scale by this to get medianAOD
        caseLoadFctr = np.random.lognormal(np.log(normFact), sigma)
    assert type(caseLoadFctr)!=str, '%s was not a recognized value for caseLoadFctr!' % caseLoadFctr
    for caseStr,loading in splitMultipleCases(caseStrs, caseLoadFctr): # loop over all cases and add them together
        valsTmp, landPrct = conCaseDefinitions(caseStr, nowPix, defineRandom)
        for key in valsTmp.keys():
            if key=='vol':
                if 'fixedcoarse' in caseStr.lower(): # HACK: this is a hack to get the coarse mode to be fixed and works for only CAMP2EX data, because index 4 is the coarse mode 
                    if 'zerocoarse' in caseStr.lower(): # HACK: this is a hack to get the coarse mode to be fixed and works for only CAMP2EX data, because index 4 is the coarse mode
                        valsTmp[key] = np.vstack([loading*valsTmp[key][:4], valsTmp[key][4]/1000])
                    else: valsTmp[key] = np.vstack([loading*valsTmp[key][:4], valsTmp[key][4]/44])
                else: valsTmp[key] = loading*valsTmp[key]  
            elif key=='vrtHght' and caseHeightKM:
                valsTmp[key][:] = caseHeightKM*1000
            if key=='triaPSD':
                if 'fixedcoarse' in caseStr.lower(): # HACK: this is a hack to get the coarse mode to be fixed and works for only CAMP2EX data, because index 4 is the coarse mode 
                    if 'zerocoarse' in caseStr.lower(): # HACK: this is a hack to get the coarse mode to be fixed and works for only CAMP2EX data, because index 4 is the coarse mode
                        valsTmp[key] = np.vstack([loading*valsTmp[key][:4], valsTmp[key][4]/250])
                    else: valsTmp[key] = np.vstack([loading*valsTmp[key][:4], valsTmp[key][4]/13])
                else: valsTmp[key] = loading*valsTmp[key]
            if key in aeroKeys and key in vals:
                    vals[key] = np.vstack([vals[key], valsTmp[key]])
            else: # implies we take the surface parameters from the last case
                vals[key] = valsTmp[key]
    nowPix.land_prct = landPrct
    if simBldProfs is not None:
        msg = 'Using sim_builder profiles requires 4 modes ordered [TOP_F, TOP_C, BOT_F, BOT_C]!'
        assert vals['vrtProf'].shape==simBldProfs.shape, msg
        vrtOrdVld = vals['vrtProf'][0,-1]<1e-4 and vals['vrtProf'][2,-1]>1e-4 # bottom bin filled in mode 2 but not in mode 0
        modeOrdVld = vals['lgrnm'][0,0]<vals['lgrnm'][1,0] and vals['lgrnm'][2,0]<vals['lgrnm'][3,0]
        assert vrtOrdVld and modeOrdVld, msg
        vals['vrtProf'] = simBldProfs
    yamlObj = rg.graspYAML(baseYAML, newTmpFile=('FWD_%s' % caseStrs))
    yamlObj.setMultipleCharacteristics(vals)
    yamlObj.access('stop_before_performing_retrieval', True, write2disk=True)
    return yamlObj

def splitMultipleCases(caseStrs, caseLoadFct=1):
    cases = []
    loadings = []
    for case in caseStrs.split('+'): # HINT: Sharons files' reader output is ordered [TOP_F, TOP_C, BOT_F, BOT_C]
        if 'case06a' in case.lower():
            cases.append(case.replace('case06a','smoke')) # smoke base τ550=1.0
            loadings.append(0.25*caseLoadFct)
            cases.append(case.replace('case06a','marine')) # marine base τ550=1.0
            loadings.append(0.1*caseLoadFct)
        elif 'case06b' in case.lower():
            cases.append(case.replace('case06b','smoke'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case06b','marine'))
            loadings.append(0.25*caseLoadFct)
        elif 'case06c' in case.lower():
            cases.append(case.replace('case06c','smoke'))
            loadings.append(0.25*caseLoadFct)
            cases.append(case.replace('case06c','pollution')) # pollution base τ550=1.0
            loadings.append(0.1*caseLoadFct)
        elif 'case06d' in case.lower():
            cases.append(case.replace('case06d','smoke'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case06d','pollution'))
            loadings.append(0.25*caseLoadFct)
        elif 'case06e' in case.lower():
            cases.append(case.replace('case06e','dust')) # dust base τ550=1.0
            loadings.append(0.25*caseLoadFct)
            cases.append(case.replace('case06e','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case06f' in case.lower():
            cases.append(case.replace('case06f','dust'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case06f','marine'))
            loadings.append(0.25*caseLoadFct)
        elif 'case06g' in case.lower():
            cases.append(case.replace('case06g','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case06g','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case06h' in case.lower():
            cases.append(case.replace('case06h','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case06h','plltdMrn')) # plltdMrn base τ550=1.0
            loadings.append(0.1*caseLoadFct)
        elif 'case06i' in case.lower():
            cases.append(case.replace('case06i','smoke'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case06i','pollution'))
            loadings.append(0.5*caseLoadFct)
        elif 'case06j' in case.lower():
            cases.append(case.replace('case06j','dustNonsph'))
            loadings.append(0.25*caseLoadFct)
            cases.append(case.replace('case06j','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case06k' in case.lower():
            cases.append(case.replace('case06k','dustNonsph'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case06k','marine'))
            loadings.append(0.25*caseLoadFct)
        elif 'case08a' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.05*caseLoadFct)
            cases.append(case.replace('case08','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08b' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.2*caseLoadFct)
            cases.append(case.replace('case08','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08c' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.3*caseLoadFct)
            cases.append(case.replace('case08','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08d' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.09*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08e' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.4*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08f' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.9*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08g' in case.lower():
            cases.append(case.replace('case08','dust'))
            loadings.append(0.11*caseLoadFct)
            cases.append(case.replace('case08','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08h' in case.lower():
            cases.append(case.replace('case08','dust'))
            loadings.append(0.44*caseLoadFct)
            cases.append(case.replace('case08','marine'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08i' in case.lower():
            cases.append(case.replace('case08','dustDesert'))
            loadings.append(0.18*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08j' in case.lower():
            cases.append(case.replace('case08','dustDesert'))
            loadings.append(0.49*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.1*caseLoadFct)
        elif 'case08k' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','marine'))
            loadings.append(0.05*caseLoadFct)
        elif 'case08l' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','plltdMrn'))
            loadings.append(0.12*caseLoadFct)
        elif 'case08m' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.09*caseLoadFct)
        elif 'case08n' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.33*caseLoadFct)
        elif 'case08o' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.7*caseLoadFct)
        elif 'case08p' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','smoke'))
            loadings.append(1.0*caseLoadFct)
        elif 'case08q' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(1.0*caseLoadFct)
        elif 'case08r' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','pollution'))
            loadings.append(1.0*caseLoadFct)
        elif 'case08s' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(1.0*caseLoadFct)
        elif 'case08t' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','dust'))
            loadings.append(1.0*caseLoadFct)
        elif 'case08u' in case.lower():
            cases.append(case.replace('case08','smoke'))
            loadings.append(0.00001)
            cases.append(case.replace('case08','dustDesert'))
            loadings.append(1.0*caseLoadFct)
        #-------------------Added by Anin-------------------#
        # Triangular case - if the size distribution is in the triangular bin
        elif 'campex_tria' in case.lower():                                 
            # for flat optical properties (n and k)
            if 'flat' in case.lower(): 	                                    
                cases.append(case.replace('campex','aerosol_campex_flat'))
            # Optical properties have close resemblence to the urban aerosols
            elif 'urban':                                                   
                cases.append(case.replace('campex','aerosol_campex_fine_urban'))
            else:
                # smoke base τ550=1.0
                cases.append(case.replace('campex','aerosol_campex'))       
            loadings.append(0.5*caseLoadFct)
        # BiModal case    
        elif 'campex_bi' in case.lower():
            if 'flat' in case.lower():
                cases.append(case.replace('campex','fit_campex_flat'))
            elif 'urban':
                cases.append(case.replace('campex','fit_campex_fine_urban'))
            else:
                cases.append(case.replace('campex','fit_campex'))
            loadings.append(0.0937*caseLoadFct)
        #-------------------Added by Greema-------------------#
        elif 'tamu_variable_sphere' in case.lower():
            cases.append(case.replace('tamu_variable_sphere','var_tamufine'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('tamu_variable_sphere','var_tamucoarse'))
            loadings.append(0.4*caseLoadFct)

        elif 'tamu_variable' in case.lower():
            cases.append(case.replace('tamu_variable','var_tamufine_nonsph'))
            loadings.append(0.1*caseLoadFct)
            cases.append(case.replace('tamu_variable','var_tamucoarse_nonsph'))
            loadings.append(0.4*caseLoadFct)
        else:
            cases.append(case)
            loadings.append(caseLoadFct)
    return zip(cases, loadings)
    
def yingxiProposalSmokeModels(siteName, wvls):
    dampFact = 2 # HACK!!!
#     dampFact = 1.414 # basically saying half the variability comes from AERONET retrieval error... (does not apply to concentration)
    vals = dict()
    aeronetWvls = [0.440, 0.675, 0.870, 1.020]
    if siteName=='Huambo':
        #                       rvF     std(rvF)                    rvC  std(rvC)
        rv = np.r_[np.random.normal(0.13776, 0.0176/dampFact), np.random.normal(3.669, 0.2308/dampFact)] #         rvVar = [0.017691875, 0.230835572]
        σ = np.r_[np.random.normal(0.38912, 0.0272/dampFact),	np.random.normal(0.6254, 0.0355/dampFact)] #         σVar = [0.027237159, 0.035588349]
        volFine = np.random.lognormal(np.log(0.07974), 0.3) #         volVar = [0.025625888, 0.013701423]
        volCoarse = np.random.lognormal(np.log(0.03884), 0.4)
        aeronet_n = [1.475, 1.504, 1.512, 1.513] # std 0.05±~0.005
        aeronet_k = [0.026, 0.022, 0.023, 0.023] # std 0.0055±0.002
    elif siteName=='NASA_Ames':
        rv = np.r_[np.random.normal(0.17271, 0.03808/dampFact), np.random.normal(3.00286, 0.5554/dampFact)]
        σ = np.r_[np.random.normal(0.46642, 0.05406/dampFact), np.random.normal(0.67840, 0.079605/dampFact)]
        volFine = np.random.lognormal(np.log(0.077829268), 1) # mean(vol)≈std(vol) => 2nd argument = 1
        volCoarse = np.random.lognormal(np.log(0.045731707), 1)
        aeronet_n = [1.4939, 1.5131, 1.5149, 1.5079] # std 0.05±~0.013
        aeronet_k = [0.00646, 0.00487, 0.00510, 0.00515] # std 0.003±0.0008
    else:
        assert False, 'Unkown siteName string!'
    volFine = max(volFine, 1e-8)
    volCoarse = max(volCoarse, 1e-8)
    vals['vol'] = np.array([[volFine],[volCoarse]])
    rv[rv<0.08] = 0.08
    rv[rv>6.0] = 6.0
    σ[σ<0.25] = 0.25
    σ[σ>0.8] = 0.8
    vals['lgrnm'] = np.vstack([rv, σ]).T
    n = np.interp(wvls, aeronetWvls, aeronet_n) + np.random.normal(0, 0.05/dampFact)
    n[n<1.35] = 1.35
    n[n>1.69] = 1.69
    # HACK: no dampFacts on these guys AND an extra special-k coarse AND we doubled std(k_fine)
#     k = np.interp(wvls, aeronetWvls, aeronet_k) + np.random.normal(0, 0.004) # 0.004 is average of Huambo and Ames std(k) above
    k = np.interp(wvls, aeronetWvls, aeronet_k) + np.random.normal(0, 2*0.004) # 0.004 is average of Huambo and Ames std(k) above
    kCoarse = np.interp(wvls, aeronetWvls, aeronet_k) # 0.004 is average of Huambo and Ames std(k) above
    k[k<1e-3] = 1e-3
    k[k>0.1]  = 0.1
    vals['n'] = np.array([n, n])
    vals['k'] = np.array([k, kCoarse])
    vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
    hgt = 1500+4500*np.random.rand()
#     hgt = 3000
    vals['vrtHght'] = [[hgt],  [hgt]] # mode 1, 2,... # Gaussian mean in meters #HACK: should be 3k
    vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
    return vals
