#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rnd
# to use the same seed for random number generator
rnd.default_rng(seed=33)
import tempfile
import os
import sys
import pickle
import re
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))
import runGRASP as rg

def conCaseDefinitions(caseStr, nowPix):
    """ '+' is used to seperate multiple cases (implemented in splitMultipleCases below)
        This function should insensitive to trailing characters in caseStr
            (e.g. 'smokeDesert' and 'smokeDeserta2' should produce same result)
    """
    vals = dict()
    wvls = np.unique([mv['wl'] for mv in nowPix.measVals])
    nwl = len(wvls)
    """ variable type appended options: 'fine'/'coarse', 'nonsph' and 'lofted' """
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
    elif 'huambo' in caseStr.lower():
        vals = yingxiProposalSmokeModels('Huambo', wvls)
        landPrct = 0 if 'ocean' in caseStr.lower() else 100
    elif 'nasaames' in caseStr.lower():
        vals = yingxiProposalSmokeModels('NASA_Ames', wvls)
        landPrct = 0 if 'ocean' in caseStr.lower() else 100
    elif 'clean' in caseStr.lower():
        σ = [0.4, 0.68] # mode 1, 2,...
        rv = [0.1, 0.84]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        if 'nonsph' in caseStr.lower():
            vals['sph'] = [[0.00001], [0.00001]] # mode 1, 2,...
        else:
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
    # Added by Anin
    elif 'aerosol_campex' in caseStr.lower(): # ALL VARIABLES WITH MODES MUST BE 2D (ie. var[mode,wl]) or [] (will not change these values)

        # A function to read the PSD from Jeff's file or ASCII and pass it to
        # this definition
        # read PSD bins
        try:
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_dVDlnr36.pkl", 'rb')
            dVdlnr = pickle.load(file)
            file.close()
            file = open("../../GSFC-ESI-Scripts/Jeff-Project/"
                        "Campex_r36.pkl", 'rb')
            radiusBin = pickle.load(file)
            file.close()
        except Exception as e:
            print('File loading error: check if the PSD file path is correct'\
                  ' or not\n %s' %e)
        # modifying PSD based on the flight and layer. This will be updated to
        # include multiple layer information
        
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
            flight_vrtHght = 500
            flight_vrtHghtStd = 200
            # layer
            if 'layer#' in caseStr.lower():
                try:
                    matchRe = re.compile(r'layer#\d{2}')
                    layer_numtemp = matchRe.search(caseStr.lower())
                    nlayer = int(layer_numtemp.group(0)[6:])
                    flight_vrtHght = 500*nlayer
                    flight_vrtHghtStd = 200
                        
                except Exception as e:
                    print('Could not find a matching string pattern: %s' %e)
                    flight_vrtHght = 500
                    flight_vrtHghtStd = 200

            print('Using the PSD from the flight# %d and layer#'\
                  ' %d' %(flight_num, nlayer))
            
            # update the PSD based on flight an d layer information
            # This needs modification to use multiple layers
            vals['triaPSD'] = [np.around(dVdlnr[flight_num-1,nlayer-1,:], decimals=3)]
        else:
            # using the first flight PSD
            print('Using the PSD from the first flight and first layer')
            vals['triaPSD'] = [np.around(dVdlnr[0,0,:], decimals=3)]
        # vals['triaPSD'] = np.vstack([np.around(dVdlnr[0,0,:], decimals=3),
        #                              np.around(dVdlnr[0,1,:], decimals=3)]) # needs edit
        # parameters above this line has to be modified [AP]
        vals['sph'] = [0.999 + rnd.uniform(-0.99, 0)] # mode 1, 2,...
        vals['vol'] = np.array([0.7326831395]) # gives AOD=4*[0.2165, 0.033499]=1.0
        vals['vrtHght'] = [flight_vrtHght] # mode 1, 2,... # Gaussian mean in meters #HACK: should be 3k
        vals['vrtHghtStd'] = [flight_vrtHghtStd] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = [np.repeat(1.5 + (rnd.uniform(-0.14, 0.15)),
                               nwl)] # mode 1
        # vals['n'] = np.vstack([vals['n'], np.repeat(1.47, nwl)]) # mode 2
        vals['k'] = [np.repeat(0.005 + (rnd.uniform(-0.004,0.004)),
                               nwl)] # mode 1
        # vals['k'] = np.vstack([vals['k'], np.repeat(0.0001, nwl)]) # mode 2
        landPrct = 0 if np.any([x in caseStr.lower() for x in ['vegetation', 'desert']]) else 0
    elif 'marine' in caseStr.lower():
        σ = [0.45, 0.70] # mode 1, 2,...
        rv = [0.2, 0.6]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
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
        if 'nonsph' in caseStr.lower():
            vals['sph'] = [[0.99999], [0.00001]] # mode fine sphere, coarse spheroid
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

def setupConCaseYAML(caseStrs, nowPix, baseYAML, caseLoadFctr=1, caseHeightKM=None, simBldProfs=None): # equal volume weighted marine at 1km & smoke at 4km -> caseStrs='marine+smoke', caseLoadFctr=[1,1], caseHeightKM=[1,4]
    """ nowPix needed to: 1) set land percentage of nowPix and 2) get number of wavelengths """
    aeroKeys = ['traiPSD','lgrnm','sph','vol','vrtHght','vrtHghtStd','vrtProf','n','k']
    vals = dict()
    for caseStr,loading in splitMultipleCases(caseStrs, caseLoadFctr): # loop over all cases and add them together
        valsTmp, landPrct = conCaseDefinitions(caseStr, nowPix)
        for key in valsTmp.keys():
            if key=='vol':
                valsTmp[key] = loading*valsTmp[key]
            elif key=='vrtHght' and caseHeightKM:
                valsTmp[key][:] = caseHeightKM*1000
            if key in aeroKeys and key in vals:
                    vals[key] = np.vstack([vals[key], valsTmp[key]])
            else: # implies we take the surface parameters from the last case
                vals[key] = valsTmp[key]
    nowPix.land_prct = landPrct # [out] – sets nowPix to match ConCase
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
            loadings.append(0.1)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.09*caseLoadFct)
        elif 'case08n' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.1)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.33*caseLoadFct)
        elif 'case08o' in case.lower():
            cases.append(case.replace('case08','smokeDesert'))
            loadings.append(0.1)
            cases.append(case.replace('case08','pollutionDesert'))
            loadings.append(0.7*caseLoadFct)
        elif 'campex' in case.lower():
            cases.append(case.replace('campex','aerosol_campex')) # smoke base τ550=1.0
            loadings.append(0.25*caseLoadFct)
            # cases.append(case.replace('case09a','marine')) # marine base τ550=1.0
            # loadings.append(0.1*caseLoadFct)
        elif 'camp_test' in case.lower():
            cases.append(case.replace('camp_test','smoke')) # smoke base τ550=1.0
            loadings.append(0.25*caseLoadFct)
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
