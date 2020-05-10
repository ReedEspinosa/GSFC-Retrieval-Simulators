#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rnd
import tempfile
from hashlib import md5
import json
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import runGRASP as rg

def conCaseDefinitions(caseStr, nowPix): 
    """ '+' is used to seperate multiple cases (implemented in splitMultipleCases below) """
    vals = dict()
    wvls = np.unique([mv['wl'] for mv in nowPix.measVals])
    nwl = len(wvls)
    """ variable type appended options: 'fine'/'coarse', 'nonsph' and 'lofted' """
    if 'variable' in caseStr.lower(): # dimensions are [mode, λ or (rv,sigma)];
        σ = 0.3+rnd.random()*0.4
        if 'fine' in caseStr.lower():
            rv = 0.12+rnd.random()*0.1 
            vals['vol'] = np.array([[np.random.normal(0.8, 0.2)]])/3 # (currently gives AOD≈1 but changes w/ intensive props.)
        elif 'coarse' in caseStr.lower():
            rv = 0.6+rnd.random()*3
            vals['vol'] = np.array([[np.random.normal(2.5, 0.5)]])/3 # (currently gives AOD≈1 but changes w/ intensive props.)
        else:
            assert False, 'variable aerosol case must be appended with either fine or coarse'
        vals['vol'][vals['vol']<0.001] = 0.001 # just in case random normal drops below zero
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.0001]] if 'nonsph' in caseStr.lower() else [[0.99999]] # mode 1, 2,...
        vals['vrtHght'] = [[3010]] if 'lofted' in caseStr.lower() else  [[1010]]  # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500]] # Gaussian sigma in meters
        vals['n'] = np.interp(wvls, [wvls[0],wvls[-1]],   1.34+rnd.random(2)*0.20)[None,:] # mode 1 # linear w/ λ
        vals['k'] = np.interp(wvls, [wvls[0],wvls[-1]], 0.0001+rnd.random(2)*0.01)[None,:] # mode 1 # linear w/ λ
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 0        
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
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        vals['bpdf'] = [] # first dim mode (N=3), second lambda
        landPrct = 0 if 'ocean' in caseStr.lower() else 100
    elif 'smoke' in caseStr.lower(): # ALL VARIABLES WITH MODES MUST BE 2D (ie. var[mode,wl]) or [] (will not change these values)
        σ = [0.4, 0.45] # mode 1, 2,...
        rv = [0.12, 0.36]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.02737365], [0.00880117]]) # gives AOD = [0.2165, 0.033499]
        vals['vrtHght'] = [[3010],  [3010]] # mode 1, 2,... # Gaussian mean in meters #HACK: should be 3k
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.54, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.47, nwl)]) # mode 2
        vals['k'] = np.repeat(0.04, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.0001, nwl)]) # mode 2
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 0
    elif 'marine' in caseStr.lower():
        σ = [0.45, 0.70] # mode 1, 2,...
        rv = [0.2, 0.6]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.00477583], [0.07941207]]) # gives AOD=[0.0287, 0.0713]
        vals['vrtHght'] = [[1010],  [1010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.415, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.363, nwl)]) # mode 2
        vals['k'] = np.repeat(0.002, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(1e-5, nwl)]) # mode 2
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 0
    elif 'pollution' in caseStr.lower():
        σ = [0.36, 0.64] # mode 1, 2,...
        rv = [0.11, 0.4]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.01787314], [0.00465671]]) # gives AOD=[0.091801 , 0.0082001] but will change if intensive props. change!
        vals['vrtHght'] = [[1010],  [1010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.45, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.5, nwl)]) # mode 2
        vals['k'] = np.repeat(0.001, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.005, nwl)]) # mode 2 # NOTE: we cut this in half from XLSX
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 0
    elif 'dust' in caseStr.lower(): # - Updated to match canonical case spreadsheet V25 -
        σ = [0.5, 0.75] # mode 1, 2,...
        rv = [0.1, 1.10]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        if 'nonsph' in caseStr.lower():
            vals['sph'] = [[0.00001], [0.00001]] # mode 1, 2,...
        else:
            vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.02164019385230769], [0.3166795960377663]]) # gives AOD= [0.13279, 0.11721] but will change if intensive props. change!)
        vals['vrtHght'] = [[3010],  [3010]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.46, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.51, nwl)]) # mode 2
        vals['k'] = np.repeat(1e-8, nwl) # mode 1
        mode2λ = [0.355, 0.380, 0.440, 0.532, 0.550, 0.870, 1.064, 2.100]
        mode2k = [0.0025, 0.0025, 0.0024, 0.0021, 0.0019, 0.0011, 0.0010, 0.0010]
        mode2Intrp = np.interp(wvls, mode2λ, mode2k)
        vals['k'] = np.vstack([vals['k'], mode2Intrp]) # mode 2 # THIS HAS A SPECTRAL DEPENDENCE IN THE SPREADSHEET
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 0
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
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
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
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
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
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 0 if 'case07' in caseStr.lower() else 100
    else:
        assert False, 'No match for caseStr: '+caseStr+'!'
    if 'monomode' in caseStr.lower(): # keep only the large of the two (or more) modes
        bigMode = np.argmax(vals['vol'])
        for key in ['vol','n','k','sph','lgrnm','vrtHght','vrtHghtStd']:
            vals[key] = np.atleast_2d(np.array(vals[key])[bigMode,:]) 
    if not vals['cxMnk'] and landPrct<100: # if not set we will use defualt conical case
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
    if not vals['brdf']  and landPrct>0: # we havn't programed these yet
        λ=[0.415, 0.470, 0.555, 0.659, 0.865, 1.24, 1.64, 2.13] # this should be ordered (interp behavoir is unexpected otherwise)
        if 'desert' in caseStr.lower(): # mean of July 1st 2019 Sahara MIAIC MODIS RTLS (MCD19A3.A2019177.h18v06.006.2019186034811.hdf)
            iso = [0.085, 0.144, 0.236, 0.379, 0.456, 0.569, 0.621, 0.605]
            vol = [0.644, 0.576, 0.462, 0.306, 0.213, 0.167, 0.159, 0.116] # MAIAC_vol/MAIAC_iso
            geo = [0.012, 0.151, 0.083, 0.096, 0.087, 0.103, 0.082, 0.092] # MAIAC_geo/MAIAC_iso
        elif 'vegetation' in caseStr.lower(): # mean of July 1st 2019 SEUS MIAIC MODIS RTLS (MCD19A3.A2019177.h11v05.006.2019186033524.hdf)
            iso = [0.02,  0.04,  0.07, 0.06,  0.42,  0.41,  0.25,  0.11] 
            vol = [0.55, 0.325, 0.61428571, 0.41666667, 0.52619048, 0.48292683, 0.392, 0.24545455] # MAIAC_vol/MAIAC_iso
            geo = [0.1, 0.35, 0.22857143, 0.23333333, 0.15952381, 0.1804878, 0.22, 0.27272727] # MAIAC_geo/MAIAC_iso
        else:
            assert False, 'Land surface type not recognized!'
        lambISO = np.interp(wvls, λ, iso)
        lambVOL = np.interp(wvls, λ, vol)
        lambGEO = np.interp(wvls, λ, geo)
        vals['brdf'] = np.vstack([lambISO, lambVOL, lambGEO])
    if 'bpdf' in vals and not vals['bpdf'] and landPrct>0: # if not set we will use defualt conical case
        if 'desert' in caseStr.lower(): # OSSE original sept. 1st test case over Sahara, BPDFCoef=7.3, NDVI=0.1
            vals['bpdf'] = 6.6*np.ones([1,nwl]) # exp(-VLIDORT_NDVI)*VLIDORT_C)
        elif 'vegetation' in caseStr.lower(): # OSSE original sept. 1st test case over SEUS, BPDFCoef=6.9, NDVI=0.9
            vals['bpdf'] = 2.8*np.ones([1,nwl]) # exp(-VLIDORT_NDVI)*VLIDORT_C)
        else:
            assert False, 'Land surface type not recognized!'
    lidarMeasLogical = np.isclose(34.5, [mv['meas_type'][0] for mv in nowPix.measVals], atol=5) # measurement types 30-39 reserved for lidar; if first meas_type is LIDAR, they all should be 
    if lidarMeasLogical.any(): 
        lidarInd = lidarMeasLogical.nonzero()[0][0]
        hValTrgt = np.array(nowPix.measVals[lidarInd]['thetav'][0:nowPix.measVals[lidarInd]['nbvm'][0]]) # HINT: this assumes all LIDAR measurement types have the same vertical range values
        vals['vrtProf'] = np.empty([len(vals['vrtHght']), len(hValTrgt)])
        for i, (mid, rng) in enumerate(zip(vals['vrtHght'], vals['vrtHghtStd'])):
            bot = max(mid[0]-2*rng[0],115) # we want to bottom two bins to go to zero (GRASP bug)
            top = mid[0]+2*rng[0]
            vals['vrtProf'][i,:] = np.logical_and(np.array(hValTrgt) > bot, np.array(hValTrgt) <= top)*1+0.000001
            vals['vrtProf'][i,2:] = np.convolve(vals['vrtProf'][i,:], np.ones(2)/2, mode='full')[3:] # smooth it, preserving zeros at ends and high concentration at bottom
        del vals['vrtHght']
        del vals['vrtHghtStd']
    return vals, landPrct

def splitMultipleCases(caseStrs, caseLoadFct):
    if caseLoadFct is None: caseLoadFct = 1
    cases = []
    loadings = []
    for case in caseStrs.split('+'):
        if 'case06a' in case.lower():
            cases.append(case.replace('case06a','smoke'))
            loadings.append(caseLoadFct)
            cases.append(case.replace('case06a','marine'))
            loadings.append(caseLoadFct)
        elif 'case06b' in case.lower():
            cases.append(case.replace('case06b','smoke'))
            loadings.append(0.4*caseLoadFct)
            cases.append(case.replace('case06b','marine'))
            loadings.append(2.5*caseLoadFct)
        elif 'case06c' in case.lower():
            cases.append(case.replace('case06c','smoke'))
            loadings.append(caseLoadFct)
            cases.append(case.replace('case06c','pollution'))
            loadings.append(caseLoadFct)
        elif 'case06d' in case.lower():
            cases.append(case.replace('case06d','smoke'))
            loadings.append(0.4*caseLoadFct)
            cases.append(case.replace('case06d','pollution'))
            loadings.append(0.4*caseLoadFct)
        elif 'case06e' in case.lower():
            cases.append(case.replace('case06e','dust'))
            loadings.append(caseLoadFct)
            cases.append(case.replace('case06e','marine'))
            loadings.append(caseLoadFct)
        elif 'case06f' in case.lower():
            cases.append(case.replace('case06f','dust'))
            loadings.append(0.4*caseLoadFct)
            cases.append(case.replace('case06f','marine'))
            loadings.append(2.5*caseLoadFct)
        else:
            cases.append(case)
            loadings.append(caseLoadFct)
        # print(cases)
    return zip(cases, loadings)

def setupConCaseYAML(caseStrs, nowPix, baseYAML, caseLoadFctr=None, caseHeightKM=None): # equal volume weighted marine at 1km & smoke at 4km -> caseStrs='marine+smoke', caseLoadFctr=[1,1], caseHeightKM=[1,4]
    fldNms = {
        'lgrnm':'size_distribution_lognormal',
        'sph':'sphere_fraction',
        'vol':'aerosol_concentration',
        'vrtHght':'vertical_profile_parameter_height',
        'vrtHghtStd':'vertical_profile_parameter_standard_deviation',
        'vrtProf':'vertical_profile_normalized',
        'n':'real_part_of_refractive_index_spectral_dependent',
        'k':'imaginary_part_of_refractive_index_spectral_dependent',
        'brdf':'surface_land_brdf_ross_li',
        'bpdf':'surface_land_polarized_maignan_breon',
        'cxMnk':'surface_water_cox_munk_iso'}
    aeroKeys = ['lgrnm','sph','vol','vrtHght','vrtHghtStd','vrtProf','n','k']
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
    bsHsh = md5(open(baseYAML,'rb').read()).hexdigest()[0:8] # unique ID from contents of original user created YAML file
    valList = json.dumps([(y.tolist() if 'numpy' in str(type(y)) else y) for y in vals.values()])
    valHsh = md5(json.dumps(valList).encode()).hexdigest()[0:8] # unique ID from this case's values
    nwl = len(np.unique([mv['wl'] for mv in nowPix.measVals]))
    newFn = 'settingsYAML_conCase%s_nwl%d_%s_%s.yml' % (caseStrs, nwl, bsHsh, valHsh)
    newPathYAML = os.path.join(tempfile.gettempdir(), newFn)
    if os.path.exists(newPathYAML): return newPathYAML, landPrct # reuse existing YAML file from this exact base YAML, con. case values and NWL
    yamlObj = rg.graspYAML(baseYAML, newPathYAML)
    yamlObj.adjustLambda(nwl)
    for key in vals.keys():
        for m in range(np.array(vals[key]).shape[0]): # loop over aerosol modes
                fldNm = '%s.%d.value' % (fldNms[key], m+1)
                yamlObj.access(fldNm, newVal=vals[key][m], write2disk=False, verbose=False) # verbose=False -> no wanrnings about creating a new mode
                if key=='vrtProf':
                    fldNm = '%s.%d.index_of_wavelength_involved' % (fldNms[key], m+1)
                    yamlObj.access(fldNm, newVal=np.zeros(len(vals[key][m]), dtype=int), write2disk=False) 
                    fldNm = '%s.%d.min' % (fldNms[key], m+1)
                    yamlObj.access(fldNm, newVal=1e-9*np.ones(len(vals[key][m])), write2disk=False)
                    fldNm = '%s.%d.max' % (fldNms[key], m+1)
                    yamlObj.access(fldNm, newVal=2*np.ones(len(vals[key][m])), write2disk=False)
    yamlObj.access('stop_before_performing_retrieval', True, write2disk=True)    
    return newPathYAML, landPrct 