#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tempfile
from hashlib import md5
import os
import sys
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import runGRASP as rg

def conCaseDefinitions(caseStr, wvls): 
    vals = dict()
    nwl = len(wvls)
    if caseStr.lower()=='smoke': # ALL VARIABLES WITH MODES MUST BE 2D (ie. var[mode,wl]) or [] (will not change these values)
        σ = [0.37, 0.4] # mode 1, 2,...
        rv = [0.12, 0.35]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.76],  [0.24]])/7.1508 # (currently gives AOD=1 but will change if intensive props. change!)
        vals['vrtHght'] = [[3000],  [3000]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.55, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.55, nwl)]) # mode 2
        vals['k'] = np.repeat(0.04, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.0001, nwl)]) # mode 2
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 100
    elif caseStr.lower()=='marine':
        σ = [0.5, 0.72] # mode 1, 2,...
        rv = [0.2, 0.55]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.05],  [0.95]])/1.1464 # (currently gives AOD=1 but will change if intensive props. change!)
        vals['vrtHght'] = [[1000],  [1000]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.415, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.363, nwl)]) # mode 2
        vals['k'] = np.repeat(0.002, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(3e-9, nwl)]) # mode 2
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 100
    elif caseStr.lower()=='pollution':
        σ = [0.36, 0.64] # mode 1, 2,...
        rv = [0.11, 0.4]*np.exp(3*np.power(σ,2)) # mode 1, 2,... (rv = rn*e^3σ)
        vals['lgrnm'] = np.vstack([rv, σ]).T
        vals['sph'] = [[0.99999], [0.99999]] # mode 1, 2,...
        vals['vol'] = np.array([[0.75],  [0.25]])/4.2924 # (currently gives AOD=1 but will change if intensive props. change!)
        vals['vrtHght'] = [[1000],  [1000]] # mode 1, 2,... # Gaussian mean in meters
        vals['vrtHghtStd'] = [[500],  [500]] # mode 1, 2,... # Gaussian sigma in meters
        vals['n'] = np.repeat(1.45, nwl) # mode 1 
        vals['n'] = np.vstack([vals['n'], np.repeat(1.5, nwl)]) # mode 2
        vals['k'] = np.repeat(0.001, nwl) # mode 1
        vals['k'] = np.vstack([vals['k'], np.repeat(0.01, nwl)]) # mode 2
        vals['brdf'] = [] # first dim mode (N=3), second lambda
        vals['cxMnk'] = [] # first dim mode (N=3), second lambda
        landPrct = 100
    else:
        assert False, 'No match for canonical case type!'
    if not vals['brdf']: # if not set we will use defualt conical case
        λ=[0.355, 0.380, 0.440, 0.532, 0.550, 0.870, 1.064, 2.100]
        R=[0.0046195003, 0.0050949964, 0.0060459884, 0.0024910956,	0.0016951599, 0.0000000000, 0.0000000000, 0.0000000000] # TODO: need to double check these units
        lambR = np.interp(wvls, λ, R)
        FresFrac = 0.9999*np.ones(nwl)
        cxMnk = (7*0.00512+0.003)/2*np.ones(nwl)
        vals['brdf'] = np.vstack([lambR, FresFrac, cxMnk])
    return vals, landPrct

def setupConCaseYAML(caseStrs, wvls, baseYAML, caseLoadFctr=None, caseHeightKM=None): # equal volume weighted marine at 1km & smoke at 4km -> caseStrs='marine+smoke', caseLoadFctr=[1,1], caseHeightKM=[1,4]
    fldNms = {
        'lgrnm':'size_distribution_lognormal',
        'sph':'sphere_fraction',
        'vol':'aerosol_concentration',
        'vrtHght':'vertical_profile_parameter_height',
        'vrtHghtStd':'vertical_profile_parameter_standard_deviation',
        'n':'real_part_of_refractive_index_spectral_dependent',
        'k':'imaginary_part_of_refractive_index_spectral_dependent',
        'brdf':'surface_land_brdf_ross_li',
        'cxMnk':'surface_water_cox_munk_iso'}
    aeroKeys = ['lgrnm','sph','vol','vrtHght','vrtHghtStd','n','k']
    vals = dict()
    for i, caseStr in enumerate(caseStrs.split('+')): # loop over all cases and add them together
        valsTmp, landPrct = conCaseDefinitions(caseStr, wvls)
        for key in valsTmp.keys():
            if key=='vol' and caseLoadFctr:
                valsTmp[key] = caseLoadFctr[i]*valsTmp[key]
            elif key=='vrtHght' and caseHeightKM:
                valsTmp[key][:] = caseHeightKM[i]*1000
            if key in aeroKeys and key in vals:
                    vals[key] = np.vstack([vals[key], valsTmp[key]])
            else: # implies we take the surface parameters from the last case
                vals[key] = valsTmp[key]
    bsHsh = md5(baseYAML.encode()).hexdigest()[0:8]
    nwl = len(wvls)
    newFn = 'settingsYAML_conCase%s_nwl%d_%s.yml' % (caseStrs, nwl, bsHsh)
    newPathYAML = os.path.join(tempfile.gettempdir(), newFn)
    yamlObj = rg.graspYAML(baseYAML, newPathYAML)
    yamlObj.adjustLambda(nwl)
    for key in vals.keys():
        for m in range(np.array(vals[key]).shape[0]): # loop over aerosol modes
                fldNm = '%s.%d.value' % (fldNms[key], m+1)
                yamlObj.access(fldNm, newVal=vals[key][m], write2disk=False, verbose=False) # verbose=False -> no wanrnings about creating a new mode
    yamlObj.access('stop_before_performing_retrieval', True, write2disk=True)    
    return newPathYAML, landPrct 

