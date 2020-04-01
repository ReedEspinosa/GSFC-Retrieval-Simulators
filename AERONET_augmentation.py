#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example MERRA-2 profile used by AERONET retrieval:
discover:/gpfsm/dnb32/dgiles/AERONET/SAMPLER/DATA/Y2019/merra2_aeronet_aop_ext500nm.20190101_00-20190131_21.nc4
AERONET summary files:
Discover:/gpfsm/dnb32/okemppin/purecases-nc/
-rwxr-xr-x 1 okemppin s1043    16392 Feb 17 19:02 puredata-combi-Level2-alm-SS-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043    16392 Feb 17 19:02 puredata-combi-Level2-hyb-SS-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043   992584 Feb 17 19:02 puredata-combi-Level2-alm-SU-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043   189768 Feb 17 19:02 puredata-combi-Level2-hyb-SU-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043   648520 Feb 17 19:02 puredata-combi-Level2-alm-OC-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043   304456 Feb 17 19:02 puredata-combi-Level2-hyb-OC-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043  2827592 Feb 17 19:02 puredata-combi-Level2-alm-DU-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043   189768 Feb 17 19:02 puredata-combi-Level2-hyb-DU-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043    16392 Feb 17 19:02 puredata-combi-Level2-alm-BC-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043    16392 Feb 17 19:02 puredata-combi-Level2-hyb-BC-0.70pureness-clean2.nc4
-rwxr-xr-x 1 okemppin s1043   196244 Feb 17 17:52 puredata-combi-Level1_5-hyb-SS-0.70pureness-clean2.nc4
...
Lookup tables with RI shape:
DISCOVER:/gpfsm/dnb32/okemppin/LUT-GSFun/

"""

import os
import sys
import re
from netCDF4 import Dataset
import numpy as np
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
from miscFunctions import checkDiscover
from MADCAP_functions import loadVARSnetCDF

typeKeys = ['DU', 'SU', 'BC', 'OC', 'SS']
almTag = 'alm'
hybTag = 'hyb'
primeTag = 'Level2'
secondTag = 'Level1_5'
thrshLen = 10 # if primeTag has fewer than this many cases we use secondTag instead
wvlsOut = [340, 355, 360, 380, 410, 532, 550, 670, 870, 910, 1064, 1230, 1550, 1650, 1880, 2130, 2250] # nm
rhTrgt = 0.75 # target RH for LUT (fraction)
if checkDiscover(): # DISCOVER
    baseDirCases = '/gpfsm/dnb32/okemppin/purecases-nc/'
    baseDirLUT = '/gpfsm/dnb32/okemppin/LUT-GSFun/'
else: # probably Reed's MacBook Air
    baseDirCases = '/Users/wrespino/Downloads/'
    baseDirLUT = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/' # only has dust LUT
baseFN = 'puredata-combi-%s-%s-%s-0.70pureness-clean2.nc4'
basePathAERO = os.path.join(baseDirCases, baseFN)
basePathLUT = {
        'DU':os.path.join(baseDirLUT, 'optics_DU.v15_6.nc'),
        'SU':os.path.join(baseDirLUT, 'optics_SU.v5_7.GSFun.nc'),
        'BC':os.path.join(baseDirLUT, 'optics_BC.v5_7.GSFun.nc'),
        'OC':os.path.join(baseDirLUT, 'optics_OC.v12_7.GSFun.nc'),
        'SS':os.path.join(baseDirLUT, 'optics_SS.v3_7.GSFun.nc')}

def main():
    k = 0
    curTag = primeTag
    while k < len(typeKeys): # loop over species (also, we make second pass if lev2 is not at thrshLen)
        rslt = parseFile(basePathAERO % (curTag, almTag, typeKeys[k]), typeKeys[k]) # read almacanter
        rslt = np.r_[rslt, parseFile(basePathAERO % (curTag, hybTag, typeKeys[k]), typeKeys[k])] # read hybrid
        if len(rslt['day']) < thrshLen and curTag==primeTag: # lev2.0 had too few cases: loop again, loading 1.5
            curTag = secondTag
        elif len(rslt['day']) >= thrshLen: # we will work with and then write this data 
            outFilePath = basePathAERO % (curTag, 'allScans', typeKeys[k])
            with Dataset(outFilePath, 'w', format='NETCDF4') as root_grp:
                writeNewData(rslt, root_grp)
            curTag = primeTag
            k = k+1
        else:
            assert False, 'No cases were found with tags %s or %s' % (primeTag, secondTag)

def parseFile(filePath, typeKey):
    rslt = loadVARSnetCDF(filePath)
    rslt['rri'] = findRefInd(rslt, typeKey, 'Refractive_Index_Real', 'refreal')
    rslt['iri'] = findRefInd(rslt, typeKey, 'Refractive_Index_Imag', 'refimag')
    rslt['ssa'] = findRefInd(rslt, typeKey, 'Single_Scatting_Albedo', 'ssa')
    return rslt
    
def findRefInd(rslt, typeKey, aeroNC4name, lutNC4name):
    """ typeKey links LUT file, aeroNC4name+_XXX is aeronet dict key, lutNC4name is LUT key 
        Note this function is likely to occasionally return RI outside the bounds of GRASP's LUT """
    LUTvars = ['bsca','bext'] if lutNC4name=='ssa' else [lutNC4name]
    LUT = loadVARSnetCDF(basePathLUT[typeKey], ['lambda', 'rh'] + LUTvars)
    rhInd = np.argmin(np.abs(LUT['rh'] - rhTrgt))
    if lutNC4name=='ssa': LUT['ssa'] = LUT['bsca']/LUT['bext'] # TODO: Why does this differ from qsca/qext? See bottom of 
    LUTri = LUT[lutNC4name][:,rhInd,:].mean(axis=0) # we average over all size modes
    LUTλ = LUT['lambda']*1e9 # m -> nm
    λaeroStr = [y for y in rslt.keys() if aeroNC4name in y] 
    λaero = np.sort([int(re.search(r'\d+$', y).group()) for y in λaeroStr]) # nm
    RIaero = np.array([rslt['%s_%d' % (aeroNC4name,λ)] for λ in λaero]).T # RI[t,λ]
    RI = np.zeros([len(RIaero), len(wvlsOut)])
    for t,ri in enumerate(RIaero): # loop over all aeronet retrievals
        LUTlowScl = ri[0]/np.interp(λaero[0], LUTλ, LUTri)
        LUTupScl = ri[-1]/np.interp(λaero[-1], LUTλ, LUTri)
        lowI = LUTλ < λaero[0]
        upI = LUTλ > λaero[-1]
        LUTlow = LUTri[lowI]*LUTlowScl
        LUTup = LUTri[upI]*LUTupScl
        RIfull = np.r_[LUTlow, ri, LUTup]
        λfull = np.r_[LUTλ[lowI], λaero, LUTλ[upI]]
        RI[t,:] = np.interp(wvlsOut, λfull, RIfull)
    return RI
        
def writeNewData(rslt, root_grp):
    root_grp.description = 'Collection of AERONET/MERRA2 state parameters'
    # dimensions
    λVar = 'lambda'
    root_grp.createDimension(λVar, len(wvlsOut))
    indVar = 'index'
    root_grp.createDimension(indVar, len(rslt['day']))
    # create and write pass through variables
    forbdnStrngs = ['Refractive_Index','Single_Scattering'] # if a key has this string anywhere it will be thrown out
    keepKeys = [s for s in rslt.keys() if np.all([s1 not in s for s1 in forbdnStrngs])]
    keepKeys = [s.replace('Extinction_Angstrom_Exponent_440_870_Total', 'AE_T') for s in keepKeys] # default variable name is too long
    newKeys = ['rri', 'iri', 'ssa']
    nc4Vars = dict()
    for key in keepKeys+newKeys:
        if key in newKeys: # spectrally dependent variable
            nc4Vars[key] = root_grp.createVariable(key, 'f4', (indVar, λVar))
            nc4Vars[key][:,:] = rslt[key]            
        else: # spectrally invariant variable
            nc4Vars[key] = root_grp.createVariable(key, 'f4', (indVar))
            nc4Vars[key][:] = rslt[key]    
        nc4Vars[key].units = getUnits(key)
        nc4Vars[key].long_name = getFullName(key)

def getFullName(key):
    modeStr = {'F':'fine mode', 'C':'coarse mode', 'T':'total'}
    riStr = {'r':'Real', 'i':'Imaginary'}
    if 'VolC' in key:
        return '%s volume concentration' % modeStr[key[-1]]
    if 'VMR' in key:
        return '%s volume median radius' % modeStr[key[-1]]
    if 'Std' in key:
        return '%s standard deviation' % modeStr[key[-1]]
    if key=='hour':
        return 'UTC '+key
    if key in ['day', 'month']:
        return key+' of year'
    if key=='pureness':
        return 'MERRA2 derived pureness of target species'
    if key=='AERONET_Site_id':
        return 'Site ID of AERONET station where data was retrieved'
    if key=='AE_T':
        return 'Angstrom Exponent derived from 440 and 870nm total AOD'
    if key in ['rri','iri']:
        return '%s Refractive Index' % riStr[key[0]]
    if key=='ssa':
        return 'Single Scattering Albedo'
    return key
        
def getUnits(key):
    if 'VolC' in key:
        return 'μm^3/μm^2'
    if 'VMR' in key:
        return 'μm'
    if 'pureness' in key:
        return 'mass fraction in range 0-1'
    return 'none'
    
if __name__ == '__main__': main()





