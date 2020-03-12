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

from MADCAP_functions import loadVARSnetCDF
import os
import re
import numpy as np


typeKeys = ['DU', 'SU', 'BC', 'OC', 'SS']
almTag = 'alm'
hybTag = 'hyb'
primeTag = 'Level2'
secondTag = 'Level1_5'
thrshLen = 10 # if primeTag has fewer than this many cases we use secondTag instead
wvlsOut = [0.340, 0.355, 0.360, 0.380, 0.410, 0.532, 0.550, 0.670, 0.870, 0.910, 1.064, 1.230, 1.550, 1.650, 1.880, 2.130, 2.250]
rhTrgt = 0.75 # target RH for lut (fraction)
baseDir = '/gpfsm/dnb32/okemppin/purecases-nc/'
baseFN = 'puredata-combi-%s-%s-%s-0.70pureness-clean2.nc4'
basePathAERO = os.path.join(baseDir, baseFN)
baseDir = '/gpfsm/dnb32/okemppin/LUT-GSFun/'
basePathLUT = {
        'DU':os.path.join(baseDir, 'GRASP_LUT-DUST_V4.GSFun.nc'),
        'SU':os.path.join(baseDir, 'optics_SU.v5_7.GSFun.nc'),
        'BC':os.path.join(baseDir, 'optics_BC.v5_7.GSFun.nc'),
        'OC':os.path.join(baseDir, 'optics_OC.v12_7.GSFun.nc'),
        'SS':os.path.join(baseDir, 'optics_SS.v3_7.GSFun.nc')}

def main():
    k = 0
    curTag = primeTag
    while k < len(typeKeys): # loop over species
        rslt = parseFile(basePathAERO % (curTag, almTag, typeKeys[k])) # read almacanter
        rslt = np.r_[rslt, parseFile(basePathAERO % (curTag, hybTag, typeKeys[k]))] # read hybrid
        if len(rslt['day']) < thrshLen and curTag==primeTag: # prime tag had too few cases
            curTag = secondTag
        elif len(rslt['day']) >= thrshLen: # we will work with and then write this data 
            outFilePath = 'XXX'
            writeNewData(rslt, outFilePath)
            curTag = primeTag
            k = k+1
        else:
            assert False, 'No cases were found with tags %s or %s' % (primeTag, secondTag)

def parseFile(filePath):
    data = loadVARSnetCDF(filePath)
    findRefInd(rslt, typeKey, 'Refractive_Index_Real', 'refreal')
    
    for k,v in data.items():
        print('%s %d' % (k,v))

def findRefInd(rslt, typeKey, aeroNC4name, lutNC4name):
    LUTvars = ['lambda', 'rh', lutNC4name]
    LUT = loadVARSnetCDF(basePathLUT[typeKey], LUTvars)
    rhInd = np.argmin(np.abs(LUT['rh'] - rhTrgt))
    LUTri = LUT[lutNC4name][:,rhInd,:].mean(axis=0) # we average over all size modes
    λaeroStr = [y for y in rslt.keys() if aeroNC4name in y] 
    λaero = np.sort([int(re.search(r'\d+$', y).group()) for y in λaeroStr]) # nm
    RIaero = np.array([rslt['%s_%d' % (aeroNC4name,λ)] for λ in λaero]).T # RI[t,λ]
    for t,ri in enumerate(RIaero):
        LUTlowScl = ri[0]/np.interp(λaero[0], LUT['lambda'], LUTri)
        LUTupScl = ri[-1]/np.interp(λaero[-1], LUT['lambda'], LUTri)
        lowI = LUT['lambda'] < λaero[0]
        upI = LUT['lambda'] > λaero[-1]
        LUTlow = ri[lowI]*LUTlowScl
        LUTup = ri[upI]*LUTupScl
        RIfull = np.r_[LUTlow, RIaero, LUTup]
        λfull = np.r_[LUT['lambda'][lowI], λaero, LUT['lambda'][lowI]]
        # now we interp λfull, RIfull TO above wavelengths
        
def writeNewData():
    assert False, "Not built"

if __name__ == '__main__': main()





