#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Print RTLS kernel averages from a MAIAC tile  """

import numpy as np
import sys
from pyhdf.SD import SD, SDC
from sklearn.decomposition.pca import PCA

vegFile = '/discover/nobackup/wrespino/synced/Working/MCD19A3.A2019177.h11v05.006.2019186033524.hdf' # vegitated surface - tile h11,v05 maps primarly to SEUS; day 177 is June 25th
dsrtFile = '/discover/nobackup/wrespino/synced/Working/MCD19A3.A2019177.h18v06.006.2019186034811.hdf' # desert surface - tile h18,v06 maps to central Saharah; day 177 is June 25th
λout = np.sort([0.355, 0.532, 1.064, 0.360,0.380,0.410,0.550,0.670,0.870,1.550,1.650, 0.645, 0.8585, 0.469, 0.555, 1.24, 1.64, 2.13, 0.412]) # [μm]
fltrPrct = 98 # [%] middle percentile of data to take (i.e. 98 -> 1%-99%)
normByIso = True # Normalize VOL and GEO by the isotropic term

def main():
    print('\n*****Vegie Surface*****\n')
    MAIAC_BRDF_stats(vegFile, normByIso, fltrPrct, λout)
    x,Nλ = MAIAC_BRDF_stats(vegFile, normByIso, fltrPrct, None, printResult=False)
    PCAonMAIAC(x, Nλ)
    print('\n\n\n*****Desert Surface*****\n')
    MAIAC_BRDF_stats(dsrtFile, normByIso, fltrPrct, λout)
    x,Nλ = MAIAC_BRDF_stats(dsrtFile, normByIso, fltrPrct, None, printResult=False)
    PCAonMAIAC(x, Nλ)
    
def MAIAC_BRDF_stats(fileName, normByIso=False, fltrPrct=100, λoutArg=None, printResult=True):
    print('--- Call to MAIAC_BRDF_stats <> V4 ---')
    λHDF = np.r_[0.645, 0.8585, 0.469, 0.555, 1.24, 1.64, 2.13, 0.412]
    λOrder = λHDF.argsort()
    λHDF_ord = λHDF[λOrder]
    λout = λHDF_ord if λoutArg is None else np.sort(λoutArg)
    hdf = SD(fileName, SDC.READ)
    Nλ = len(λHDF_ord)
    Nλout = len(λout)
    vals = dict()
    badInd = []
    keys = ['Kiso', 'Kvol', 'Kgeo']
    for key in keys:
        hdfStrct = hdf.select(key)
        vals[key] = hdfStrct[:,:,:].reshape([Nλ,-1])
        badInd.append(np.any(vals[key]==hdfStrct.attributes()['_FillValue'], axis=0))
        vals[key] = vals[key][λOrder,:]*hdfStrct.attributes()['scale_factor'] # sort MAIAC data by λ
        if normByIso and key is not 'Kiso':
            with np.errstate(divide='ignore'): # Kiso may be zero
                with np.errstate(invalid='ignore'): # apparently numpy gives two warnings for divide by zero? 
                    vals[key] = vals[key]/vals['Kiso'] 
    badInds = np.any(badInd, axis=0)
    for key in keys: vals[key] = vals[key][:, ~badInds]
    if printResult: print('%d/%d remaining pixels had atleast one fill value' % (np.sum(badInds), len(badInds)))
    if fltrPrct < 100:
        badInd = []
        for key in keys:
            upBnds = np.atleast_2d(np.percentile(vals[key], (100+fltrPrct)/2, axis=1)).T
            badInd.append(np.any(vals[key] > upBnds, axis=0))
            lowBnds = np.atleast_2d(np.percentile(vals[key], (100-fltrPrct)/2, axis=1)).T
            badInd.append(np.any(vals[key] < lowBnds, axis=0))
        badInds = np.any(badInd, axis=0)
        for key in keys: vals[key] = vals[key][:, ~badInds]
        if printResult: print('%d/%d pixels had atleast one outlier' % (np.sum(badInds), len(badInds)))
    if λoutArg is None: # we output at original MAIAC λ
        valsOut = vals    
    else: # we need to interpolate to new λ
        shiftWave = lambda a: np.array([np.interp(λout, λHDF_ord, b) for b in a.T]).T
        valsOut = dict()
        for key, val in vals.items(): valsOut[key] = shiftWave(val)
    if printResult:
        print('λ[μm], '+", ".join('%s[mean],  %s[std]' % (s,s) for s in keys))
        for λ, λval in enumerate(λout):
            print('%5.3f,' % λval, end='')
            print(",".join('%11.4f,%11.4f' % (c[λ,:].mean(),c[λ,:].std()) for c in valsOut.values())) 
        print('-----')
        print('mean over pixel & λ (%s):     ' % ",".join(str(s) for s in keys), end='')
        print(",".join('%9.4f' % y.mean() for y in valsOut.values())) # valsOut -> mean weighted by output λ, not MODIS λ
        print('std. dev. over pixel & λ (%s):' % ",".join(str(s) for s in keys), end='')        
        print(",".join('%9.4f' % y.std() for y in valsOut.values())) # valsOut -> mean weighted by output λ, not MODIS λ
        print('-----')
        rdInd = np.argmin(np.abs(λHDF_ord - 0.65))
        irInd = np.argmin(np.abs(λHDF_ord - 0.8))
        irrdλvals = (λHDF_ord[irInd]*1000, λHDF_ord[rdInd]*1000)*2
        print('NDVI = (Kiso[%dnm]-Kiso[%dnm])/(Kiso[%dnm]+Kiso[%dnm])' % irrdλvals)
        NDVIs = (vals['Kiso'][irInd,:]-vals['Kiso'][rdInd,:])/(vals['Kiso'][irInd,:]+vals['Kiso'][rdInd,:]) # vals -> work on MODIS λ, interpolated data
        print('NDVI mean:      %9.4f' % NDVIs.mean())
        print('NDVI std. dev.: %9.4f' % NDVIs.std())
        print('-----')
    x = np.vstack([y for y in valsOut.values()])
    return x, Nλout

def PCAonMAIAC(x, Nλout):
    pca = PCA(n_components=3*Nλout)
    xStnrd = (x-x.mean(axis=1)[:,None])/x.std(axis=1)[:,None] # standardize x
    pca.fit(xStnrd.T)
    varExpln = np.sum(pca.explained_variance_ratio_[0:5]*100)
    print('Percent of variance explained by first five principle components %5.2f%%' % varExpln) 
    varExpln = np.sum(pca.explained_variance_ratio_[0:(Nλout+2)]*100)
    print('Percent of variance explained by first Nλ+2=%d principle components %5.2f%%' % (Nλout+2, varExpln)) 
    
if __name__ == "__main__": main()


"""
APR 23, 2020
------------
SETTINGS:
vegFile = '/discover/nobackup/wrespino/synced/Working/MCD19A3.A2019177.h11v05.006.2019186033524.hdf' # vegitated surface - tile h11,v05 maps primarly to SEUS; day 177 is June 25th
dsrtFile = '/discover/nobackup/wrespino/synced/Working/MCD19A3.A2019177.h18v06.006.2019186034811.hdf' # desert surface - tile h18,v06 maps to central Saharah; day 177 is June 25th
λout = np.sort([0.355, 0.532, 1.064, 0.360,0.380,0.410,0.550,0.670,0.870,1.550,1.650, 0.645, 0.8585, 0.469, 0.555, 1.24, 1.64, 2.13, 0.412]) # [μm]
fltrPrct = 98 # [%] middle percentile of data to take (i.e. 98 -> 1%-99%)
normByIso = True # Normalize VOL and GEO by the isotropic term


OUTPUT:
wrespino@discover34:MADCAP_scripts> python printRTLS.py

*****Vegie Surface*****

--- Call to MAIAC_BRDF_stats <> V4 ---
579261/1440000 remaining pixels had atleast one fill value
170783/860739 pixels had atleast one outlier
λ[μm], Kiso[mean],  Kiso[std], Kvol[mean],  Kvol[std], Kgeo[mean],  Kgeo[std]
0.355,     0.0237,     0.0114,     0.7342,     0.9639,     0.0098,     0.4210
0.360,     0.0237,     0.0114,     0.7342,     0.9639,     0.0098,     0.4210
0.380,     0.0237,     0.0114,     0.7342,     0.9639,     0.0098,     0.4210
0.410,     0.0237,     0.0114,     0.7342,     0.9639,     0.0098,     0.4210
0.412,     0.0237,     0.0114,     0.7342,     0.9639,     0.0098,     0.4210
0.469,     0.0368,     0.0150,     0.3829,     0.5665,     0.3476,     0.1010
0.532,     0.0644,     0.0159,     0.5536,     0.4197,     0.2833,     0.0814
0.550,     0.0723,     0.0164,     0.6023,     0.3985,     0.2649,     0.0794
0.555,     0.0745,     0.0165,     0.6159,     0.3947,     0.2598,     0.0792
0.645,     0.0560,     0.0245,     0.5980,     0.5899,     0.2563,     0.1027
0.670,     0.0989,     0.0191,     0.5936,     0.5213,     0.2444,     0.0922
0.859,     0.4225,     0.0603,     0.5609,     0.1826,     0.1545,     0.0498
0.870,     0.4222,     0.0595,     0.5595,     0.1799,     0.1552,     0.0492
1.064,     0.4160,     0.0472,     0.5349,     0.1563,     0.1668,     0.0419
1.240,     0.4104,     0.0399,     0.5127,     0.1762,     0.1773,     0.0413
1.550,     0.2828,     0.0343,     0.4588,     0.2417,     0.2120,     0.0546
1.640,     0.2457,     0.0393,     0.4432,     0.2823,     0.2221,     0.0623
1.650,     0.2430,     0.0392,     0.4414,     0.2836,     0.2232,     0.0625
2.130,     0.1128,     0.0393,     0.3531,     0.4821,     0.2735,     0.0945
-----
mean over pixel & λ (Kiso,Kvol,Kgeo):        0.1619,   0.5727,   0.1731
std. dev. over pixel & λ (Kiso,Kvol,Kgeo):   0.1562,   0.6035,   0.2496
-----
NDVI = (Kiso[858nm]-Kiso[645nm])/(Kiso[858nm]+Kiso[645nm])
NDVI mean:         0.7617
NDVI std. dev.:    0.1091
-----
--- Call to MAIAC_BRDF_stats <> V4 ---
Percent of variance explained by first five principle components 77.79%
Percent of variance explained by first Nλ+2=10 principle components 92.19%



*****Desert Surface*****

--- Call to MAIAC_BRDF_stats <> V4 ---
0/1440000 remaining pixels had atleast one fill value
230082/1440000 pixels had atleast one outlier
λ[μm], Kiso[mean],  Kiso[std], Kvol[mean],  Kvol[std], Kgeo[mean],  Kgeo[std]
0.355,     0.0859,     0.0201,     0.6729,     0.3634,    -0.0741,     0.1692
0.360,     0.0859,     0.0201,     0.6729,     0.3634,    -0.0741,     0.1692
0.380,     0.0859,     0.0201,     0.6729,     0.3634,    -0.0741,     0.1692
0.410,     0.0859,     0.0201,     0.6729,     0.3634,    -0.0741,     0.1692
0.412,     0.0859,     0.0201,     0.6729,     0.3634,    -0.0741,     0.1692
0.469,     0.1453,     0.0243,     0.5822,     0.1466,     0.1474,     0.0580
0.532,     0.2143,     0.0332,     0.4969,     0.1343,     0.0989,     0.0601
0.550,     0.2340,     0.0364,     0.4726,     0.1342,     0.0850,     0.0611
0.555,     0.2394,     0.0373,     0.4658,     0.1344,     0.0811,     0.0614
0.645,     0.3838,     0.0593,     0.3087,     0.1294,     0.0959,     0.0568
0.670,     0.3929,     0.0606,     0.2978,     0.1258,     0.0948,     0.0562
0.859,     0.4619,     0.0722,     0.2161,     0.1187,     0.0860,     0.0533
0.870,     0.4653,     0.0723,     0.2147,     0.1180,     0.0864,     0.0530
1.064,     0.5234,     0.0751,     0.1914,     0.1143,     0.0937,     0.0493
1.240,     0.5762,     0.0798,     0.1703,     0.1244,     0.1003,     0.0476
1.550,     0.6166,     0.0755,     0.1626,     0.1144,     0.0848,     0.0459
1.640,     0.6283,     0.0746,     0.1604,     0.1155,     0.0803,     0.0459
1.650,     0.6280,     0.0744,     0.1594,     0.1153,     0.0805,     0.0459
2.130,     0.6126,     0.0713,     0.1133,     0.1198,     0.0909,     0.0475
-----
mean over pixel & λ (Kiso,Kvol,Kgeo):        0.3448,   0.3883,   0.0493
std. dev. over pixel & λ (Kiso,Kvol,Kgeo):   0.2173,   0.3024,   0.1235
-----
NDVI = (Kiso[858nm]-Kiso[645nm])/(Kiso[858nm]+Kiso[645nm])
NDVI mean:         0.0923
NDVI std. dev.:    0.0185
-----
--- Call to MAIAC_BRDF_stats <> V4 ---
Percent of variance explained by first five principle components 89.97%
Percent of variance explained by first Nλ+2=10 principle components 97.70%

"""
