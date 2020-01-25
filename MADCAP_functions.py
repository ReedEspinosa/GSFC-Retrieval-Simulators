#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import warnings
from netCDF4 import Dataset
import hashlib
import fnmatch
import os

def loadVARSnetCDF(filePath, varNames=None):
    warnings.simplefilter('ignore') # ignore missing_value not cast warning
    measData = dict()
    netCDFobj = Dataset(filePath)
    if varNames is None: varNames = netCDFobj.variables.keys()
    for varName in varNames:
        if varName in netCDFobj.variables.keys():
            measData[varName] = np.array(netCDFobj.variables[varName])
        else:
            warnings.warn('Could not find %s variable in netCDF data' % varName)
    netCDFobj.close()
    warnings.simplefilter('always')
    return measData 

def hashFileSHA1(filePath):
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open(filePath, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

def findNewestMatch(directory, pattern='*'):
    nwstTime = 0
    for file in os.listdir(directory):
        filePath = os.path.join(directory, file)
        if fnmatch.fnmatch(file, pattern) and os.path.getmtime(filePath) > nwstTime:
            nwstTime = os.path.getmtime(filePath)
            newestFN = filePath 
    if nwstTime > 0:
        return newestFN
    else:
        return ''
    
def ordinal2datetime(ordinal):
    dtObjDay = dt.fromordinal(np.int(np.floor(ordinal)))
    dtObjTime = timedelta(seconds=np.remainder(ordinal, 1)*86400)
    dtObj = dtObjDay + dtObjTime
    return dtObj
            
class osseData(object):
    def __init__(self, radianceFNfrmtStr=None, wvls=None, varNames=None):
        self.measData = None # measData has relevent netCDF data
        self.invldInd = None # invldInd has times that were deemed invalid
        
    def osse2graspRslts (self):
        """ osse2graspRslts will convert that data in measData to the format used to store GRASP's output
                OUT: a list of Npixels dicts with all the keys mapping measData as rslts[nPix][var][nAng, λ] """
        if not self.checkPolarimeterLoaded(): return
        rsltVars = ['fit_I','fit_Q','fit_U','fit_VBS','fit_VExt','fit_DP','fit_LS',          'vis','fis']
        mdVars   = [    'I',    'Q',    'U',    'VBS',    'VExt',    'DP',    'LS','sensor_zenith','fis']        
        Npix = len(self.measData[0]['dtNm']) # this assumes all λ have the same # of pixels
        Nλ = len(self.measData)
        rslts = [] 
        for k in range(Npix): # loop over pixels
            rslt = dict()
            for l, md in enumerate(self.measData): # loop over λ -- HINT: we assume all λ have the same # of measurements for each msType
                for rv,mv in zip(rsltVars,mdVars): # loop over potential variables
                    if l==0: rslt[rv] = np.empty(len(md[mv][k,:]), Nλ)*np.nan # len(md[mv][k,:]) will change between imagers(Nang) & LIDAR(Nhght)
                    if mv in md: rslt[rv][:,l] = md[mv][k,:]
            rslt['datetime'] = ordinal2datetime(md['dtNm'][k]) # we keep using md b/c all λ should be the same for these vars
            rslt['sza'] = self.checkReturnField(self, md, 'solar_zenith', k)
            rslt['latitude'] = self.checkReturnField(self, md, 'trjLat', k)
            rslt['longitude'] = self.checkReturnField(self, md, 'trjLon', k)
            rslt['land_prct'] = self.checkReturnField(self, md, 'land_prct', k, 100) # TODO: we still need to get land_prct from somewhere...
            rslts.append(rslt)
        return rslts

    def checkReturnField(self, dictObj, field, ind, defualtVal=0):
        if field in dictObj: 
            return dictObj[field][ind] 
        else:
            warnings.warn('%s was not available for OSSE pixel %d, specifying a value of %8.4f.' % (dictObj,ind,defualtVal))
            return defualtVal
    
    # TODO: we still need to add function(s) to read microphysics, AOD, etc.
    
    def readLevBdata(self, levBFN): # TODO: option to get levBFN automaticly from radianceFNfrmtStr
        """ Read in levelB data to obtain pressure and then surface altitude"""
        if not self.checkPolarimeterLoaded(): return
        scaleHght = 8000 # scale height (meters) for presure to alt. conversion, 8km is consistent w/ GRASP
        stndPres = 1.01e5 # standard pressure (Pa)
        minAlt = -100 # defualt GRASP build complains below -100m
        levB_data = loadVARSnetCDF(levBFN, varNames=['PS'])
        surfPres = np.delete(levB_data['PS'], self.invldInd)
        maslTmp = [scaleHght*np.log(stndPres/PS) for PS in surfPres]
        maslTmp = max(maslTmp, minAlt)
        for md in self.measData: md['masl'] = maslTmp
        
    def checkPolarimeterLoaded(self):
        """ Internal method to check if data has been loaded with readPolarimeterNetCDF """
        if self.measData:
            return True
        else:
            print('\x1b[31m' + 'You must first load polarimeter data with readPolarimeterNetCDF()!' + '\x1b[0m')
            return False
        
    def readPolarimeterNetCDF(self, radianceFNfrmtStr, wvls, varNames=None, datStr='20000101'):
        """ readPolarimeterNetCDF will read a simulated polarimeter data from VLIDORT OSSE
                IN: radianceFNfrmtStr is a string with full path to OSSE files w/ %d replacing λ values
                    wvls is list of λ values in μm, varNames* is list of a subset of variables to load
                OUT: will set measData and invldInd, no data returned """
        dayDtNm = dt.strptime(datStr, "%Y%m%d").toordinal()
        Nwvlth = len(wvls) # TODO: we should automate this to use all availabe wavelengths (based on ls of parrent dir)
        measData = [{} for _ in range(Nwvlth)]
        invldInd = np.array([])
        # load data and check for valid indices (I>=0)
        for i,wvl in enumerate(wvls): # we need to loop over λ twice, first to find valid ind., second to clean data
            radianceFN = radianceFNfrmtStr % (wvl*1000)
            measData[i] = loadVARSnetCDF(radianceFN, varNames)
            invldInd = np.append(invldInd, np.nonzero((measData[i]['I']<0).any(axis=1))[0])
        invldInd = np.array(np.unique(invldInd), dtype='int') # only take points w/ I>0 at all wavelengths & angles
        # convert data to GRASP friendly format
        for i in range(Nwvlth):
            for varName in np.setdiff1d(list(measData[i].keys()), ['x', 'y', 'lev']):
                measData[i][varName] = np.delete(measData[i][varName], invldInd, axis=0)
            measData[i]['dtNm'] = dayDtNm + measData[i]['time'] if 'time' in measData[i] else dayDtNm
            if 'I' in measData[i]:
                measData[i]['I'] = measData[i]['I']*np.pi # GRASP "I"=R=L/FO*pi 
                if 'Q' in measData[i].keys(): measData[i]['Q'] = measData[i]['Q']*np.pi 
                if 'U' in measData[i].keys(): measData[i]['U'] = measData[i]['U']*np.pi
                if 'Q' in measData[i].keys() and 'U' in measData[i].keys():
                    measData[i]['DOLP'] = np.sqrt(measData[i]['Q']**2+measData[i]['U']**2)/measData[i]['I']
            if 'surf_reflectance' in measData[i]:
                measData[i]['I_surf'] = measData[i]['surf_reflectance']*np.cos(30*np.pi/180)
                if 'surf_reflectance_Q_scatplane' in measData[i]:
                    measData[i]['Q_surf'] = measData[i]['surf_reflectance_Q_scatplane']*np.cos(30*np.pi/180)
                    measData[i]['U_surf'] = measData[i]['surf_reflectance_U_scatplane']*np.cos(30*np.pi/180)
                    print('%4.2fμm Q[U]_surf derived from surf_reflectance_Q[U]_scatplane (scat. plane system)' % wvls[i])
                else:
                    measData[i]['Q_surf'] = measData[i]['surf_reflectance_Q']*np.cos(30*np.pi/180)
                    measData[i]['U_surf'] = measData[i]['surf_reflectance_U']*np.cos(30*np.pi/180)
                    print('%4.2fμm Q[U]_surf derived from surf_reflectance_Q[U] (meridian system)' % wvls[i])
                if (measData[i]['I_surf'] > 0).all():
                    measData[i]['DOLP_surf'] = np.sqrt(measData[i]['Q_surf']**2+measData[i]['U_surf']**2)/measData[i]['I_surf']
                else:
                    measData[i]['DOLP_surf'] = np.full(measData[i]['I_surf'].shape, np.nan)
            if 'Q_scatplane' in measData[i]: measData[i]['Q_scatplane'] = measData[i]['Q_scatplane']*np.pi
            if 'U_scatplane' in measData[i]: measData[i]['U_scatplane'] = measData[i]['U_scatplane']*np.pi
            measData[i]['fis'] = measData[i]['solar_azimuth'] - measData[i]['sensor_azimuth'] 
            measData[i]['fis'][measData[i]['fis']<0] = measData[i]['fis'][measData[i]['fis']<0] + 360  # GRASP accuracy degrades when phi<0
            self.measData = measData
            self.invldInd = invldInd    
        