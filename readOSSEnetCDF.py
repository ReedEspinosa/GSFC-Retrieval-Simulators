#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import glob
import re
from datetime import datetime as dt
import warnings
from MADCAP_functions import loadVARSnetCDF, ordinal2datetime

class osseData(object): # TODO: we still need to add function(s) to microphysics, AOD, etc.
    def __init__(self, fpDict=None):
        """
        fpDict has fields (* -> optional): 
            'polarNc4FP' (string) - full file path of polarimeter data with λ (in nm) replaced w/ %d
            'wvls'* (list of floats) - wavelengths to process in μm, not present or None -> determine them from polarNc4FP
            'dateTime'* (datetime obj.) - day and hour of measurement (min. & sec. loaded from file)
            'asmNc4FP'* (string) - gpm-g5nr.lb2.asm_Nx.YYYYMMDD_HH00z.nc4 file path (has FRLAND for land percentage)
            'metNc4FP'* (string) - gpm-g5nr.lb2.met_Nx.YYYYMMDD_HH00z.nc4 file path (has PS for surface alt.)
        """
        self.wvls = []
        self.measData = None # measData has relevent netCDF data
        self.invldInd = np.array([])
        self.invldIndPurged = False
        if not fpDict: return
        self.verbose = True if 'verbose' in fpDict and fpDict['verbose'] else False
        if ('wvls' not in fpDict) or (not fpDict['wvls']): self.λSearch(fpDict['polarNc4FP'])
        if 'dateTime' not in fpDict: fpDict['dateTime'] = None
        self.readPolarimeterNetCDF(fpDict['polarNc4FP'], fpDict['wvls'], dateTime=fpDict['dateTime'])
        if 'asmNc4FP' in fpDict:
            self.readLevBasmData(fpDict['asmNc4FP'])
        if 'metNc4FP' in fpDict:
            self.readLevBmetData(fpDict['metNc4FP'])
        self.purgeInvldInd()
        
    def λSearch(self, levCFN):
        levCfiles = glob.glob(levCFN.replace('%d','[0-9]*'))
        for fn in levCfiles:
            mtch = re.match(levCFN.replace('%d','([0-9]+)?'), fn)
            if mtch: self.wvls.append(float(mtch.group(1))/1000)
        if self.wvls:
            if self.verbose: 
                print('The following wavelengths were found:', end =" ")
                print('%s μm' % ', '.join([str(λ) for λ in np.sort(self.wvls)]))
            return True
        else:
            warnings.warn('No wavelengths found for the pattern %s' % levCFN)
            return False
        
    def osse2graspRslts(self, NpixMax=None):
        """ osse2graspRslts will convert that data in measData to the format used to store GRASP's output
                IN: NpixMax -> no more than this many pixels will be returned in rslts list (primarly for testing)
                OUT: a list of Npixels dicts with all the keys mapping measData as rslts[nPix][var][nAng, λ] """
        if not self.checkPolarimeterLoaded(): return
        rsltVars = ['fit_I','fit_Q','fit_U','fit_VBS','fit_VExt','fit_DP','fit_LS',          'vis','fis']
        mdVars   = [    'I',    'Q',    'U',    'VBS',    'VExt',    'DP',    'LS','sensor_zenith','fis']        
        Npix = len(self.measData[0]['dtNm']) # this assumes all λ have the same # of pixels
        if NpixMax: Npix = max(Npix, NpixMax)
        Nλ = len(self.measData)
        rslts = [] 
        for k in range(Npix): # loop over pixels
            rslt = dict()
            for l, md in enumerate(self.measData): # loop over λ -- HINT: we assume all λ have the same # of measurements for each msType
                for rv,mv in zip(rsltVars,mdVars): # loop over potential variables
                    if l==0: rslt[rv] = np.empty(len(md[mv][k,:]), Nλ)*np.nan # len(md[mv][k,:]) will change between imagers(Nang) & LIDAR(Nhght)
                    if mv in md: 
                        rslt[rv][:,l] = md[mv][k,:]
                        if l==0 and k==0 and self.verbose: print('%s found in OSSE data' % mv)
                    elif l==0 and k==0 and self.verbose: print('%s NOT found in OSSE data' % mv)
            rslt['lambda'] = self.wvls
            rslt['datetime'] = ordinal2datetime(md['dtNm'][k]) # we keep using md b/c all λ should be the same for these vars
            rslt['sza'] = self.checkReturnField(self, md, 'solar_zenith', k)
            rslt['latitude'] = self.checkReturnField(self, md, 'trjLat', k)
            rslt['longitude'] = self.checkReturnField(self, md, 'trjLon', k)
            rslt['land_prct'] = self.checkReturnField(self, md, 'land_prct', k, 100)
            rslts.append(rslt)
            if self.verbose:
                frmStr = 'Converted pixel #%d (%s), [%6.2f°, %6.2f°], θs=%4.1f, %d%% land'
                ds = rslt['datetime'].strftime("%d/%m/%Y %H:%M:%S")
                print(frmStr % (k, ds, rslt['latitude'], rslt['longitude'], rslt['sza'], rslt['land_prct']))
        return rslts

    def checkReturnField(self, dictObj, field, ind, defualtVal=0):
        if field in dictObj: 
            return dictObj[field][ind] 
        else:
            warnings.warn('%s was not available for OSSE pixel %d, specifying a value of %8.4f.' % (dictObj,ind,defualtVal))
            return defualtVal
    
    def readLevBmetData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height 
            These files are in the LevB data folders and have the form gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 """
        if not self.loadingChecks(filename=levBFN): return
        scaleHght = 8000 # scale height (meters) for presure to alt. conversion, 8km is consistent w/ GRASP
        stndPres = 1.01e5 # standard pressure (Pa)
        minAlt = -100 # defualt GRASP build complains below -100m
        levB_data = loadVARSnetCDF(levBFN, varNames=['PS', 'PBLH'])
        surfPres = np.delete(levB_data['PS'], self.invldInd)
        maslTmp = [scaleHght*np.log(stndPres/PS) for PS in surfPres]
        maslTmp = max(maslTmp, minAlt)
        for md in self.measData: md['masl'] = maslTmp
        for md in self.measData: md['PBLH'] = np.delete(levB_data['PBLH'], self.invldInd) # PBL height in m -- looks a little fishy, should double check w/ patricia
    
    def readLevBasmData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height"""
        if not self.loadingChecks(filename=levBFN): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['FRLAND', 'FRLANDICE'])
        icePix = np.nonzero(levB_data['FRLANDICE'] > 1e-5)[0]
        self.invldInd = np.append(self.invldInd, icePix)
        for md in self.measData: md['land_prct'] = levB_data['FRLAND'] # PBL height in m -- looks a little fishy, should double check w/ patricia
                
    def readPolarimeterNetCDF(self, radianceFNfrmtStr, varNames=None, dateTime=None):
        """ readPolarimeterNetCDF will read a simulated polarimeter data from VLIDORT OSSE
                IN: radianceFNfrmtStr is a string with full path to OSSE files w/ %d replacing λ values
                    varNames* is list of a subset of variables to load
                OUT: will set self.measData and add to invldInd, no data returned """
        if not self.loadingChecks(inPolarimeterNetCDF=True): return # we don't pass filename b/c we only have a pattern at this point
        if not dateTime:
            try:
                dateStrMtch = re.search('[0-9]{8}_[0-9]{4}z', radianceFNfrmtStr)
                dateTime = dt.strptime(dateStrMtch.group(0), "%Y%m%d_%H%Mz")
            except:
                warnings.warn('Using Jan 01, 2000 as date, could not parse polarimeter filename format string (radianceFNfrmtStr = %s)' % radianceFNfrmtStr)
                dateTime = dt.strptime('20000101', "%Y%m%d")
        dayDtNm = dateTime.toordinal()
        measData = [{} for _ in range(len(self.wvls))]
        for i,wvl in enumerate(self.wvls):
            # load data and check for valid indices (I>=0)
            radianceFN = radianceFNfrmtStr % (wvl*1000)
            if self.verbose: print('Processing data from %s' % radianceFN)
            measData[i] = loadVARSnetCDF(radianceFN, varNames)
            self.invldInd = np.append(self.invldInd, np.nonzero((measData[i]['I']<0).any(axis=1))[0]) # only take points w/ I>0 at all wavelengths & angles 
            # convert data to GRASP friendly format
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
                    if self.verbose: print('%4.2fμm Q[U]_surf derived from surf_reflectance_Q[U]_scatplane (scat. plane system)' % wvl)
                else:
                    measData[i]['Q_surf'] = measData[i]['surf_reflectance_Q']*np.cos(30*np.pi/180)
                    measData[i]['U_surf'] = measData[i]['surf_reflectance_U']*np.cos(30*np.pi/180)
                    if self.verbose: print('%4.2fμm Q[U]_surf derived from surf_reflectance_Q[U] (meridian system)' % wvl)
                if (measData[i]['I_surf'] > 0).all():
                    measData[i]['DOLP_surf'] = np.sqrt(measData[i]['Q_surf']**2+measData[i]['U_surf']**2)/measData[i]['I_surf']
                else:
                    measData[i]['DOLP_surf'] = np.full(measData[i]['I_surf'].shape, np.nan)                
            if 'Q_scatplane' in measData[i]: measData[i]['Q_scatplane'] = measData[i]['Q_scatplane']*np.pi
            if 'U_scatplane' in measData[i]: measData[i]['U_scatplane'] = measData[i]['U_scatplane']*np.pi
            measData[i]['fis'] = measData[i]['solar_azimuth'] - measData[i]['sensor_azimuth'] 
            measData[i]['fis'][measData[i]['fis']<0] = measData[i]['fis'][measData[i]['fis']<0] + 360  # GRASP accuracy degrades when φ<0
        self.measData = measData  
    
    def loadingChecks(self, inPolarimeterNetCDF=False, filename=None):
        """ Internal method to check if data has been loaded with readPolarimeterNetCDF """
        assert self.wvls, 'Wavelengths must be set in order to load data!'
        if (self.measData or inPolarimeterNetCDF) and not self.invldIndPurged:
            if self.verbose and filename: print('Processing data from %s' % filename)
            return True
        elif not self.invldIndPurged:
            warnings.warn('You must first load polarimeter data with readPolarimeterNetCDF()!')
        else:
            warnings.warn('You can not load any more data after purging invalid indices once. If you really force loading set invldIndPurged=False.')
        return False
      
    def purgeInvldInd(self):
        """ The method will remove all invldInd from measData """
        if self.invldIndPurged:
            warnings.warn('You should only purge invalid indices once. If you really want to purge again set invldIndPurged=False.')
            return
        self.invldInd = np.array(np.unique(self.invldInd), dtype='int')
        for md in self.measData:
            for varName in np.setdiff1d(list(self.md.keys()), ['x', 'y', 'lev']):
                md[varName] = np.delete(self.md[varName], self.invldInd, axis=0)
        self.invldIndPurged = True        