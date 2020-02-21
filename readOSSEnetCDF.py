#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains the definition of the osseData class, which imports OSSE NetCDF data and, if desired, exports it to a list of rslts dicts employing the convention used in GRASP_Scripts. """

import numpy as np
import glob
import re
import warnings
import os
import copy
import datetime as dt
from MADCAP_functions import loadVARSnetCDF

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
        self.measData = None # measData has relevent netCDF data
        self.invldInd = np.array([])
        self.invldIndPurged = False
        if not fpDict: return
        self.verbose = True if 'verbose' in fpDict and fpDict['verbose'] else False
        if ('wvls' not in fpDict) or (not fpDict['wvls']): 
            self.wvls = self.λSearch(fpDict['polarNc4FP'])
        else: 
            self.wvls = fpDict['wvls']
        if 'dateTime' not in fpDict: fpDict['dateTime'] = None
        self.readPolarimeterNetCDF(fpDict['polarNc4FP'], dateTime=fpDict['dateTime'])
        if 'asmNc4FP' in fpDict: self.readLevBasmData(fpDict['asmNc4FP'])
        if 'metNc4FP' in fpDict: self.readLevBmetData(fpDict['metNc4FP'])
        if not ('lcExt' in fpDict: self.convertStateVars2GRASP(fpDict['lcExt'])): # TODO: create the method
        self.purgeInvldInd()
        
    def λSearch(self, levCFN):
        wvls = []
        levCfiles = glob.glob(levCFN.replace('%d','[0-9]*'))
        for fn in levCfiles:
            mtch = re.match(levCFN.replace('%d','([0-9]+)?'), fn)
            if mtch: wvls.append(float(mtch.group(1))/1000)
        wvls = np.sort(wvls)
        if wvls and self.verbose: 
                print('The following wavelengths were found:', end =" ")
                print('%s μm' % ', '.join([str(λ) for λ in wvls]))
        elif not wvls:
            warnings.warn('No wavelengths found for the pattern %s' % levCFN)
        return wvls
        
    def osse2graspRslts(self, NpixMax=None):
        """ osse2graspRslts will convert that data in measData to the format used to store GRASP's output
                IN: NpixMax -> no more than this many pixels will be returned in rslts list (primarly for testing)
                OUT: a list of Npixels dicts with all the keys mapping measData as rslts[nPix][var][nAng, λ] 
                NOTE: GRASP can only use one SZA value & simulator just takes 1st value. Maybe just make them all equal to the average? """
        assert self.measData, 'No OSSE data loaded, nothing to convert to GRASP rslts list!'
        rsltVars = ['fit_I','fit_Q','fit_U','fit_VBS','fit_VExt','fit_DP','fit_LS',          'vis','fis',         'sza']
        mdVars   = [    'I',    'Q',    'U',    'VBS',    'VExt',    'DP',    'LS','sensor_zenith','fis','solar_zenith']        
        measData = self.convertPolar2GRASP() # we can chain existing measData when we add LIDAR
        assert len(self.rtrvdData) == self.Npix, 'OSSE observable and state variable date structures contain a differnt number of pixels!' # TODO: Should Npix above be set as self.Npix and this check performed in convertStateVars2GRASP()?
        if NpixMax: Npix = min(self.Npix, NpixMax)
        Nλ = len(self.measData)
        rslts = [] 
        for k,rd in enumerate(self.rtrvdData[0:Npix]): # loop over pixels
            rslt = dict()
            for rv,mv in zip(rsltVars,mdVars): # loop over potential variables
                for l, md in enumerate(measData): # loop over λ -- HINT: we assume all λ have the same # of measurements for each msType
                    if mv in md: # this is an observable
                        if rv not in rslt: rslt[rv] = np.empty([len(md[mv][k,:]), Nλ])*np.nan # len(md[mv][k,:]) will change between imagers(Nang) & LIDAR(Nhght)
                        rslt[rv][:,l] = md[mv][k,:]
                        if k==0 and self.verbose: print('%s at %4.2f found in OSSE observables' % (mv, self.wvls[l]))
                if rv not in rslt and self.verbose: print('%s at %4.2f NOT found in OSSE data' % (mv, self.wvls[l]))
            rslt['lambda'] = self.wvls
            rslt['datetime'] = md['dtObj'][k] # we keep using md b/c all λ should be the same for these vars
            rslt['latitude'] = self.checkReturnField(md, 'trjLat', k)
            rslt['longitude'] = self.checkReturnField(md, 'trjLon', k)
            rslt['land_prct'] = self.checkReturnField(md, 'land_prct', k, 100)
            if k==0 and self.verbose: 
                for mv in rd.keys(): print('%s found in OSSE state variables' % mv)
            rslt = {**rslt , **rd}
            rslts.append(rslt)
            if self.verbose:
                frmStr = 'Converted pixel #%d (%s), [LAT:%6.2f°, LON:%6.2f°], θs=%4.1f (±%4.1f), %d%% land'
                ds = rslt['datetime'].strftime("%d/%m/%Y %H:%M:%S")
                sza = np.mean(rslt['sza'])
                Δsza = np.abs(rslt['sza']-sza).max()
                print(frmStr % (k, ds, rslt['latitude'], rslt['longitude'], sza, Δsza, rslt['land_prct']))
        return rslts

    def convertStateVars2GRASP(self, levBFN):
        """ osse2graspRslts will read OSSE state variable file(s) and convert to the format used to store GRASP's output
                IN: fileName of lidar lcExt file TODO: add other files w/ more than τ and ω0
                OUT: a list of Npixels dicts - rtrvdData[nPix][var][mode, λ] 
                NOTE: None yet... """
        loadErr = self.loadingChecks(filename=levBFN)
        if loadErr is not 1: self.rtrvdData = np.repeat(dict(), self.Npix)
        if loadErr > 0:
            if loadErr == 2:        
                warnings.warn('State variables could not be loaded!')
            return False            
        for t in range(self.Npix):
        #TODO: integrate AOD and ssa at each pixel and dump them in self.rtrvdData
        #TODO: does self.Npix get set in polarimeter observable loading?
        #TODO: these state variables need to be purged (current method won't do it)
        #TODO: how to map PBLH -> level for PBL aod and SSA? Sent email to Pete and Patricia...
        #        self.measData[0]['PBLH']
        return True

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
        maslTmp = [max(scaleHght*np.log(stndPres/PS), minAlt) for PS in levB_data['PS']]
        for md in self.measData: md['masl'] = maslTmp
        for md in self.measData: md['PBLH'] = levB_data['PBLH'] # PBL height in m -- looks a little fishy, should double check w/ patricia
    
    def readLevBasmData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height"""
        if not self.loadingChecks(filename=levBFN): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['FRLAND', 'FRLANDICE'])
        icePix = np.nonzero(levB_data['FRLANDICE'] > 1e-5)[0]
        self.invldInd = np.append(self.invldInd, icePix).astype(int)
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
                dateTime = dt.datetime.strptime(dateStrMtch.group(0), "%Y%m%d_%H%Mz")
            except:
                warnings.warn('Using Jan 01, 2000 as date, could not parse polarimeter filename format string (radianceFNfrmtStr = %s)' % radianceFNfrmtStr)
                dateTime = dt.datetime.strptime('20000101', "%Y%m%d")
        self.measData = [{} for _ in range(len(self.wvls))]
        for i,wvl in enumerate(self.wvls):
            # load data and check for valid indices (I>=0)
            radianceFN = radianceFNfrmtStr % (wvl*1000)
            if self.verbose: print('Processing data from %s' % radianceFN)
            self.measData[i] = loadVARSnetCDF(radianceFN, varNames)
            tShft = self.measData[i]['time'] if 'time' in self.measData[i] else 0
            self.measData[i]['dtObj'] = [dateTime + dt.timedelta(seconds=int(ts)) for ts in tShft]
            invldIndλ = np.nonzero((self.measData[i]['I']<0).any(axis=1))[0]
            self.invldInd = np.append(self.invldInd, invldIndλ).astype(int) # only take points w/ I>0 at all wavelengths & angles
        self.Npix = len(self.measData[0]['dtObj']) # this assumes all λ have the same # of pixels
        
    def convertPolar2GRASP(self, measData=None):
        """ convert OSSE "measDdata" to a GRASP friendly format 
            IN: if measData argument is provided we work with that, else we use self.measData (this allows chaining with other converters)
            OUT: the converted measData list is returned, self.measData will remain unchanged """
        if not measData: measData = copy.deepcopy(self.measData) # We will return a measData in GRASP format, self.measData will remain unchanged.
        if 'solar_azimuth' not in measData[0]: 
            warnings.warn('solar_azimuth field not found in measData. Was polarimeter data loaded propertly?')
        for wvl,md in zip(self.wvls, measData):            
            if 'I' in md:
                md['I'] = md['I']*np.pi # GRASP "I"=R=L/FO*pi 
                if 'Q' in md.keys(): md['Q'] = md['Q']*np.pi 
                if 'U' in md.keys(): md['U'] = md['U']*np.pi
                if 'Q' in md.keys() and 'U' in md.keys():
                    md['DOLP'] = np.sqrt(md['Q']**2+md['U']**2)/md['I']
            if 'surf_reflectance' in md:
                md['I_surf'] = md['surf_reflectance']*np.cos(30*np.pi/180)
                if 'surf_reflectance_Q_scatplane' in md:
                    md['Q_surf'] = md['surf_reflectance_Q_scatplane']*np.cos(30*np.pi/180)
                    md['U_surf'] = md['surf_reflectance_U_scatplane']*np.cos(30*np.pi/180)
                    if self.verbose: print('%4.2fμm Q[U]_surf derived from surf_reflectance_Q[U]_scatplane (scat. plane system)' % wvl)
                else:
                    md['Q_surf'] = md['surf_reflectance_Q']*np.cos(30*np.pi/180)
                    md['U_surf'] = md['surf_reflectance_U']*np.cos(30*np.pi/180)
                    if self.verbose: print('%4.2fμm Q[U]_surf derived from surf_reflectance_Q[U] (meridian system)' % wvl)
                if (md['I_surf'] > 0).all():
                    md['DOLP_surf'] = np.sqrt(md['Q_surf']**2+md['U_surf']**2)/md['I_surf']
                else:
                    md['DOLP_surf'] = np.full(md['I_surf'].shape, np.nan)                
            if 'Q_scatplane' in md: md['Q_scatplane'] = md['Q_scatplane']*np.pi
            if 'U_scatplane' in md: md['U_scatplane'] = md['U_scatplane']*np.pi
            md['fis'] = md['solar_azimuth'] - md['sensor_azimuth'] 
            md['fis'][md['fis']<0] = md['fis'][md['fis']<0] + 360  # GRASP accuracy degrades when φ<0
        return measData
    
    def loadingChecks(self, inPolarimeterNetCDF=False, filename=None):
        """ Internal method to check if data has been loaded with readPolarimeterNetCDF """
        errCode = 2 #
        if filename and not os.path.exists(filename): 
          warnings.warn('Could not find the file %s' % filename)  
        elif not self.wvls: '
            warnings.warn('Wavelengths must be set in order to load data!')
        elif (self.measData or inPolarimeterNetCDF) and not self.invldIndPurged:
            if self.verbose and filename: print('Processing data from %s' % filename)
            errCode 0
        elif not self.invldIndPurged:
            warnings.warn('You must first load polarimeter data with readPolarimeterNetCDF()!')
        else:
            warnings.warn('You can not load any more data after purging invalid indices once. If you really force loading set invldIndPurged=False.')
            errCode = 1
        return errCode
      
    def purgeInvldInd(self):
        """ The method will remove all invldInd from measData """
        if self.invldIndPurged:
            warnings.warn('You should only purge invalid indices once. If you really want to purge again set invldIndPurged=False.')
            return
        self.invldInd = np.array(np.unique(self.invldInd), dtype='int')
        for md in self.measData:
            for varName in np.setdiff1d(list(md.keys()), ['x', 'y', 'lev']):
                md[varName] = np.delete(md[varName], self.invldInd, axis=0)
        self.invldIndPurged = True
        if self.verbose: 
            print('%d pixels with negative or bad-data-flag reflectances were purged.' % len(self.invldInd))