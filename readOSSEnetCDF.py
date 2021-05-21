#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains the definition of the osseData class, which imports OSSE NetCDF data and, if desired,
exports it to a list of rslts dicts employing the convention used in GRASP_Scripts. """

import numpy as np
import glob
import re
import warnings
import os
import copy
import datetime as dt
from MADCAP_functions import loadVARSnetCDF, downsample1d

GRAV = 9.80616 # m/s^2
SCALE_HGHT = 8000 # scale height (meters) for presure to alt. conversion, 8 km is consistent w/ GRASP
STAND_PRES = 1.01e5 # standard pressure (Pa)
MIN_ALT = -100 # defualt GRASP build complains below -100m
λFRMT = ' %5.3f μm'
βLowLim = 2.0e-10 # lower bound for lidar backscatter and extinction values to be passed into rslt dict, and ultimatly to GRASP
FINE_MODE_THESH = 0.5 # inclusive[exclusive] upper[lower] bound of fine[coarse] mode (μm)


class osseData(object):
    def __init__(self, osseDataPath, orbit, year, month, day=1, hour=0, random=False, wvls=None, lidarVersion=None, \
                 maxSZA=None, oceanOnly=False, loadDust=True, loadPSD=True, pixInd=None, verbose=False):
        """
        IN: orbit - 'ss450' or 'gpm'
            year, month, day - integers specifying data of pixels to load
            hour - integer from 0 to 23 specifying the starting hour of pixels to load
            random - logical, use 10,000 randomly selected pixels for that month (day & hour not needed if random=true)
            'wvls'* (_list_ [not np.array] of floats) - wavelengths to process in μm, not present or None -> determine them from polarNc4FP
            lidarVersion - int w/ lidar instrument version (e.g. 9), x100 for instrument w/o noise (e.g. 900), None to not load lidar
            maxSZA - filter out pixels with mean SZA above this value (degrees)
            oceanOnly - true to skip retrievals on land pixels
            loadDust - load size distributions of dust, as well as for total aerosol (ignored if loadPSD==False)
            loadPSD - skip reading and converting of PSD data to save processing time
        Note: buildFpDict() method below will setup file paths mirroring the path structure on DISCOVER (more details in its header comment)
        """
        self.measData = None # measData has observational netCDF data
        self.rtrvdData = None # rtrvdData has state variables from netCDF data
        self.invldInd = np.array([])
        self.invldIndPurged = False
        self.loadingCalls = []
        self.pblInds = None
        self.maxSZA = maxSZA
        self.oceanOnly = oceanOnly
        self.lidarVersion = lidarVersion
        self.verbose = verbose
        self.vldPolarλind = None # needed b/c measData at lidar data λ is missing some variables
        self.buildFpDict(osseDataPath, orbit, year, month, day, hour, random, self.lidarVersion)
        self.wvls = self.λSearch() if wvls is None else list(wvls) # wvls should be a list
        if self.lidarVersion is None or self.lidarVersion==0:
            if self.verbose: print('Lidar version not provided, not attempting to load any lidar data')
            self.wvls = [wvl for wvl in self.wvls if wvl not in [0.355,0.532,1.064]] # Remove lidar wavelengths in case caller provided them
            self.loadAllData(loadLidar=False, loadDust=loadDust, loadPSD=loadPSD, pixInd=pixInd)
        else:
            if not self.lidarVersion==6: self.wvls = [wvl for wvl in self.wvls if not wvl==0.355] # Remove lidar wavelengths in case caller provided them
            self.loadAllData(loadLidar=True, loadDust=loadDust, loadPSD=loadPSD, pixInd=pixInd)

    def loadAllData(self, loadLidar=True, loadDust=True, loadPSD=True, pixInd=None):
        """Loads NetCDF OSSE data into memory, see buildFpDict below for variable descriptions"""
        assert self.readPolarimeterNetCDF(pixInd=pixInd), 'This class currently requires a polarNc4FP file to be present.' # reads polarNc4FP
        self.readasmData(pixInd=pixInd) # reads asmNc4FP
        self.readmetData(pixInd=pixInd) # reads metNc4FP
        self.readaerData(pixInd=pixInd) # reads aerNc4FP
        self.readStateVars(pixInd=pixInd) # reads lcExt
        self.readStateVars(finemode=True, pixInd=pixInd) # reads lcExt (finemode version)
        if loadPSD: self.readPSD(incldDust=loadDust, baseSDparamsOnly=False, pixInd=pixInd) # reads aerSD
        if loadLidar: self.readlidarData(pixInd=pixInd) # reads lc2Lidar
        self.purgeInvldInd(self.maxSZA, self.oceanOnly)

    def buildFpDict(self, osseDataPath, orbit, year, month, day=1, hour=0, random=False, lidarVer=0):
        """
        See _init_ for input variable definitions
        Sets the self.fpDict dictionary with filepath fields:
            'polarNc4FP' (string) - full file path of polarimeter data with λ (in nm) replaced w/ %d
            'dateTime'* (datetime obj.) - day and hour of measurement (min. & sec. loaded from file)
            'asmNc4FP'* (string) - gpm-g5nr.lb2.asm_Nx.YYYYMMDD_HH00z.nc4 file path (FRLAND for land percentage)
            'metNc4FP'* (string) - gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 file path (PS for surface alt.)
            'aerNc4FP'* (string) - gpm-g5nr.lb2.aer_Nv.YYYYMMDD_HH00z.nc4 file path (DELP/AIRDEN for level heights)
            'psdNc4FP'* (string) - gpm-g5nr.lb2.aer_SD.YYYYMMDD_HH00z.nc4 file path (contains speciated & total size distributions)
            'lcExt'* (string)    - gpm-g5nr.lc.ext.YYYYMMDD_HH00z.%dnm path (has τ, SSA, CRI, etc.)
            'lc2Lidar'* (string) - gpm-lidar-g5nr.lc2.YYYYMMDD_HH00z.%dnm path to file w/ simulated [noise added] lidar measurements
        All of the above files should contain noise free data, except lc2Lidar -
        """
        assert osseDataPath is not None, 'osseDataPath, Year, month and orbit must be provided to build fpDict'
        if lidarVer is None: lidarVer=0
        ldStr = 'LIDAR%02d' % (np.floor(lidarVer/100) if lidarVer >= 100 else lidarVer)
        if random:
            tmStr = 'random.%04d%02d01_0000z' % (year, month)
            dtTple = (year, month)
            pathFrmt = os.path.join(osseDataPath, orbit.upper(), 'Level%s', 'Y%04d','M%02d', '%s')
        else:
            tmStr = '%04d%02d%02d_%02d00z' % (year, month, day, hour)
            dtTple = (year, month, day)
            pathFrmt = os.path.join(osseDataPath, orbit.upper(), 'Level%s', 'Y%04d','M%02d', 'D%02d', '%s')
        self.fpDict = {
            'polarNc4FP': pathFrmt % (('C',)+dtTple+(orbit+'-polar07-g5nr.lc.vlidort.'+tmStr+'_%dd00nm.nc4',)),
            'asmNc4FP'  : pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.asm_Nx.'+tmStr+'.nc4',)),
            'metNc4FP'  : pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.met_Nv.'+tmStr+'.nc4',)),
            'aerNc4FP'  : pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.aer_Nv.'+tmStr+'.nc4',)),
            'psdNc4FP'  : pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.aer_SD.'+tmStr+'.nc4',)),
            'lcExt'     : pathFrmt % (('C',)+dtTple+(orbit+'-g5nr.lc.ext.'+tmStr+'.%dnm.nc4',)),
            'lc2Lidar'  : pathFrmt % (('D',)+dtTple+(orbit+'-g5nr.lc2.'+tmStr+'.%dnm.'+ldStr+'.nc4',)), # will need a str replace, e.g. VAR.replace('LIDAR', 'LIDAR09')
            'savePath'  : pathFrmt % (('E',)+dtTple+(orbit+'-g5nr.leV%02d.GRASP.%s.%s.'+tmStr+'.pkl',)) # % (vrsn, yamlTag, archName) needed from calling function
        }
        self.fpDict['dateTime'] = dt.datetime(year, month, day, hour)
        self.fpDict['noisyLidar'] = lidarVer < 100 # e.g. 0900 is noise free, but 09 is not
        if random:
            self.fpDict['lc2Lidar'] = self.fpDict['lc2Lidar'].replace('random.','').replace('nc4','RANDOM.7000m.nc4')
        return self.fpDict

    def λSearch(self):
        wvls = []
        posKeys = ['polarNc4FP', 'lc2Lidar'] # we only loop over files with lidar or polarimter data
        posPaths = [self.fpDict[key] for key in posKeys if key in self.fpDict]
        for levCFN in posPaths:
            levCfiles = glob.glob(levCFN.replace('%d','[0-9]*'))
            for fn in levCfiles:
                mtch = re.match(levCFN.replace('%d','([0-9]+)?'), fn)
                wvlVal = float(mtch.group(1))/1000
                if mtch and wvlVal not in wvls: wvls.append(wvlVal)
        wvls = list(np.sort(wvls)) # this will also convert from list to numpy array
        if wvls and self.verbose:
            print('The following wavelengths were found:', end=" ")
            print('%s μm' % ', '.join([str(λ) for λ in wvls]))
        elif not wvls:
            warnings.warn('No wavelengths found for the patterns:\n%s' % "\n".join(posPaths))
        return wvls

    def osse2graspRslts(self, newLayers=None, dateTimeSorted=True):
        """ osse2graspRslts will convert that data in measData to the format used to store GRASP's output
                IN: newLidarLayers -> a list of layer heights (in meters) at which to return the lidar signal
                    dateTimeSorted -> rslts will be sorted by datetime (as GRASP requires), but will lose alignment with order of pixels in this object
                OUT: a list of nPix dicts with all the keys mapping measData as rslts[nPix][var][nAng, λ]
                NOTE: GRASP SDATA only takes one SZA value but all rslt dicts have one for each view angle (SZA of first angle will be pulled by nowPix.populateFromRslt()) """
        assert self.measData, 'self.measData must be set (e.g through readPolarimeterNetCDF()) before calling this method!'
        rsltVars = ['fit_I','fit_Q','fit_U','fit_VBS','fit_VExt','fit_DP','fit_LS',          'vis','fis',         'sza', 'RangeLidar', 'range']
        mdVars   = [    'I',    'Q',    'U',    'VBS',    'VExt',    'DP',    'LS','sensor_zenith','fis','solar_zenith', 'RangeLidar', 'range']
        measData = self.convertPolar2GRASP()
        measData = self.convertLidar2GRASP(measData, newLayers)
        Nλ = len(measData)
        assert len(self.rtrvdData) == self.Npix, \
            'OSSE observable and state variable data structures contain a differnt number of pixels!'
        assert ('aod' not in self.rtrvdData[0]) or (len(self.rtrvdData[0]['aod']) == Nλ), \
            'OSSE observable and state variable data structures contain a differnt number of wavelengths!'
        pixInd = np.r_[0:len(self.rtrvdData)]
        if dateTimeSorted: pixInd = np.take(pixInd, np.argsort(measData[self.vldPolarλind]['dtObj'][pixInd]))
        rslts = []
        for k,rd in zip(pixInd, self.rtrvdData[pixInd]): # loop over pixels
            rslt = dict()
            for rv,mv in zip(rsltVars,mdVars): # loop over potential variables
                for l, md in enumerate(measData): # loop over λ -- HINT: we assume all λ have the same # of measurements for each msType
                    if mv in md: # this is an observable
                        if rv not in rslt: rslt[rv] = np.empty([len(md[mv][k,:]), Nλ])*np.nan # len(md[mv][k,:]) will change between imagers(Nang) & LIDAR(Nhght)
                        rslt[rv][:,l] = md[mv][k,:]
                        if k==pixInd[0] and self.verbose>1: print(('%s at '+λFRMT+' found in OSSE observables') % (mv, self.wvls[l]))
                if k==pixInd[0] and rv not in rslt and self.verbose>1: print('%s NOT found in OSSE data' % mv)
            rslt['lambda'] = np.asarray(self.wvls)
            rslt['datetime'] = measData[self.vldPolarλind]['dtObj'][k] # we keep using md b/c all λ should be the same for these vars
            rslt['latitude'] = self.checkReturnField(measData[self.vldPolarλind], 'trjLat', k)
            rslt['longitude'] = self.checkReturnField(measData[self.vldPolarλind], 'trjLon', k)
            rslt['land_prct'] = self.checkReturnField(measData[self.vldPolarλind], 'land_prct', k, 100)
            if k==pixInd[0] and self.verbose>1:
                for mv in rd.keys(): print('%s found in OSSE state variables' % mv)
            rslt = {**rslt, **rd}
            rslts.append(rslt)
            if self.verbose:
                frmStr = 'Converted pixel #%05d (%s), [LAT:%6.2f°, LON:%6.2f°], %3.0f%% land, asl=%4.0f m'
                ds = rslt['datetime'].strftime("%d/%m/%Y %H:%M:%S")
                sza, Δsza = self.angleVals(rslt['sza'])
                vφa, Δvφa = self.angleVals(rslt['fis'])
                with np.errstate(invalid='ignore'): vza, Δvza = self.angleVals(rslt['vis']*(1-2*(rslt['fis']>180))) # need with statement b/c fis could be NaN at lidar λ
                frmStr = frmStr + ', θs=%4.1f° (±%4.1f°), φ=%4.1f° (±%4.1f°), θv=%4.1f° (±%4.1f°)'
                print(frmStr % (k, ds, rslt['latitude'], rslt['longitude'], rslt['land_prct'],
                                rd['masl'], sza, Δsza, vφa, Δvφa, vza, Δvza))
        # TODO: USE norm2absExtProf TO ADD 'βext' to rslts... but actually that is normalized and independent of λ
        #          'βext' should really be the mode resolved 'volProf'...
        return rslts

    def angleVals(self, keyVal):
        angle = np.nanmean(keyVal) # nanmean b/c lidar channels will have nans
        Δangle = np.nanmax(np.abs(keyVal-angle))
        return angle, Δangle

    def checkReturnField(self, dictObj, field, ind, defualtVal=0):
        if field in dictObj:
            return dictObj[field][ind]
        else:
            warnings.warn('%s was not available for OSSE pixel %d, specifying a value of %8.4f.' % (dictObj,ind,defualtVal))
            return defualtVal

    def readPolarimeterNetCDF(self, varNames=None, pixInd=None):
        """ readPolarimeterNetCDF will read a simulated polarimeter data from VLIDORT OSSE
                IN: varNames* is list of a subset of variables to load
                OUT: will set self.measData and add to invldInd, no data returned """
        if not self._loadingChecks(functionName='readPolarimeterNetCDF'): return False # we don't pass filename b/c we only have a pattern at this point
        self.measData = [{} for _ in range(len(self.wvls))]
        for i,wvl in enumerate(self.wvls):
            radianceFN = self.fpDict['polarNc4FP'] % (wvl*1000)
            if os.path.exists(radianceFN): # load data and check for valid indices (I>=0)
                if self.verbose: print('Processing data from %s' % radianceFN)
                self.measData[i] = loadVARSnetCDF(radianceFN, varNames, keepTimeInd=pixInd,verbose=self.verbose)
                tShft = self.measData[i]['time'] if 'time' in self.measData[i] else 0
                self.measData[i]['dtObj'] = np.array([self.fpDict['dateTime'] + dt.timedelta(seconds=int(ts)) for ts in tShft])
                with np.errstate(invalid='ignore'): # we need this b/c we will be comaring against NaNs
                    invldBool = np.logical_or(np.isnan(self.measData[i]['I']), self.measData[i]['I'] < 0)
                invldIndλ = invldBool.any(axis=1).nonzero()[0]
                self.invldInd = np.append(self.invldInd, invldIndλ).astype(int) # only take points w/ I>0 at all wavelengths & angles
                if not self.vldPolarλind: self.vldPolarλind = i # store the 1st λ index with polarimeter data
            elif self.verbose:
                print('No polarimeter data found at' + λFRMT % wvl)
        NpixByλ = np.array([len(md['dtObj']) for md in self.measData if 'dtObj' in md])
        assert len(NpixByλ)>0, 'Polarimeter data was not found at any wavelength!'
        assert np.all(NpixByλ[0] == NpixByλ), 'This class assumes that all λ have the same number of pixels.'
        self.Npix = NpixByλ[0]
        self.rtrvdData = np.array([{} for _ in range(self.Npix)])
        return True

    def readasmData(self, pixInd=None):
        """ Read in levelB data to obtain the fraction of land in each pixel"""
        levBFN = self.fpDict['asmNc4FP']
        call1st = 'readPolarimeterNetCDF'
        if not self._loadingChecks(prereqCalls=call1st, filename=levBFN, functionName='readasmData'): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['FRLAND', 'FRLANDICE'], keepTimeInd=pixInd, verbose=self.verbose)
        icePix = np.nonzero(levB_data['FRLANDICE'] > 1e-5)[0]
        self.invldInd = np.append(self.invldInd, icePix).astype(int)
        levB_data['FRLAND'][levB_data['FRLAND']>0.99] = 1.0
        levB_data['FRLAND'][levB_data['FRLAND']<0.01] = 0.0
        for md in self.measData: md['land_prct'] = 100*levB_data['FRLAND'] # OSSE data is 0-1 but GRASP wants true percentage (0-100)

    def readmetData(self, pixInd=None):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height
            These files are in the LevB data folders and have the form gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 """
        levBFN = self.fpDict['metNc4FP']
        call1st = 'readPolarimeterNetCDF'
        if not self._loadingChecks(prereqCalls=call1st, filename=levBFN, functionName='readmetData'): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['PS', 'PBLH'], keepTimeInd=pixInd, verbose=self.verbose)
        maslTmp = np.array([max(SCALE_HGHT*np.log(STAND_PRES/PS), MIN_ALT) for PS in levB_data['PS']])
        for maslVal,rd in zip(maslTmp, self.rtrvdData): rd['masl'] = maslVal
        for pblhVal,rd in zip(levB_data['PBLH'], self.rtrvdData): rd['pblh'] = pblhVal

    def readaerData(self, pixInd=None):
        """ Read in levelB data to obtain vertical layer heights """ # HOW TO PS WORK IN HERE? gpm-g5nr.lb2.aer_Nv.20060801_0000z.nc4.. is that aerNc4FP? it has PS and the vars below...
        levBFN = self.fpDict['aerNc4FP']
        preReqs = ['readPolarimeterNetCDF','readmetData']
        if not self._loadingChecks(prereqCalls=preReqs, filename=levBFN, functionName='readaerData'): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['AIRDENS', 'DELP'], keepTimeInd=pixInd, verbose=self.verbose) # air density [kg/m^3], pressure thickness [Pa]
        self.Nlayers = levB_data['AIRDENS'].shape[1]
        self.pblInds = np.zeros((self.Npix, self.Nlayers), dtype=np.bool)
        for k,(airdens,delp) in enumerate(zip(levB_data['AIRDENS'], levB_data['DELP'])): # loop over pixels
            ze = (delp[::-1]/airdens[::-1]/GRAV).cumsum()[::-1] # profiles run top down so we reverse order for cumsum
            rng = (np.r_[ze[1::],0] + ze)/2 # ze is then the top of the layers (?), so rng is midpoints
            pblTopInd = np.argmin(np.abs(self.rtrvdData[k]['pblh'] - rng))
            self.pblInds[k, pblTopInd:self.Nlayers] = True
            for md in self.measData:
                if 'range' not in md: # i.e. k==0
                    md['range'] = np.full([self.Npix, len(delp)], np.nan)
                md['range'][k,:] = rng

    def readStateVars(self, finemode=False, pixInd=None):
        """ readStateVars will read OSSE state variable file(s) and convert to the format used to store GRASP's output
            IN: dictionary stateVarFNs containing key 'lcExt' with path to lidar "truth" file
            RESULT: sets self.rtrvdData, numpy array of Npixels dicts -> rtrvdData[nPix][varKey][mode, λ]"""
        if finemode:
            modeStr = '_fine'
            lcExtFNPtrn = self.fpDict['lcExt'].replace('g5nr.lc.ext.', 'g5nr.lc.ext.finemode.')
            assert not self.fpDict['lcExt'] == lcExtFNPtrn, 'Unable to convert lcExt file patern to finemode format.'
        else:
            modeStr = ''
            lcExtFNPtrn = self.fpDict['lcExt']
        if 'lcExt' not in self.fpDict:
            if self.verbose: print('lcExt filename not provided, no state variable data read.')
            return False
        netCDFvarNames = ['tau','ssa','refr','refi','reff','vol','g','ext2back'] # Order is important! (see for loop in rtrvdDataSetPixels())
        preReqs = ['readPolarimeterNetCDF','readaerData']
        if not self._loadingChecks(prereqCalls=preReqs, functionName='readStateVars'): return
        for λ,wvl in enumerate(self.wvls): # NOTE: will only use data corresponding to λ values in measData
            lcExtFN = lcExtFNPtrn % (wvl*1000)
            if os.path.exists(lcExtFN):
                if self.verbose: print('Processing data from %s' % lcExtFN)
                lcD = loadVARSnetCDF(lcExtFN, varNames=netCDFvarNames, keepTimeInd=pixInd, verbose=self.verbose)
                for t in range(lcD['reff'].shape[0]): # HACK (next 4 lines) until we better understand zeros and NaNs in OSSE output
                    for lev in range(lcD['reff'].shape[1]):
                        if np.isnan(lcD['reff'][t,lev]) or lcD['reff'][t,lev] < 1e-12:
                            lcD['reff'][t,lev] = 1e-7 if lev==0 else lcD['reff'][t,lev-1]
                timeLoopVars = [lcD[key] for key in netCDFvarNames] # list of values for keys defined above
                timeLoopVars.append(self.rtrvdData)
                self._rtrvdDataSetPixels(timeLoopVars, λ, km=modeStr) # contains list of rtrvdData dicts, which are mutable
                pblStr = modeStr + 'PBL' if finemode else '_PBL'
                self._rtrvdDataSetPixels(timeLoopVars, λ, self.pblInds, km=pblStr)
                ftStr = modeStr + 'FT' if finemode else '_FT'
                self._rtrvdDataSetPixels(timeLoopVars, λ, ~self.pblInds, km=ftStr)
                if finemode: self._estimateCoarseMode()
            elif self.verbose:
                msg = 'No lcExt%s state variable data found at' % modeStr.replace('_',' ')
                print(msg + λFRMT % wvl)

    def _rtrvdDataSetPixels(self, timeLoopVars, λ, hghtInds=None, km=''):
        """ hghtInd is a logical 2D array [Npix X Nlayer] with true at index of values to use """
        if hghtInds is None: hghtInds = np.ones((self.Npix, self.Nlayers), dtype=np.bool)
        firstλ = 'aod'+km not in timeLoopVars[-1][0]
        for τ,ω,n,k,rEff,V,g,S,rd,hInd in zip(*timeLoopVars, hghtInds): # loop over each pixel and vertically average
            if firstλ:
                for varKey in ['aod'+km,'ssa'+km,'n'+km,'k'+km,'g'+km, 'LidarRatio'+km]: # spectrally dependent vars only, reff doesn't need allocation
                    rd[varKey] = np.full(len(self.wvls), np.nan)
            if np.any(np.isnan(rEff)):
                print('¡¡¡¡NANS IN REFF!!!!')
                return
            rd['aod'+km][λ] = τ[hInd].sum()
            if rd['aod'+km][λ]==0:
                print('IT WAS ZERO')
            rd['ssa'+km][λ] = np.sum(τ[hInd]*ω[hInd])/rd['aod'+km][λ] # ω=Σβh/Σαh & ωh*αh=βh => ω=Σωh*αh/Σαh
            rd['n'+km][λ] = np.sum(τ[hInd]*n[hInd])/rd['aod'+km][λ] # n is weighted by τ (this is a non-physical quantity)
            rd['k'+km][λ] = np.sum(τ[hInd]*k[hInd])/rd['aod'+km][λ] # k is weighted by τ (this is a non-physical quantity)
            rEff[rEff < 1e-12] = 1e-12
            rEffTot = np.sum(V[hInd])/np.sum(V[hInd]/rEff[hInd]) # see bottom of this method for derivation
            if firstλ:
                rd['rEff'+km] = rEffTot
            elif not np.isclose(rd['rEff'+km], rEffTot) and not np.isnan(rEffTot):
                warnings.warn('The current rEff (%8.5f μm) differs from the first rEff (%8.5f μm) derived.' % (rEffTot, rd['rEff'+km]))
            rd['g'+km][λ] = np.sum(τ[hInd]*ω[hInd]*g[hInd])/np.sum(τ[hInd]*ω[hInd]) # see bottom of this method for derivation
            rd['LidarRatio'+km][λ] = np.sum(τ[hInd])/np.sum(τ[hInd]/S[hInd]) # S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh)
        """ --- reff vertical averaging (reff_h=Rh, # conc.=Nh, vol_conc.=Vh, height_ind=h) ---
            reff = ∫ ΣNh*r^3*dr/∫ ΣNh*r^2*dr = 3/4/π*ΣVh/∫ Σnh*r^2*dr
            Rh = (3/4/π*Vh)/∫ nh*r^2*dr -> reff = 3/4/π*ΣVh/(Σ(3/4/π*Vh)/Rh) = ΣVh/Σ(Vh/Rh)
            --- g vertical averaging (normalized PF = P11[θ], absolute PF = F11[θ]) ---
            P11[θ] = F11[θ]/β = Σ(τh*ωh*P11h[θ])/Στh*ωh -> g = ∫ P11[θ]...dθ = ∫ Σ(τh*ωh*P11h[θ])/Στh*ωh...dθ
            g = Σ(τh*ωh*∫P11h[θ]...dθ)/Στh*ωh = Σ(τh*ωh*gh)/Σ(τh*ωh) """

    def _estimateCoarseMode(self):
        """"Add the total, PBL and FT coarse mode values of τ, n & k to self.rtrvdData"""
        layerStrs = ['', 'PBL', 'FT']
        riStrs = ['n', 'k']
        for rd in self.rtrvdData: # loop over pixels
            for ls in layerStrs: # loop over vertical layers
                tauTotKey = 'tau' if ls=='' else 'tau_'+ls # we only need it in FT or PBL case
                rd['tau_coarse'+ls] = rd[tauTotKey] - rd['tau_fine'+ls]
                for ri in riStrs: # loop over n & k
                    termTot = rd[tauTotKey]/rd['tau_coarse'+ls]*rd[ri]
                    termFine = rd['tau_fine'+ls]/rd['tau_coarse'+ls]*rd[ri+'_fine'+ls]
                    rd[ri+'_coarse'+ls] = termTot - termFine # nc = τ/τc*n - τf/τc*nf (uses τ weighted averaging)

    def readPSD(self, incldDust=True, baseSDparamsOnly=False, pixInd=None):
        """ Read in size distribution data and separate by mode/height(PBL vs FT) """
        dists2pull = ['TOTdist'] # could add SUdist, SSdist, OCPHOBICdist, OCPHILICdist, etc.
        if incldDust: dists2pull.append('DUdist')
        levBFN = self.fpDict['psdNc4FP']
        preReqs = ['readPolarimeterNetCDF','readmetData']
        if not self._loadingChecks(prereqCalls=preReqs, filename=levBFN, functionName='readPSD'): return
        psdData = loadVARSnetCDF(levBFN, varNames=['radius']+dists2pull, keepTimeInd=pixInd, verbose=self.verbose) # radius [m], dv/dr [m^3/m]
        fineInd = psdData['radius'] <= FINE_MODE_THESH/1e6
        crseInd = psdData['radius'] > FINE_MODE_THESH/1e6
        for getKey in dists2pull:
            for dvdr,rd,pblInd in zip(psdData[getKey], self.rtrvdData, self.pblInds): # loop over pixels
                if 'r' not in rd: rd['r'] = psdData['radius']*1e6
                dvdr = dvdr/1e6
                prfx = '' if getKey=='TOTdist' else '_'+getKey.replace('dist','')
                self._calcPSDvals(rd, dvdr, prfx, 'fine', modeInds=fineInd)
                self._calcPSDvals(rd, dvdr, prfx, 'coarse', modeInds=crseInd)
                self._calcPSDvals(rd, dvdr, prfx, '')
                if not baseSDparamsOnly:
                    self._calcPSDvals(rd, dvdr, prfx, 'finePBL', hgtInds=pblInd, modeInds=fineInd)
                    self._calcPSDvals(rd, dvdr, prfx, 'coarsePBL', hgtInds=pblInd, modeInds=crseInd)
                    self._calcPSDvals(rd, dvdr, prfx, 'PBL', hgtInds=pblInd)
                    self._calcPSDvals(rd, dvdr, prfx, 'fineFT', hgtInds=~pblInd, modeInds=fineInd)
                    self._calcPSDvals(rd, dvdr, prfx, 'coarseFT', hgtInds=~pblInd, modeInds=crseInd)
                    self._calcPSDvals(rd, dvdr, prfx, 'FT', hgtInds=~pblInd)
        if not incldDust: return
        for pstFx in ['', '_fine', '_coarse']:
            for pstFx2 in ['', 'PBL', 'FT']:
                fullFx = pstFx+'_'+pstFx2 if not pstFx2=='' and pstFx=='' else pstFx+pstFx2
                for rd in self.rtrvdData: # loop over pixels & find sphere fraction (assumes dust is only (and completely) non-spherical type)
                    rd['sph'+fullFx] = 1 - rd['vol_DU'+fullFx]/rd['vol'+fullFx]

    def _calcPSDvals(self, rd, dvdr, prfx, var, hgtInds=None, modeInds=slice(None)):
        """ Not hgtInds and modeInds should be logical (although ints may also work)"""
        rdSetKey = prfx if var=='' else '%s_%s' % (prfx, var)
        if hgtInds is None: hgtInds = np.ones(self.Nlayers, dtype=np.bool) # use all vertical layers
        # N(r) [integrated over height]
        rd['dVdlnr'+rdSetKey] = np.zeros(len(rd['r']))
        rd['dVdlnr'+rdSetKey][modeInds] = dvdr[hgtInds][:, modeInds].sum(axis=0)*rd['r'][modeInds] # TODO: FIX THESE UNITS!
        # V(h) [integrated over size]
        rd['volProf'+rdSetKey] = np.zeros(self.Nlayers)
        vPrfFineKy = 'volProf%s_fine' % prfx
        vPrfCoarseKy = 'volProf%s_coarse' % prfx
        assert sum(hgtInds)>0, 'At least one entry in hgtInds must be True – the layer has to be somewhere.'
        for Indlyr, dvdrLyr in zip(np.where(hgtInds)[0], dvdr[hgtInds][:,modeInds]): # loop over layers – top down (IndLyr=71 is bottom)
            if 'fine' in var and not 'fine'==var and vPrfFineKy in rd: # we want PBL(FT) specific but already have column
                rd['volProf'+rdSetKey][Indlyr] = rd[vPrfFineKy][Indlyr]
            elif 'coarse' in var and not 'coarse'==var and vPrfCoarseKy in rd: # we want PBL(FT) specific but already have column
                rd['volProf'+rdSetKey][Indlyr] = rd[vPrfCoarseKy][Indlyr]
            elif vPrfFineKy in rd and vPrfCoarseKy in rd and var not in ['fine', 'coarse']: # we want total, just sum fine+coarse mode profiles
                rd['volProf'+rdSetKey][Indlyr] = rd[vPrfFineKy][Indlyr] + rd[vPrfCoarseKy][Indlyr]
            else: # we got nothing, need to perform the _SLOW_ integration
                rd['volProf'+rdSetKey][Indlyr] = np.trapz(dvdrLyr, x=rd['r'][modeInds])
        # total volume [integrate over both]
        rd['vol'+rdSetKey] = rd['volProf'+rdSetKey].sum()

    def readlidarData(self,pixInd=None):
        ncDataVars09   = {'LS':'INSTRUMENT_ATB'}
        ncDataVarsHSRL = {'VBS':'HSRL_BACKSCAT', 'VExt':'HSRL_EXTINCTION'}
        call1st = 'readPolarimeterNetCDF'
        if not self._loadingChecks(prereqCalls=call1st, functionName='readlidarData'): return # removing this prereq requires factoring out creation of measData and setting some keys (e.g. time, dtObj) in polarimeter reader
        for i,wvl in enumerate(self.wvls):
            hsrlAtnBck = self.lidarVersion in [5, 6] and np.isclose(wvl, 1.064, atol=0.01)
            if self.lidarVersion==9 or hsrlAtnBck:
                ncDataVars = copy.copy(ncDataVars09) # we may add "_NOISE" but need to preserve unaltered defintion
                filePath = self.fpDict['lc2Lidar'].replace('LIDAR05','LIDAR09').replace('LIDAR06','LIDAR09') # HACK: no 1064nm HSRL simulations – use lidar09 noise free data
            elif self.lidarVersion in [5, 6]:
                ncDataVars = copy.copy(ncDataVarsHSRL)
                filePath = self.fpDict['lc2Lidar'].replace('nc4','30m.nc4') # 30m is on end of HSRL files, but not lidar09
                if self.lidarVersion == 6: filePath = filePath.replace('7000m','50000m') # lidar 6 only has 50km res
            else:
                assert False, '%d is not a valid lidarVersion!' % self.lidarVersion
            if self.fpDict['noisyLidar'] and not hsrlAtnBck:
                for k in ncDataVars.keys(): ncDataVars[k] = ncDataVars[k]+'_NOISE'
            lidarFN = filePath % (wvl*1000)
            if os.path.exists(lidarFN): # data is there, load it
                if self.verbose: print('Processing data from %s' % lidarFN)
                lidar_data = loadVARSnetCDF(lidarFN, varNames=['lev', 'SURFACE_ALT']+list(ncDataVars.values()), keepTimeInd=pixInd, verbose=self.verbose)
                self.measData[i]['RangeLidar'] = lidar_data['lev']*1e3 # km -> m
                self.measData[i]['SurfaceAlt'] = lidar_data['SURFACE_ALT']*1e3 # km -> m
                for mdKey, ldKey in ncDataVars.items():
                    self.measData[i][mdKey] = lidar_data[ldKey]/1e3 # km-1 [sr-1] -> m-1 [sr-1]
            elif self.verbose:
                print('No lidar data found at' + λFRMT % wvl)
        return

    def convertLidar2GRASP(self, measData=None, newLayers=None):
        """ convert OSSE lidar "measDdata" to a GRASP friendly format
            IN: if measData argument is provided we work with that, else we use self.measData (this allows chaining with other converters)
                if newLayers is not provided, some elements of measData[n]['RangeLidar'] will correspond to below ground level
            OUT: the converted measData list is returned, self.measData will remain unchanged """
        if not measData:
            assert self.measData, 'measData must be provided or self.measData must be set!'
            measData = copy.deepcopy(self.measData) # We will return a measData in GRASP format, self.measData will remain unchanged.
        if newLayers is not None: newLayers = np.sort(newLayers) # sort in ascending order for downsample1d (flip later for GRASP which needs descending order)
        for wvl,md in zip(self.wvls, measData): # loop over λ
            if 'RangeLidar' in md:
                assert np.all(np.diff(md['RangeLidar'])<0), 'RangeLidar should be in descending order.'
                if self.verbose: print('Converting lidar data to GRASP units at' + λFRMT % wvl)
                md['RangeLidar'] = np.tile(md['RangeLidar'], [self.Npix,1]) - md['SurfaceAlt'][:,None] # RangeLidar is now relative to the surface
                if newLayers is not None:
                    for key in [k for k in ['LS', 'VExt', 'VBS',' DP'] if k in md]: # loop over lidar measurement types
                        tempSig = np.empty((self.Npix, len(newLayers)))
                        for t in range(self.Npix):
                            vldInd = np.logical_and(md['RangeLidar'][t,:]>30, ~np.isnan(md[key][t,:]))  # we will remove lidar returns corresponding to below ground (background subtraction)
                            alt = md['RangeLidar'][t, vldInd]
                            sig = md[key][t, vldInd]
                            tempSig[t,:] = downsample1d(alt[::-1], sig[::-1], newLayers)[::-1] # alt and sig where in descending order (relative to alt)
                        md[key] = tempSig
                    md['RangeLidar'] = np.tile(newLayers[::-1], [self.Npix,1]) # now we flip to descending order
                if 'LS' in md: # normalize att. backscatter signal
                    normConstants = -np.trapz(md['LS'], x=md['RangeLidar'], axis=1) # sign flip needed since Range is in descending order
                    md['LS'] = md['LS']/normConstants[:,None]
                for mdKey in ['LS','VBS','VExt']:
                    if mdKey in md: md[mdKey][md[mdKey] < βLowLim] = βLowLim # GRASP will not tolerate negative backscatter values
            elif self.verbose:
                print('No lidar data to convert at' + λFRMT % wvl)
        return measData

    def convertPolar2GRASP(self, measData=None):
        """ convert OSSE polarimeter radiance "measDdata" to a GRASP friendly format
            IN: if measData argument is provided we work with that, else we use self.measData (this allows chaining with other converters)
            OUT: the converted measData list is returned, self.measData will remain unchanged """
        if not measData:
            assert self.measData, 'measData must be provided or self.measData must be set!'
            measData = copy.deepcopy(self.measData) # We will return a measData in GRASP format, self.measData will remain unchanged.
        for wvl,md in zip(self.wvls, measData):
            if 'solar_azimuth' in md:
                if self.verbose: print('Converting polarimeter data to GRASP units at' + λFRMT % wvl)
                if 'I' in md:
                    md['I'] = md['I']*np.pi # GRASP "I"=R=L/FO*pi
                    if 'Q' in md.keys(): md['Q'] = md['Q']*np.pi
                    if 'U' in md.keys(): md['U'] = md['U']*np.pi
                    if 'Q' in md.keys() and 'U' in md.keys():
                        md['DOLP'] = np.sqrt(md['Q']**2+md['U']**2)/md['I']
                if 'surf_reflectance' in md:
                    md['I_surf'] = md['surf_reflectance']*np.cos(md['solar_zenith']*np.pi/180)
                    with np.errstate(invalid='ignore'): # ignore runtime warnings, bug in anaconda Numpy w/ zeros stored as float32 (see https://github.com/numpy/numpy/issues/11448)
                        if 'surf_reflectance_Q_scatplane' in md:
                            md['Q_surf'] = md['surf_reflectance_Q_scatplane']*np.cos(md['solar_zenith']*np.pi/180)
                            md['U_surf'] = md['surf_reflectance_U_scatplane']*np.cos(md['solar_zenith'])
                            if self.verbose: print('    %5f μm Q[U]_surf derived from surf_reflectance_Q[U]_scatplane (scat. plane system)' % wvl)
                        else:
                            md['Q_surf'] = md['surf_reflectance_Q']*np.cos(md['solar_zenith']*np.pi/180)
                            md['U_surf'] = md['surf_reflectance_U']*np.cos(md['solar_zenith']*np.pi/180)
                            if self.verbose: print('    %5f μm Q[U]_surf derived from surf_reflectance_Q[U] (meridian system)' % wvl)
                        if (md['I_surf'] > 0).all():
                            md['DOLP_surf'] = np.sqrt(md['Q_surf']**2+md['U_surf']**2)/md['I_surf']
                        else:
                            md['DOLP_surf'] = np.full(md['I_surf'].shape, np.nan)
                if 'Q_scatplane' in md: md['Q_scatplane'] = md['Q_scatplane']*np.pi
                if 'U_scatplane' in md: md['U_scatplane'] = md['U_scatplane']*np.pi
                md['fis'] = md['solar_azimuth'] - md['sensor_azimuth']
                md['fis'][md['fis']<0] = md['fis'][md['fis']<0] + 360  # GRASP accuracy degrades when φ<0
            elif self.verbose:
                print('No polarimeter data to convert at' + λFRMT % wvl)
        return measData

    def _loadingChecks(self, prereqCalls=False, filename=None, functionName=None):
        """ Internal method to check if data has been loaded with readPolarimeterNetCDF
            Returns true if loading was succesfull, false if file does not exist, or throws exception if another error is found
        """
        if type(prereqCalls)==str: prereqCalls = [prereqCalls] # a string was passed, make it a list
        if functionName: self.loadingCalls.append(functionName)
        if filename and not os.path.exists(filename):
            if self.verbose: print('Could not find the file %s' % filename)
            return False # file does not exist
        if not self.wvls:
            assert False, 'Wavelengths must be set in order to load data!'
        if self.invldIndPurged:
            assert False, 'You should not load additional data after purging invalid indices. If you really force loading set self.invldIndPurged=False.'
        if prereqCalls and np.any([fn not in self.loadingCalls for fn in prereqCalls]):
            prereqStr = ', '.join(prereqCalls)
            fnStr = functionName if functionName else 'the method above this method (_loadingChecks) in the stack.'
            assert False, 'The methods %s must be called before %s. Data not loaded!' % (prereqStr, fnStr)
        if self.verbose and filename: print('Processing data from %s' % filename)
        return True

    def purgeInvldInd(self, maxSZA=None, oceanOnly=False):
        """ The method will remove all invldInd from measData """
        timeInvariantVars = ['ocean_refractive_index','x', 'y', 'RangeLidar', 'lev', 'rayleigh_depol_ratio']
        if self.invldIndPurged:
            warnings.warn('You should only purge invalid indices once. If you really want to purge again set self.invldIndPurged=False.')
            return
        if maxSZA is not None:
            overSZAInds = []
            for ind, sza in enumerate(self.measData[self.vldPolarλind]['solar_zenith']):
                if not np.all(np.isnan(sza)) and np.any(sza>maxSZA): overSZAInds.append(ind)
            self.invldInd = np.append(self.invldInd, overSZAInds).astype(int)
            if self.verbose: print('%d pixels exlcuded for SZA>%4.1f°' % (len(overSZAInds), maxSZA))
        if oceanOnly:
            landInd = np.where(self.measData[0]['land_prct']>0)[0].tolist()
            self.invldInd = np.append(self.invldInd, landInd).astype(int)
            if self.verbose: print('%d pixels containing land were excluded' % len(landInd))
        self.invldInd = np.array(np.unique(self.invldInd), dtype='int')
        if self.verbose:
            allKeys = np.unique(sum([list(md.keys()) for md in self.measData],[]))
            virgin = {key:True for key in allKeys}
        for λ,md in enumerate(self.measData):
            for varName in np.setdiff1d(list(md.keys()), timeInvariantVars):
                if self.verbose and virgin[varName]: strArgs = [varName, md[varName].shape]
                md[varName] = np.delete(md[varName], self.invldInd, axis=0)
                if self.verbose and virgin[varName]:
                    strArgs.append(md[varName].shape)
                    print('Purging %s -- original shape: %s -> new shape:%s' % tuple(strArgs))
                    virgin[varName] = False
        if self.pblInds is not None: self.pblInds = np.delete(self.pblInds, self.invldInd, axis=0)
        if self.rtrvdData is not None: # we loaded state variable data that needs purging
            if self.verbose: startShape = self.rtrvdData.shape
            self.rtrvdData = np.delete(self.rtrvdData, self.invldInd)
            if self.verbose:
                shps = (startShape, self.rtrvdData.shape)
                print('Purging rtrvdData (state variables) -- orignal shape: %s -> new shape: %s' % shps)
        self.invldIndPurged = True
        self.Npix = len(self.measData[self.vldPolarλind]['dtObj']) # this assumes all λ have the same # of pixels
        if self.verbose:
            print('%d pixels with negative reflectances, bad-data-flag or excluded SZAs were purged from all variables.' % len(self.invldInd))

    def plotGlobe(self, colorVar='ssa', sizeVar='tau', indLabel=True):
        os.environ["PROJ_LIB"] = "/Users/wrespino/anaconda3/share/proj" # fix for "KeyError: 'PROJ_LIB'" bug
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 6))
        m = Basemap(projection='merc', resolution='c', llcrnrlon=-81, llcrnrlat=-57.6, urcrnrlon=65, urcrnrlat=47)
        m.bluemarble(scale=1);
        #m.shadedrelief(scale=0.2)
        #m.fillcontinents(color=np.r_[176,204,180]/255,lake_color='white')
        x, y = m(lon, lat)
        plt.scatter(x, y, c='r', s=30, facecolors='none', edgecolors='r', cmap='plasma')
        [plt.text(x0,y0,'%d'%ID, color='r') for x0,y0,ID in zip(x,y,siteID)]
        plt.title('AERONET Sites')
        cbar = plt.colorbar()
        cbar.set_label("Elevation (m)", FontSize=14)
