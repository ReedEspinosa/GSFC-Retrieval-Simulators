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

GRAV = 9.80616 # m/s^2
SCALE_HGHT = 8000 # scale height (meters) for presure to alt. conversion, 8 km is consistent w/ GRASP
STAND_PRES = 1.01e5 # standard pressure (Pa)
MIN_ALT = -100 # defualt GRASP build complains below -100m


class osseData(object):
    def __init__(self, fpDict=None, verbose=False):
        """
        fpDict is a dictionary with fields (* -> optional):
            'polarNc4FP' (string) - full file path of polarimeter data with λ (in nm) replaced w/ %d
            'wvls'* (list of floats) - wavelengths to process in μm, not present or None -> determine them from polarNc4FP
            'verbose'* (logical) - verbose output from all methods of the class
            'dateTime'* (datetime obj.) - day and hour of measurement (min. & sec. loaded from file)
            'asmNc4FP'* (string) - gpm-g5nr.lb2.asm_Nx.YYYYMMDD_HH00z.nc4 file path (FRLAND for land percentage)
            'metNc4FP'* (string) - gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 file path (PS for surface alt.)
            'aerNc4FP'* (string) - gpm-g5nr.lb2.aer_Nv.YYYYMMDD_HH00z.nc4 file path (DELP/AIRDEN for level heights)
            'lcExt'* (string)    - gpm-g5nr.lc.ext.YYYYMMDD_HH00z.%dnm path (has τ and SSA)
            'lc2Lidar'* (string) - gpm-lidar-g5nr.lc2.YYYYMMDD_HH00z.%dnm path to file w/ simulated [noise added] lidar measurements
            'stateVar'* (string) - file with state variable truth, NOT YET DEVELOPED (may ultimatly be multiple files)
        - All of the above files should contain noise free data, except lc2Lidar -
        Note: buildFpDict() method below will setup fpDict with files mirroring paths on DISCOVER
        If fpDict is not provided when the object is initialized the data can be loaded with the loadAllData() method
        """
        self.measData = None # measData has observational netCDF data
        self.rtrvdData = None # rtrvdData has state variables from netCDF data
        self.invldInd = np.array([])
        self.invldIndPurged = False
        self.loadingCalls = []
        self.pblTopInd = None
        self.verbose = verbose
        self.fpDict = fpDict
        if self.fpDict: self.loadAllData()

    def loadAllData(self, fpDict=None):
        """Loads NetCDF OSSE data into memory, see buildFpDict below for variable descriptions"""
        if fpDict: self.fpDict = fpDict
        assert self.fpDict, 'fpDict has not been defined, data can not be loaded'
        if 'verbose' in self.fpDict: self.verbose = self.fpDict['verbose']
        if ('wvls' not in self.fpDict) or (not self.fpDict['wvls']):
            self.wvls = self.λSearch(self.fpDict['polarNc4FP'])
        else:
            self.wvls = self.fpDict['wvls']
        if 'dateTime' not in self.fpDict: self.fpDict['dateTime'] = None
        assert self._safe2load('polarNc4FP'), 'This class currently requires a polarNc4FP file to be present.'
        self.readPolarimeterNetCDF(self.fpDict['polarNc4FP'], dateTime=self.fpDict['dateTime'])
        if self._safe2load('asmNc4FP'): self.readasmData(self.fpDict['asmNc4FP'])
        if self._safe2load('metNc4FP'): self.readmetData(self.fpDict['metNc4FP'])
        if self._safe2load('aerNc4FP'): self.readaerData(self.fpDict['aerNc4FP'])
        self.readStateVars(self.fpDict) # handles its own checks since wavelength is required to determine filename
        if self._safe2load('lc2Lidar'): self.readlidarData(self.fpDict['lc2Lidar'])
        self.purgeInvldInd()

    def buildFpDict(self, osseDataPath, orbit, year, month, day=1, hour=0, random=False):
        """
        returns fpDict dictionary with filepath fields shown described in __init__ docstring
        IN: orbit - 'ss450' or 'gpm'
            year, month, day - integers specifying data of pixels to load
            hour - integer from 0 to 23 specifying the starting hour of pixels to load
            random - logical, use 1e4 randomly selected pixels for that month (day & hour not needed if random=true)
        """
        assert not osseDataPath is None, 'osseDataPath, Year, month and orbit must be provided to build fpDict'
        tmStr = '%04d%02d%02d_%02d00z' % (year, month, day, hour)
        if random:
            tmStr = 'random.'+tmStr
            dtTple = (year, month)
            pathFrmt = os.path.join(osseDataPath, orbit.upper(), 'Level%s', 'Y%04d','M%02d', '%s')
        else:
            dtTple = (year, month, day)
            pathFrmt = os.path.join(osseDataPath, orbit.upper(), 'Level%s', 'Y%04d','M%02d', 'D%02d', '%s')
        self.fpDict = {
            'polarNc4FP': pathFrmt % (('C',)+dtTple+(orbit+'-polar07-g5nr.lc.vlidort.'+tmStr+'_%dd00nm.nc4',)),
            'asmNc4FP': pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.asm_Nx.'+tmStr+'.nc4',)),
            'metNc4FP': pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.met_Nv.'+tmStr+'.nc4',)),
            'aerNc4FP': pathFrmt % (('B',)+dtTple+(orbit+'-g5nr.lb2.aer_Nv.'+tmStr+'.nc4',)),
            'lcExt': pathFrmt % (('C',)+dtTple+(orbit+'-g5nr.lc.ext.'+tmStr+'.%dnm.nc4',)),
            'lc2Lidar': pathFrmt % (('D',)+dtTple+(orbit+'-g5nr.lc2.'+tmStr+'.%dnm.LIDAR.nc4',)), # will need a str replace, e.g. VAR.replace('LIDAR', 'LIDAR09')
            'savePath': pathFrmt % (('E',)+dtTple+(orbit+'-g5nr.leV%02d.GRASP.%s.%s.'+tmStr+'.pkl',)) # % (vrsn, yamlTag, archName) needed from calling function
        }
        return self.fpDict
    
    def _safe2load(self, fileKey):
        if fileKey not in self.fpDict: return False
        if not os.path.isfile(self.fpDict[fileKey]): return False
        return True

    def λSearch(self, fpDict):
        wvls = []
        posKeys = ['polarNc4FP', 'lcExt', 'lc2Lidar'] # we only loop over these specific keys (if they exist)
        for levCFN in [fpDict[key] for key in posKeys if key in fpDict]:
            levCfiles = glob.glob(levCFN.replace('%d','[0-9]*'))
            for fn in levCfiles:
                mtch = re.match(levCFN.replace('%d','([0-9]+)?'), fn)
                wvlVal = float(mtch.group(1))/1000
                if mtch and wvlVal not in wvls: wvls.append(wvlVal)
        wvls = np.sort(wvls)
        if wvls and self.verbose:
            print('The following wavelengths were found:', end=" ")
            print('%s μm' % ', '.join([str(λ) for λ in wvls]))
        elif not wvls:
            warnings.warn('No wavelengths found for the pattern %s' % levCFN)
        return wvls

    def osse2graspRslts(self, NpixMax=None):
        """ osse2graspRslts will convert that data in measData to the format used to store GRASP's output
                IN: NpixMax -> no more than this many pixels will be returned in rslts list (primarly for testing)
                OUT: a list of Npixels dicts with all the keys mapping measData as rslts[nPix][var][nAng, λ]
                    keys in output dictionary:
                NOTE: GRASP can only use one SZA value & simulator just takes 1st value. Maybe just make them all equal to the average? """
        assert self.measData, 'self.measData must be set (e.g through readPolarimeterNetCDF()) before calling this method!'
        rsltVars = ['fit_I','fit_Q','fit_U','fit_VBS','fit_VExt','fit_DP','fit_LS',          'vis','fis',         'sza']
        mdVars   = [    'I',    'Q',    'U',    'VBS',    'VExt',    'DP',    'LS','sensor_zenith','fis','solar_zenith']
        measData = self.convertPolar2GRASP() # we can chain existing measData when we add LIDAR
        Nλ = len(measData)
        assert len(self.rtrvdData) == self.Npix, \
            'OSSE observable and state variable data structures contain a differnt number of pixels!'
        assert ('aod' not in self.rtrvdData[0]) or (len(self.rtrvdData[0]['aod']) == Nλ), \
            'OSSE observable and state variable data structures contain a differnt number of wavelengths!'
        Npix = min(self.Npix, NpixMax) if NpixMax else self.Npix
        rslts = []
        for k,rd in enumerate(self.rtrvdData[0:Npix]): # loop over pixels
            rslt = dict()
            for rv,mv in zip(rsltVars,mdVars): # loop over potential variables
                for l, md in enumerate(measData): # loop over λ -- HINT: we assume all λ have the same # of measurements for each msType
                    if mv in md: # this is an observable
                        if rv not in rslt: rslt[rv] = np.empty([len(md[mv][k,:]), Nλ])*np.nan # len(md[mv][k,:]) will change between imagers(Nang) & LIDAR(Nhght)
                        rslt[rv][:,l] = md[mv][k,:]
                        if k==0 and self.verbose: print('%s at %4.2f found in OSSE observables' % (mv, self.wvls[l]))
                if k==0 and rv not in rslt and self.verbose: print('%s NOT found in OSSE data' % mv)
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
                frmStr = 'Converted pixel #%05d (%s), [LAT:%6.2f°, LON:%6.2f°], %3.0f%% land, asl=%4.0f m'
                ds = rslt['datetime'].strftime("%d/%m/%Y %H:%M:%S")
                sza, Δsza = self.angleVals(rslt['sza'])
                vφa, Δvφa = self.angleVals(rslt['fis'])
                vza, Δvza = self.angleVals(rslt['vis']*(1-2*(rslt['fis']>180)))
                frmStr = frmStr + ', θs=%4.1f° (±%4.1f°), φ=%4.1f° (±%4.1f°), θv=%4.1f° (±%4.1f°)'
                print(frmStr % (k, ds, rslt['latitude'], rslt['longitude'], 100*rslt['land_prct'], \
                                measData[0]['masl'][k], sza, Δsza, vφa, Δvφa, vza, Δvza))
        return rslts

    def angleVals(self, keyVal):
        angle = np.mean(keyVal)
        Δangle = np.abs(keyVal-angle).max()
        return angle, Δangle

    def checkReturnField(self, dictObj, field, ind, defualtVal=0):
        if field in dictObj:
            return dictObj[field][ind]
        else:
            warnings.warn('%s was not available for OSSE pixel %d, specifying a value of %8.4f.' % (dictObj,ind,defualtVal))
            return defualtVal

    def readmetData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height
            These files are in the LevB data folders and have the form gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 """
        call1st = 'readPolarimeterNetCDF'
        if not self.loadingChecks(prereqCalls=call1st, filename=levBFN, functionName='readmetData'): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['PS', 'PBLH'], verbose=self.verbose)
        maslTmp = np.array([max(SCALE_HGHT*np.log(STAND_PRES/PS), MIN_ALT) for PS in levB_data['PS']])
        for md in self.measData: md['masl'] = maslTmp # suface height [m], we use GRASP atmosphere to get better agreement w/ ROT that ze below
        for md in self.measData: md['PBLH'] = levB_data['PBLH'] # PBL height in m -- looks a little fishy, should double check w/ patricia

    def readasmData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height"""
        call1st = 'readPolarimeterNetCDF'
        if not self.loadingChecks(prereqCalls=call1st, filename=levBFN, functionName='readasmData'): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['FRLAND', 'FRLANDICE'], verbose=self.verbose)
        icePix = np.nonzero(levB_data['FRLANDICE'] > 1e-5)[0]
        self.invldInd = np.append(self.invldInd, icePix).astype(int)
        for md in self.measData: md['land_prct'] = levB_data['FRLAND'] # PBL height in m

    def readlidarData(self, levBFN):
        call1st = 'readPolarimeterNetCDF'
        if not self.loadingChecks(prereqCalls=call1st, functionName='readlidarData'): return # removing this prereq requires factoring out creation of measData and setting some keys (e.g. time, dtObj) in polarimeter reader
        assert False, 'This function is under development'

    def readStateVars(self, stateVarFNs):
        """ readStateVars will read OSSE state variable file(s) and convert to the format used to store GRASP's output
            IN: dictionary stateVarFNs containing key 'lcExt' with path to lidar "truth" file
            RESULT: sets self.rtrvdData, numpy array of Npixels dicts -> rtrvdData[nPix][varKey][mode, λ]"""
        self.rtrvdData = np.array([{} for _ in range(self.Npix)])
        if 'lcExt' not in stateVarFNs: 
            if self._safe2load('stateVar'): print('stateVar file can not be read without an lcExt file!')
            if self.verbose: print('lcExt filename not provided, no state variable data read.')
            return False
        netCDFvarNames = ['tau','ssa','refr','refi','reff','vol','g','ext2back'] # Order is important! (see for loop in rtrvdDataSetPixels())
        call1st = ['readPolarimeterNetCDF','readaerData']
        if not self.loadingChecks(prereqCalls=call1st, functionName='readStateVars'): return
        for λ,wvl in enumerate(self.wvls): # NOTE: will only use data corresponding to λ values in measData
            lcExtFN = self.fpDict['lcExt'] % (wvl*1000)
            if os.path.exists(lcExtFN):
                if self.verbose: print('Processing data from %s' % lcExtFN)
                lcD = loadVARSnetCDF(lcExtFN, varNames=netCDFvarNames, verbose=self.verbose)
                for t in range(lcD['reff'].shape[0]): # HACK (next 4 lines) until we better understand zeros and NaNs in OSSE output
                    for lev in range(lcD['reff'].shape[1]):
                        if np.isnan(lcD['reff'][t,lev]) or lcD['reff'][t,lev] < 1e-12:
                           lcD['reff'][t,lev] = 1e-7 if lev==0 else lcD['reff'][t,lev-1]
                timeLoopVars = [lcD[key] for key in netCDFvarNames]
                timeLoopVars.append(self.rtrvdData)
                self.rtrvdDataSetPixels(timeLoopVars, λ) # rtrvdData [out] is a dict, which is mutable
                hghtInd = np.r_[[np.zeros(self.Npix)], [self.pblTopInd]].T
                self.rtrvdDataSetPixels(timeLoopVars, λ, hghtInd, '_PBL')
            elif self.verbose:
                print('No lcExt profile data found at %4.2f μm' % wvl)
            if self._safe2load('stateVar'):
                assert False, 'We can not handle stateVar path yet!'

    def readaerData(self, levBFN):
        """ Read in levelB data to obtain vertical layer heights """

        preReqs = ['readPolarimeterNetCDF','readmetData']
        if not self.loadingChecks(prereqCalls=preReqs, filename=levBFN, functionName='readaerData'): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['AIRDENS', 'DELP'], verbose=self.verbose) # air density [kg/m^3], pressure thickness [Pa]
        self.pblTopInd = np.full(self.Npix, np.nan)
        for k,(airdens,delp) in enumerate(zip(levB_data['AIRDENS'], levB_data['DELP'])):
            ze = (delp[::-1]/airdens[::-1]/GRAV).cumsum()[::-1] # profiles run top down so we reverse order for cumsum
            rng = (np.r_[ze[1::],0] + ze)/2
            self.pblTopInd[k] = np.argmin(np.abs(self.measData[0]['PBLH'][k] - rng))
            for md in self.measData:
                if 'range' not in md: # i.e. k==0
                    md['range'] = np.full([self.Npix, len(delp)], np.nan)
                md['range'][k,:] = rng

    def rtrvdDataSetPixels(self, timeLoopVars, λ, hghtInd=None, km=''):
        """ hghtInd (Npixels x 2) vector denoating top, bottom (lev=0 is TOA in G5NR) lev index of PBL """
        if hghtInd is None: hghtInd = np.repeat(slice(None), self.Npix)
        firstλ = 'aod'+km not in timeLoopVars[-1][0]
        for τ,ω,n,k,rEff,V,g,S,rd,hSlc in zip(*timeLoopVars, hghtInd): # loop over each pixel and vertically average
            if not type(hSlc) is slice: hSlc = slice(int(hSlc[0]), int(hSlc[1]+1))
            if firstλ:
                for varKey in ['aod'+km,'ssa'+km,'n'+km,'k'+km,'g'+km, 'LidarRatio'+km]: # spectrally dependent vars only, reff doesn't need allocation
                    rd[varKey] = np.full(len(self.wvls), np.nan)
            if np.all(~np.isnan(rEff)):
                rd['aod'+km][λ] = τ[hSlc].sum()
                rd['ssa'+km][λ] = np.sum(τ[hSlc]*ω[hSlc])/rd['aod'+km][λ] # ω=Σβh/Σαh & ωh*αh=βh => ω=Σωh*αh/Σαh
                rd['n'+km][λ] = np.sum(τ[hSlc]*n[hSlc])/rd['aod'+km][λ] # n is weighted by τ (this is a non-physical quantity)
                rd['k'+km][λ] = np.sum(τ[hSlc]*k[hSlc])/rd['aod'+km][λ] # k is weighted by τ (this is a non-physical quantity)
                rEff[rEff < 1e-12] = 1e-12
                rEffTot = np.sum(V[hSlc])/np.sum(V[hSlc]/rEff[hSlc]) # see bottom of this method for derivation
                if firstλ:
                    rd['rEff'+km] = rEffTot
                elif not np.isclose(rd['rEff'+km], rEffTot):
                    warnings.warn('The current rEff (%8.5f μm) differs from the first rEff (%8.5f μm) derived.' % (rEffTot, rd['rEff'+km]))
                rd['g'+km][λ] = np.sum(τ[hSlc]*ω[hSlc]*g[hSlc])/np.sum(τ[hSlc]*ω[hSlc]) # see bottom of this method for derivation
                rd['LidarRatio'+km][λ] = np.sum(τ[hSlc])/np.sum(τ[hSlc]/S[hSlc]) # S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh)
            else:
                print('¡¡¡¡NANS IN REFF!!!!')
        """ --- reff vertical averaging (reff_h=Rh, # conc.=Nh, vol_conc.=Vh, height_ind=h) ---
            reff = ∫ ΣNh*r^3*dr/∫ ΣNh*r^2*dr = 3/4/π*ΣVh/∫ Σnh*r^2*dr
            Rh = (3/4/π*Vh)/∫ nh*r^2*dr -> reff = 3/4/π*ΣVh/(Σ(3/4/π*Vh)/Rh) = ΣVh/Σ(Vh/Rh)
            --- g vertical averaging (normalized PF = P11[θ], absolute PF = F11[θ]) ---
            P11[θ] = F11[θ]/β = Σ(τh*ωh*P11h[θ])/Στh*ωh -> g = ∫ P11[θ]...dθ = ∫ Σ(τh*ωh*P11h[θ])/Στh*ωh...dθ
            g = Σ(τh*ωh*∫P11h[θ]...dθ)/Στh*ωh = Σ(τh*ωh*gh)/Σ(τh*ωh) """

    def readPolarimeterNetCDF(self, radianceFNfrmtStr, varNames=None, dateTime=None):
        """ readPolarimeterNetCDF will read a simulated polarimeter data from VLIDORT OSSE
                IN: radianceFNfrmtStr is a string with full path to OSSE files w/ %d replacing λ values
                    varNames* is list of a subset of variables to load
                OUT: will set self.measData and add to invldInd, no data returned """
        if not self.loadingChecks(functionName='readPolarimeterNetCDF'): return # we don't pass filename b/c we only have a pattern at this point
        if not dateTime:
            try:
                dateStrMtch = re.search('[0-9]{8}_[0-9]{4}z', radianceFNfrmtStr)
                dateTime = dt.datetime.strptime(dateStrMtch.group(0), "%Y%m%d_%H%Mz")
            except:
                warnings.warn('Using Jan 01, 2000 as date, could not parse polarimeter filename format string (radianceFNfrmtStr = %s)' % radianceFNfrmtStr)
                dateTime = dt.datetime.strptime('20000101', "%Y%m%d")
        self.measData = [{} for _ in range(len(self.wvls))]
        for i,wvl in enumerate(self.wvls):
            radianceFN = radianceFNfrmtStr % (wvl*1000)
            if os.path.exists(radianceFN): # load data and check for valid indices (I>=0)
                if self.verbose: print('Processing data from %s' % radianceFN)
                self.measData[i] = loadVARSnetCDF(radianceFN, varNames, verbose=self.verbose)
                tShft = self.measData[i]['time'] if 'time' in self.measData[i] else 0
                self.measData[i]['dtObj'] = np.array([dateTime + dt.timedelta(seconds=int(ts)) for ts in tShft])
                with np.errstate(invalid='ignore'): # we need this b/c we will be comaring against NaNs
                    invldBool = np.logical_or(np.isnan(self.measData[i]['I']), \
                                              self.measData[i]['I'] < 0 \
                                              )
                invldIndλ = invldBool.any(axis=1).nonzero()[0]
                self.invldInd = np.append(self.invldInd, invldIndλ).astype(int) # only take points w/ I>0 at all wavelengths & angles
            elif self.verbose:
                print('No polarimeter data found at %4.2f μm' % wvl)
        self.Npix = len(self.measData[0]['dtObj']) # this assumes all λ have the same # of pixels

    def convertPolar2GRASP(self, measData=None):
        """ convert OSSE "measDdata" to a GRASP friendly format
            IN: if measData argument is provided we work with that, else we use self.measData (this allows chaining with other converters)
            OUT: the converted measData list is returned, self.measData will remain unchanged """
        if not measData:
            assert self.measData, 'measData must be provided or self.measData must be set!'
            measData = copy.deepcopy(self.measData) # We will return a measData in GRASP format, self.measData will remain unchanged.
        assert 'solar_azimuth' in measData[0], \
            'solar_azimuth field required for conversion but it was not found in measData. Was polarimeter data loaded propertly?'
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

    def loadingChecks(self, prereqCalls=False, filename=None, functionName=None):
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
            fnStr = functionName if functionName else 'the method above this method (loadingChecks) in the stack.'
            assert False, 'The methods %s must be called before %s. Data not loaded!' % (prereqStr, fnStr)
        if self.verbose and filename: print('Processing data from %s' % filename)
        return True

    def purgeInvldInd(self):
        """ The method will remove all invldInd from measData """
        timeInvariantVars = ['ocean_refractive_index','x', 'y', 'lev', 'rayleigh_depol_ratio']
        if self.invldIndPurged:
            warnings.warn('You should only purge invalid indices once. If you really want to purge again set self.invldIndPurged=False.')
            return
        self.invldInd = np.array(np.unique(self.invldInd), dtype='int')
        for λ,md in enumerate(self.measData):
            for varName in np.setdiff1d(list(md.keys()), timeInvariantVars):
                if self.verbose and λ==0: strArgs = [varName, md[varName].shape]
                md[varName] = np.delete(md[varName], self.invldInd, axis=0)
                if self.verbose and λ==0:
                    strArgs.append(md[varName].shape)
                    print('Purging %s -- original shape: %s -> new shape:%s' % tuple(strArgs))
        if self.pblTopInd is not None: self.pblTopInd = np.delete(self.pblTopInd, self.invldInd)
        if self.rtrvdData is not None: # we loaded state variable data that needs purging
            if self.verbose: startShape = self.rtrvdData.shape
            self.rtrvdData = np.delete(self.rtrvdData, self.invldInd)
            if self.verbose:
                shps = (startShape, self.rtrvdData.shape)
                print('Purging rtrvdData (state variables) -- orignal shape: %s -> new shape: %s' % shps)
        self.invldIndPurged = True
        self.Npix = len(self.measData[0]['dtObj']) # this assumes all λ have the same # of pixels
        if self.verbose:
            print('%d pixels with negative or bad-data-flag reflectances were purged from all variables.' % len(self.invldInd))
        
