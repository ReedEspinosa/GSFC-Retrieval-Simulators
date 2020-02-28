#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains the definition of the osseData class, which imports OSSE NetCDF data and, if desired, exports it to a list of rslts dicts employing the convention used in GRASP_Scripts. """

import numpy as np
import glob
import re
import warnings
import os
import copy
import inspect
import datetime as dt
from MADCAP_functions import loadVARSnetCDF

class osseData(object):
    def __init__(self, fpDict=None, verbose=False):
        """
        fpDict has fields (* -> optional): 
            'polarNc4FP' (string) - full file path of polarimeter data with λ (in nm) replaced w/ %d
            'wvls'* (list of floats) - wavelengths to process in μm, not present or None -> determine them from polarNc4FP
            'dateTime'* (datetime obj.) - day and hour of measurement (min. & sec. loaded from file)
            'asmNc4FP'* (string) - gpm-g5nr.lb2.asm_Nx.YYYYMMDD_HH00z.nc4 file path (FRLAND for land percentage)
            'metNc4FP'* (string) - gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 file path (PS for surface alt.)
            'aerNc4FP'* (string) - gpm-g5nr.lb2.aer_Nv.YYYYMMDD_HH00z.nc4 file path (DELP/AIRDEN for level heights)
            'lcExt'* (string) - gpm-g5nr.lc.ext.YYYYMMDD_HH00z.%dnm path (has τ and SSA)
            'lc2Lidar'* (string) - gpm-lidar-g5nr.lc2.YYYYMMDD_HH00z.%dnm path to file w/ simulated [noise added] lidar measurements
            'stateVar'* (string) - file with state variable truth, NOT YET DEVELOPED (may ultimatly be multiple files)
            - All of the above files should contain noise free data, except lc2Lidar -
        """
        self.measData = None # measData has relevent netCDF data
        self.invldInd = np.array([])
        self.invldIndPurged = False
        self.loadingCalls = []
        self.pblTopInd = False
        if not fpDict: return
        if ('wvls' not in fpDict) or (not fpDict['wvls']): 
            self.wvls = self.λSearch(fpDict['polarNc4FP'])
        else: 
            self.wvls = fpDict['wvls']
        if 'dateTime' not in fpDict: fpDict['dateTime'] = None
        self.readPolarimeterNetCDF(fpDict['polarNc4FP'], dateTime=fpDict['dateTime'])
        if 'asmNc4FP' in fpDict: self.readasmData(fpDict['asmNc4FP'])
        if 'metNc4FP' in fpDict: self.readmetData(fpDict['metNc4FP'])
        if 'aerNc4FP' in fpDict: self.readaerData(fpDict['aerNc4FP'])
        if 'lcExt' in fpDict: self.readStateVars(fpDict)
        if 'lc2Lidar' in fpDict: self.readlidarData(fpDict['lc2Lidar'])
        self.purgeInvldInd()
        
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
        if not self.loadingChecks(prereqCalls='readPolarimeterNetCDF', filename=None): return
        rsltVars = ['fit_I','fit_Q','fit_U','fit_VBS','fit_VExt','fit_DP','fit_LS',          'vis','fis',         'sza']
        mdVars   = [    'I',    'Q',    'U',    'VBS',    'VExt',    'DP',    'LS','sensor_zenith','fis','solar_zenith']        
        measData = self.convertPolar2GRASP() # we can chain existing measData when we add LIDAR
        Nλ = len(measData)
        assert len(self.rtrvdData) == self.Npix, 'OSSE observable and state variable data structures contain a differnt number of pixels!'
        assert len(self.rtrvdData[0]['aod']) == Nλ, 'OSSE observable and state variable data structures contain a differnt number of wavelengths!'
        if NpixMax: Npix = min(self.Npix, NpixMax)
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

    def checkReturnField(self, dictObj, field, ind, defualtVal=0):
        if field in dictObj: 
            return dictObj[field][ind] 
        else:
            warnings.warn('%s was not available for OSSE pixel %d, specifying a value of %8.4f.' % (dictObj,ind,defualtVal))
            return defualtVal
        
    def readmetData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height 
            These files are in the LevB data folders and have the form gpm-g5nr.lb2.met_Nv.YYYYMMDD_HH00z.nc4 """
        if not self.loadingChecks(prereqCalls='readPolarimeterNetCDF', filename=levBFN): return
        scaleHght = 8000 # scale height (meters) for presure to alt. conversion, 8km is consistent w/ GRASP
        stndPres = 1.01e5 # standard pressure (Pa)
        minAlt = -100 # defualt GRASP build complains below -100m
        levB_data = loadVARSnetCDF(levBFN, varNames=['PS', 'PBLH'])
        maslTmp = [max(scaleHght*np.log(stndPres/PS), minAlt) for PS in levB_data['PS']]
        for md in self.measData: md['masl'] = maslTmp # suface height [m], we use GRASP atmosphere to get better agreement w/ ROT that ze below
        for md in self.measData: md['PBLH'] = levB_data['PBLH'] # PBL height in m -- looks a little fishy, should double check w/ patricia
    
    def readasmData(self, levBFN):
        """ Read in levelB data to obtain pressure and then surface altitude along w/ PBL height"""
        if not self.loadingChecks(prereqCalls='readPolarimeterNetCDF', filename=levBFN): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['FRLAND', 'FRLANDICE'])
        icePix = np.nonzero(levB_data['FRLANDICE'] > 1e-5)[0]
        self.invldInd = np.append(self.invldInd, icePix).astype(int)
        for md in self.measData: md['land_prct'] = levB_data['FRLAND'] # PBL height in m

    def readlidarData(self, levBFN):
        if self.loadingChecks(prereqCalls='readPolarimeterNetCDF')>0: return # removing this prereq requires factoring out creation of measData and setting some keys (e.g. time, dtObj) in polarimeter reader
        assert False, 'This function is under development'

    def readStateVars(self, stateVarFNs):
        """ readStateVars will read OSSE state variable file(s) and convert to the format used to store GRASP's output
            IN: dictionary stateVarFNs containing key 'lcExt' with path to lidar "truth" file
            RESULT: sets self.rtrvdData, numpy array of Npixels dicts -> rtrvdData[nPix][varKey][mode, λ]"""
        netCDFvarNames = ['tau','ssa','refr','refi','reff','vol','g','ext2Back'] # Order is important! (see for loop in rtrvdDataSetPixels())
        if self.loadingChecks(prereqCalls=['readPolarimeterNetCDF','readaerData'])>0: return
        assert 'lcExt' in stateVarFNs, 'This method requires an lcExt file path to be present'
        self.rtrvdData = np.array([{} for _ in range(self.Npix)])
        for λ,wvl in enumerate(self.wvls): # NOTE: will only use data corresponding to λ values in measData
            lcExtFN = stateVarFNs['lcExt'] % (wvl*1000)
            if os.path.exists(lcExtFN):
                if self.verbose: print('Processing data from %s' % lcExtFN)        
                lcD = loadVARSnetCDF(lcExtFN, varNames=netCDFvarNames)
                timeLoopVars = [lcD[k] for k in netCDFvarNames]
                timeLoopVars.append(self.rtrvdData)
                for rd in self.rtrvdData: 
                    self.rtrvdDataSetPixels(rd, timeLoopVars, λ) # rd is a dict, which is mutable
                    hghtInd = np.r_[[np.zeros(self.Npix)], [self.pblTopInd]].T
                    self.rtrvdDataSetPixels(rd, timeLoopVars, λ, hghtInd, '_PBL')
            elif self.verbose:
                print('No lcExt profile data found at %4.2f μm' % wvl)
            if 'stateVar' in stateVarFNs:
                assert False, 'We can not handle stateVar path yet!'   
    
    def readaerData(self, levBFN):
        """ Read in levelB data to obtain vertical layer heights """
        grav = 9.80616 # m/s^2
        if not self.loadingChecks(prereqCalls=['readPolarimeterNetCDF','readmetData'], filename=levBFN): return
        levB_data = loadVARSnetCDF(levBFN, varNames=['AIRDEN', 'DELP']) # air density [kg/m^3], pressure thickness [Pa]
        self.pblTopInd = np.full(self.Npix, np.nan)
        for k,(airdens,delp) in enumerate(zip(levB_data['AIRDEN'], levB_data['DELP'])):
            ze = (delp[::-1]/airdens[::-1]*grav).cumsum()[::-1] # profiles run top down so we reverse order for cumsum
            rng = (np.r_[ze[1::],0] + ze)/2
            if not np.isclose(self.measData[0]['masl'][k], ze[-1], atol=500):
                msg = 'At pixel k=%d exp. atmosphere derived masl (%dm) was more than 500m off lower level edge (%dm).'
                warnings.warn(msg % (k, self.measData[0]['masl'][k], ze[-1]))
            self.pblTopInd = np.argmin(np.abs(self.measData[0]['PBL'][k] - rng))
            for md in self.measData: # i.e. k==0
                if 'range' not in md:
                    md['range'] = np.full([self.Npix, len(delp)], np.nan)
                md['range'][k,:] = rng

    def rtrvdDataSetPixels(self, rd, timeLoopVars, λ, hghtInd=None, km=''):
        hghtSlice = []
        for h in hghtInd: # hghtInd (Npixels x 2) vector denoating top, bottom (lev=0 is TOA in G5NR) lev index of PBL
            hghtSlice.append(slice(h[0],h[1]+1) if hghtInd else slice(None))
        timeLoopVars.append(hghtSlice)
        firstλ = 'aod'+km in rd
        if firstλ:     
            for k in ['aod'+km,'ssa'+km,'n'+km,'k'+km,'g'+km, 'LidarRatio'+km]: # spectrally dependent vars only, reff doesn't need allocation
                rd[k] = np.full(len(self.wvls), np.nan)
        for τ,ω,n,k,rEff,V,g,S,rd,hSlc in zip(*timeLoopVars): # loop over each pixel and vertically average
            rd['aod'+km][λ] = τ[hSlc].sum()
            rd['ssa'+km][λ] = np.sum(τ[hSlc]*ω[hSlc])/rd['aod'+km][λ] # ω=Σβh/Σαh & ωh*αh=βh => ω=Σωh*αh/Σαh
            rd['n'+km][λ] = np.sum(τ[hSlc]*n[hSlc])/rd['aod'+km][λ] # n is weighted by τ (this is a non-physical quantity)
            rd['k'+km][λ] = np.sum(τ[hSlc]*k[hSlc])/rd['aod'+km][λ] # k is weighted by τ (this is a non-physical quantity)
            rEffTot = np.sum(V[hSlc])/np.sum(V[hSlc]/rEff[hSlc]) # see bottom of this method for derivation
            if firstλ:
                rd['rEff'+km] = rEffTot
            elif not np.isclose(rd['rEff'+km], rEffTot):
                warnings.warn('The current rEff (%8.5fμm) differs from the first rEff (%8.5fμm) derived.' % (rEffTot, rd['rEff'+km][λ]))
            rd['g'+km][λ] = np.sum(τ[hSlc]*ω[hSlc]*g[hSlc])/np.sum(τ[hSlc]*ω[hSlc]) # see bottom of this method for derivation
            rd['LidarRatio'+km][λ] = np.sum(τ[hSlc])/np.sum(τ[hSlc]/S[hSlc]) # S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh)
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
        if not self.loadingChecks(): return # we don't pass filename b/c we only have a pattern at this point
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
                self.measData[i] = loadVARSnetCDF(radianceFN, varNames)
                tShft = self.measData[i]['time'] if 'time' in self.measData[i] else 0
                self.measData[i]['dtObj'] = [dateTime + dt.timedelta(seconds=int(ts)) for ts in tShft]
                invldIndλ = np.nonzero((self.measData[i]['I']<0).any(axis=1))[0]
                self.invldInd = np.append(self.invldInd, invldIndλ).astype(int) # only take points w/ I>0 at all wavelengths & angles
            elif self.verbose:
                print('No polarimeter data found at %4.2f μm' % wvl)
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
    
    def loadingChecks(self, prereqCalls=False, filename=None):
        """ Internal method to check if data has been loaded with readPolarimeterNetCDF """
        if type(prereqCalls)==str: prereqCalls = [prereqCalls] # a string was passed, make it a list
        callingFun = inspect.getouterframes(inspect.currentframe(), 2)
        self.loadingCalls.append(callingFun)
        if filename and not os.path.exists(filename): 
            warnings.warn('Could not find the file %s' % filename)
            return 3
        if not self.wvls:
            warnings.warn('Wavelengths must be set in order to load data!')
            return 4
        if self.invldIndPurged:
            warnings.warn('You can not load any more data after purging invalid indices once. If you really force loading set invldIndPurged=False.')
            return 1
        if prereqCalls and prereqCalls not in self.loadingCalls:
            prereqStr = ', '.join(prereqCalls)
            warnings.warn('The methods %s must be called before %s:' % (prereqStr,callingFun))
            return 2            
        if self.verbose and filename: print('Processing data from %s' % filename)
        return 0
      
    def purgeInvldInd(self):
        """ The method will remove all invldInd from measData """
        if self.invldIndPurged:
            warnings.warn('You should only purge invalid indices once. If you really want to purge again set invldIndPurged=False.')
            return
        self.invldInd = np.array(np.unique(self.invldInd), dtype='int')
        for md in self.measData:
            for varName in np.setdiff1d(list(md.keys()), ['x', 'y', 'lev']):
                md[varName] = np.delete(md[varName], self.invldInd, axis=0)
        if self.pblTopInd: self.pblTopInd = np.delete(self.pblTopInd, self.invldInd)
        if self.rtrvdData: self.rtrvdData = np.delete(self.rtrvdData, self.invldInd)
        self.invldIndPurged = True
        if self.verbose: 
            print('%d pixels with negative or bad-data-flag reflectances were purged.' % len(self.invldInd))