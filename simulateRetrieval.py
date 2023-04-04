#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pickle
import shutil
import os
import warnings
import datetime as dt
import numpy as np
from scipy.stats import norm
import sys
from os import path
GRASP_Python_Path = path.join(path.dirname(path.dirname(__file__)), "GSFC-GRASP-Python-Interface")
if GRASP_Python_Path not in sys.path: sys.path.append(GRASP_Python_Path)
import runGRASP as rg
import miscFunctions as ms
from glob import glob

class simulation(object):
    def __init__(self, nowPix=None, picklePath=None, standardizePSD=True):
        """
        nowPix – dummy pixel object with measurement geometry, wavelengths, etc. (i.e. info in a GRASP SDATA)
        picklePath – path to pickle file from a previously run simulation (either this or nowPix must be provided)
        standardizePSD – regrid all (fwd and back) particle size disruptions to a common set of radii
        """
        assert not (nowPix and picklePath), 'Either nowPix or picklePath should be provided, but not both.'
        if picklePath: self.loadSim(picklePath, standardizePSD)
        if nowPix is None:
            self.nowPix = None
            return
        if not isinstance(nowPix, (list, np.ndarray)):
            nowPix = [nowPix]
        for pix in nowPix:
            assert np.all([np.all(np.diff(mv['meas_type'])>0) for mv in pix.measVals]), 'nowPix.measVals[l][\'meas_type\'] must be in ascending order at each l'
        self.nowPix = copy.deepcopy(nowPix) # we will change this, bad form not to make our own copy
        self.rsltBck = None
        self.rsltFwd = None

    def runSim(self, fwdData, bckYAML, Nsims=1, maxCPU=4, maxT=None, binPathGRASP=None, savePath=None,
               lightSave=False, intrnlFileGRASP=None, releaseYAML=True, rndIntialGuess=False,
               dryRun=False, workingFileSave=False, fixRndmSeed=False, radianceNoiseFun=None, verbose=False, delTempFiles=False):
        """
        <> runs the simulation for given set of simulated and inversion conditions <>
        fwdData -> yml file path for GRASP fwd model OR graspYAML object OR a list of either of the prior two OR "results style" list of dicts
                        if a list of YAML files, len(self.nowPix) can be unity OR len(fwdData) [if latter, they pair one-to-one]
        bckYAML -> yml file path for GRASP inversion OR graspYAML object [only one allowed here, no lists]
        Nsims -> number of noise perturbations applied to each fwd model [Nsims*len(fwdData) total retrievals]
        maxCPU -> the retrieval load will be spread accross maxCPU processes
        binPathGRASP -> path to GRASP binary, if None default from graspRun is used
        savePath -> path to save pickle w/ simulated retrieval results, lightSave -> remove PM and extinction profile data to save space
        intrnlFileGRASP -> alternative path to GRASP kernels, overwrites value in YAML files
        releaseYAML=True -> auto adjust back yaml Nλ and number of vertical bins to match the forward simulated data
        rndIntialGuess=True -> overwrite initial guesses in bckYAML w/ uniformly distributed random values between min & max
        dryRun -> run foward model and then return noise added graspDB object, without performing the retrievals
        workingFileSave -> create ZIP with the GRASP SDATA, YAML and Output files used in the run, saved to savePath + .zip
        fixRndmSeed -> Use same random seed for the measurement noise added (each pixel will have identical noise values)
                            Only works if nowPix.measVals[n]['errorModel'] uses the `random` module to generate noise for all n
        radianceNoiseFun -> a function with 1st arg λ (μm), 2nd arg rslt dict & 3rd arg verbose bool to map rslt fit_I/Q/U to retrieval (back) SDATA
                                See addError() at bottom of architectureMap in ACCP folding of MADCAP scripts for an example
                                This option will override an error model in nowPix; set to None add no noise to OSSE polarimeter
        delTempFiles -> The flag deletes the temporary files and directories created by graspRun. 
                        In the DISCOVER server, there is a limit on the number of files that can be stored at a time in the storage.
                        If you are running a large number of simulations, this flag should be set to 'True'. 
        """
        if fixRndmSeed and not rndIntialGuess:
            warnings.warn('Identical noise values and initial guess used in each pixel, repeating EXACT same retrieval %d times!' % Nsims)
        if fixRndmSeed and maxCPU<Nsims:
            warnings.warn('A single YAML is used on multiple identical pixels with identical noise values => repeating EXACT same retrieval more than once! Avoid by setting fixRndmSeed=False OR maxCPU>=Nsims.')
        # ADAPT fwdData/RUN THE FOWARD MODEL
        if not type(fwdData) == list: fwdData = [fwdData]
        isYaml = lambda obj : (type(obj) == str and obj[-3:] == 'yml') or type(obj)==rg.graspYAML
        if isYaml(fwdData[0]): # we are using GRASP's fwd model
            assert self.nowPix is not None, 'A dummy pixel (nowPix) with an error function (addError) is required to run the simulation.'
            if verbose: print('Calculating forward model "truth"...')
            gObjFwd = []
            for i,fd in enumerate(fwdData):
                pix = self.nowPix[0] if len(self.nowPix)==1 else self.nowPix[i]
                gObjFwd.append(rg.graspRun(fd))
                gObjFwd[-1].addPix(pix)
            gDBFwd = rg.graspDB(gObjFwd, maxCPU=maxCPU)
            self.rsltFwd = gDBFwd.processData(maxCPU, binPathGRASP,
                                              krnlPathGRASP=intrnlFileGRASP,
                                              rndGuess=False)[0]
            assert len(self.rsltFwd)==len(fwdData), 'Forward calucation was not fully successfull, halting the simulation.'
        elif type(fwdData[0]) == dict: # likely OSSE from netCDF
            self.rsltFwd = fwdData
        else:
            assert False, 'Unrecognized data type, fwdModelYAMLpath should be path to YAML file or a DICT!'
        if self.nowPix is None: self.nowPix = [rg.pixel()] # nowPix is optional argument in OSSE case, make sure it exists <<<<<
        loopInd = np.tile(np.r_[0:len(self.rsltFwd)], Nsims)
        if verbose: print('Forward model "truth" obtained')
        # ADD NOISE AND PERFORM RETRIEVALS
        if verbose: print('Inverting noised-up measurements...')
        gObjBck = rg.graspRun(bckYAML, releaseYAML=releaseYAML, verbose=False) # verbose=False -> we won't see path of temp, pre-gDB graspRun
        if fixRndmSeed: strtSeed = np.random.randint(low=0, high=2**32-1)
        localVerbose = verbose
        for tOffset, i in enumerate(loopInd): # loop over each simulated pixel, later split up into maxCPU calls to GRASP
            if fixRndmSeed: np.random.seed(strtSeed) # reset to same seed, adding same noise to every pixel
            nowPix = self.nowPix[0] if len(self.nowPix)==1 else self.nowPix[i]
            nowPix.populateFromRslt(self.rsltFwd[i], radianceNoiseFun=radianceNoiseFun, verbose=localVerbose)
            if len(np.unique(loopInd)) != len(loopInd): # we are using the same rsltFwd dictionary more than once
                nowPix.dtObj = nowPix.dtObj + dt.timedelta(seconds=tOffset) # increment hour otherwise GRASP will whine
            gObjBck.addPix(nowPix) # addPix performs a deepcopy on nowPix, won't be impact by next iteration through loopInd
            localVerbose = False # verbose output for just one pixel should be sufficient
        # TODO: the following should go after gDB.processData, and only include cases for which rsltBck was successfully derived
        if len(self.rsltFwd)>1: self.rsltFwd = np.tile(self.rsltFwd, Nsims) # make len(rsltBck)==len(rsltFwd)... very memory inefficient though so only do it in more complicated len(self.rstlFwd)>1 cases
        gDB = rg.graspDB(gObjBck, maxCPU=maxCPU, maxT=maxT)
        if not dryRun:
            self.rsltBck, failedRuns = gDB.processData(maxCPU, binPathGRASP,
                                                       krnlPathGRASP=intrnlFileGRASP,
                                                       rndGuess=rndIntialGuess)
            assert len(self.rsltBck)>0, 'No inversion output could be read, halting the simulation (no data was saved).'
            if failedRuns.any():  self.rsltFwd = [rf for rf,failed in zip(self.rsltFwd, failedRuns) if failed==False] # remove runs from rsltFwd for which the inversion was not succesful
            if 'pixNumber' in self.rsltFwd[0]: self._rsltFwdInd2rsltBck()
            self._addReffMode(modeCut=0.5) # try to create mode resolved rEff with split at 0.5 μm (if it isn't already there)
            # SAVE RESULTS
            if savePath: self.saveSim(savePath, lightSave, verbose)
        else:
            if savePath: warnings.warn('This was a dry run. No retrievals were performed and no results were saved.')
            for gObj in gDB.grObjs: gObj.writeSDATA()
        if workingFileSave and savePath: # TODO: build zip from original tmp folders without making extra copies to disk, see first answer here: https://stackoverflow.com/questions/458436/adding-folders-to-a-zip-file-using-python
            fullSaveDir = savePath[0:-4]
            if verbose: print('Packing GRASP working files up into %s' %  fullSaveDir + '.zip')
            if os.path.exists(fullSaveDir): shutil.rmtree(fullSaveDir)
            os.makedirs(fullSaveDir)
            if not type(fwdData[0]) == dict: # If yes, then this was an OSSE run (no forward calc. or gDBFwd object)
                for i, gb in enumerate(gDBFwd.grObjs):
                    shutil.copytree(gb.dirGRASP, os.path.join(fullSaveDir,'forwardCalc%03d' % i))
            for i, gb in enumerate(gDB.grObjs):
                shutil.copytree(gb.dirGRASP, os.path.join(fullSaveDir,'inversion%03d' % i))
            shutil.make_archive(fullSaveDir, 'zip', fullSaveDir)
            shutil.rmtree(fullSaveDir)
        # delete the temporary files used for GRASP run
        if delTempFiles:
            try:
                if not type(fwdData[0]) == dict:
                    [os.system('rm -rf %s/' %gb.dirGRASP) for i, gb in enumerate(gDBFwd.grObjs)]
                [os.system('rm -rf %s/' %gb.dirGRASP) for i, gb in enumerate(gDB.grObjs)]
            except:
                print('Issue with deleting the temp files related to GRASP run')
        return

    def _rsltFwdInd2rsltBck(self):
        assert len(self.rsltBck)==len(self.rsltFwd), 'rsltFwd (N=%d) and rsltBck (N=%d) must be same length to transfer pixNumber indices!' % (len(self.rsltFwd), len(self.rsltBck))
        warned = False
        for rb,rf in zip(self.rsltBck, self.rsltFwd):
            latMisMatch = 'latitude' in rf and 'latitude' in rb and not np.isclose(rf['latitude'], rb['latitude'], atol=0.01)
            lonMisMatch = 'longitude' in rf and 'longitude' in rb and not np.isclose(rf['longitude'], rb['longitude'], atol=0.01)
            if lonMisMatch or latMisMatch:
                if not warned: 
                    warnings.warn('rsltFwd and rsltBck LAT and/or LON did not match for at least one pixel, setting pixNumber=-1')
                rb['pixNumber'] = -1
                warned = True
            else:
                rb['pixNumber'] = rf['pixNumber']

    def saveSim(self, savePath, lightSave=False, verbose=False):
        if not os.path.exists(os.path.dirname(savePath)):
            print('savePath (%s) did not exist, creating it...' % os.path.dirname(savePath))
            os.makedirs(os.path.dirname(savePath))
        if lightSave: self._makeRsltLight(self.rsltFwd, self.rsltBck)
        self.rsltBck[0]['version'] = rg.RSLT_DICT_VERSION if 'RSLT_DICT_VERSION' in dir(rg) else '0.0'
        self.rsltFwd[0]['version'] = rg.RSLT_DICT_VERSION if 'RSLT_DICT_VERSION' in dir(rg) else '0.0'
        if verbose: print('Saving simulation results to %s' %  savePath)
        with open(savePath, 'wb') as f:
            pickle.dump(list(self.rsltBck), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(list(self.rsltFwd), f, pickle.HIGHEST_PROTOCOL)

    def saveSim_netCDF(self, savePath, verbose=False):
        if not os.path.exists(os.path.dirname(savePath)) and not len(os.path.dirname(savePath))==0:
            print('savePath (%s) did not exist, creating it...' % os.path.dirname(savePath))
            os.makedirs(os.path.dirname(savePath))
        gRun = rg.graspRun()
        for dsc,rslts in zip(['TRUTH', 'RETRIEVED'], (self.rsltFwd, self.rsltBck)):
            savePathNow = savePath.replace('.nc4','') + ('_%s.nc4' % dsc)
            if verbose: print('Saving simulation %s results to %s' %  (dsc.lower(), savePathNow))
            gRun.output2netCDF(savePathNow, rsltDict=rslts)

    def loadSim(self, picklePath, standardizePSD=True, saveMerge=True, forceMerge=False, lightSave=False):
        """
        Load simulation results from one or more pickle files
        picklePath – Full path to the file, or glob-like string that will match more than one files to be merged
        standardizePSD – All fwd and back PSD placed on a common radii grid (see self._standardizePSD method)
        saveMerge – If a merge is done on multiple pickles, save results into a single pickle (format below)
        forceMerge – Force a merge, even if a matching saved merged file exists
        """
        splitPath = path.split(picklePath)
        savePATH = path.join(splitPath[0], 'MERGED_' + splitPath[1].replace('*','ALL'))
        # If already exists, load the file
        if os.path.exists(savePATH) and not forceMerge:
            files = [savePATH]
        else:
            files = glob(picklePath)
        assert len(files)>0, 'No files found!'
        if len(files)==1:
            self.rsltFwd, self.rsltBck = self._loadData(files[0])
        else:
            self.rsltFwd = []
            self.rsltBck = []
            print('Building %s - Nfiles=%d' % (saveFN, len(files)))
            for file in files: # loop over all found files
                rsltFwd, rsltBck = self._loadData(file)
                if lightSave: self._makeRsltLight(rsltFwd, rsltBck) # do this now instead of save to reduce memory footprint
                Nrepeats = 1 if len(rsltBck)==len(rsltFwd) else len(rsltBck)
                for _ in range(Nrepeats): self.rsltFwd = self.rsltFwd + rsltFwd
                self.rsltBck = self.rsltBck + rsltBck
        if standardizePSD: self._standardizePSD()
        if saveMerge: self.saveSim(savePATH, verbose=True)

    def _loadData(self, picklePath, verbose=True):
        with open(picklePath, 'rb') as f:
            rsltBck = rg.frmtLoadedRslts(pickle.load(f))
            try:
                rsltFwd = rg.frmtLoadedRslts(pickle.load(f))
            except EOFError: # this was an older file (created before Jan 2020)
                rsltFwd = [rsltBck[-1]] # rsltFwd as a array of len==0 (not totaly backward compatible, it used to be straight dict)
                rsltBck = rsltBck[:-1]        
        if verbose: print('Loaded %d pixels from %s.' % (len(rsltBck), picklePath))
        return rsltFwd, rsltBck
    
    def _makeRsltLight(self, rsltFwd, rsltBck):
        for pmStr in ['angle', 'p11','p12','p22','p33','p34','p44','range','βext']:
            for rb in rsltBck: rb.pop(pmStr, None)
            if len(rsltFwd) > 1:
                for rf in rsltFwd: rf.pop(pmStr, None)
    
    def _standardizePSD(self):    
        """
        Place all fwd and back PSD on a common radii grid spanning smallest to largest size from all PSDs
        """
        if ('r' not in self.rsltFwd[0]) or ('r' not in self.rsltBck[0]): return # we have no PSD to standardize
        fidelityFactor = 2 # increase to shrink new grid spacing (dlnr_new = min(dlnr)/fidelityFactor)
          # This handles radii that vary from pixel-to-pixel but takes ~1 sec for sim with 100k pixels
#         if np.all([rs['r'][:,0]==rs['r'][0,0] for rs in self.rsltFwd]) & \
#            np.all([rs['r'][:,0]==rs['r'][0,0] for rs in self.rsltBck]) & \
#            np.all([rs['r'][:,-1]==rs['r'][0,-1] for rs in self.rsltFwd]) & \
#            np.all([rs['r'][:,-1]==rs['r'][0,-1] for rs in self.rsltBck]): # This also still needs back-to-fwd cross check to be complete (see below)
#            return # interpolation is slow; skip it if not required
        # This is effectively instantaneous but will cause method to fail to standardize if radii vary pixel-to-pixel
        if np.all(self.rsltFwd[0]['r'][:,0]==self.rsltFwd[0]['r'][0,0]) and \
           np.all(self.rsltBck[0]['r'][:,0]==self.rsltBck[0]['r'][0,0]) and \
           np.all(self.rsltFwd[0]['r'][:,-1]==self.rsltFwd[0]['r'][0,-1]) and \
           np.all(self.rsltBck[0]['r'][:,-1]==self.rsltBck[0]['r'][0,-1]) and \
           self.rsltFwd[0]['r'][0,0]==self.rsltBck[0]['r'][0,0] and \
           self.rsltFwd[0]['r'][0,-1]==self.rsltBck[0]['r'][0,-1]:
           return # interpolation is slow; skip it if not required
        # This handles radii that vary from pixel-to-pixel but takes ~10 sec for sim with 100k pixels
#         rmin = np.inf
#         rmax = 0
#         dlnr = np.inf
#         for rsltDictList in [self.rsltFwd, self.rsltBck]:
#             for rs in rsltDictList: # best to loop over fwd & back separately because they are not guaranteed to have same len()
#                 rmin = min(rmin, rs['r'].min())
#                 rmax = max(rmax, rs['r'].max())
#                 dlnr = min(dlnr, np.diff(np.log(rs['r'])).min()) # possibly change this to log10 everywhere for efficiency at end
        rmin = min(self.rsltFwd[0]['r'].min(), self.rsltBck[0]['r'].min())
        rmax = max(self.rsltFwd[0]['r'].max(), self.rsltBck[0]['r'].max())
        dlnrFwd = np.diff(np.log(self.rsltFwd[0]['r'])).min()
        dlnrBck = np.diff(np.log(self.rsltBck[0]['r'])).min()
        dlnr = min(dlnrFwd, dlnrBck)
        N = int(fidelityFactor*(np.log(rmax)-np.log(rmin))/dlnr+1) # <- ln(r_max) = ln(r_min)+lndr*(N-1)
        r_stand = np.logspace(np.log10(rmin), np.log10(rmax), N)
        for rsltDictList in [self.rsltFwd, self.rsltBck]:
            for i,rs in enumerate(rsltDictList): # best to loop over fwd & bck separately because they are not guaranteed to have same len()
                Nmode = rs['r'].shape[0]
                dvdlnr = np.empty((Nmode, len(r_stand)))
                for j,(r,dv) in enumerate(zip(rs['r'], rs['dVdlnr'])): # loop over modes
                    dvdlnr[j,:] = np.interp(r_stand, r, dv, left=0, right=0) # this could cause interpolation errors...
                    leftInd = r_stand<r.min()
                    if 'rv' in rs and leftInd.any(): # fill in in below current modes lower bound using rv/σ
                        r_left = np.r_[r_stand[leftInd], r[0]]
                        dv_left = ms.logNormal(rs['rv'][j], rs['sigma'][j], r_left)[0]
                        dvdlnr[j,leftInd] = dv[0]/dv_left[-1]*dv_left[:-1] # stitch to existing dvdlnr
                rsltDictList[i]['dVdlnr'] = dvdlnr
                rsltDictList[i]['r'] = np.tile(r_stand, [Nmode,1])

    def _addReffMode(self, modeCut=None, Force=False):
        oneAdded = False
        if Force or ('rEffMode' not in self.rsltFwd[0] and ('rv' in self.rsltFwd[0] or 'dVdlnr' in self.rsltFwd[0])):
            for rf in self.rsltFwd: rf['rEffMode'] = self.ReffMode(rf, modeCut=modeCut).squeeze()
            oneAdded = not oneAdded
        if Force or ('rEffMode' not in self.rsltBck[0] and ('rv' in self.rsltBck[0] or 'dVdlnr' in self.rsltBck[0])):
            for rb in self.rsltBck: rb['rEffMode'] = self.ReffMode(rb, modeCut=modeCut).squeeze()
            oneAdded = not oneAdded
        if oneAdded:
            warnings.warn('We added rEffMode to one of fwd/bck but not the other. This may cause inconsistency if definitions differ.')

    def conerganceFilter(self, χthresh=None, σ=None, forceχ2Calc=False, verbose=False, minSaved=2): # TODO: LIDAR bins with ~0 concentration are dominating this metric...
        """ Only removes data from resltBck if χthresh is provided, χthresh=1.5 seems to work well
        Now we use costVal from GRASP if available (or if forceχ2Calc==True), χthresh≈2.5 is probably better
        NOTE: if forceχ2Calc==True or χthresh~=None this will permanatly alter the values of rsltBck/rsltFwd
        """
        if σ is None:
            σ={'I'   :0.003, # relative
               'QoI' :0.0015, # absolute
               'UoI' :0.005, # absolute
               'Q'   :0.0015, # absolute in terms of Q/I
               'U'   :0.005, # absolute in terms of U/I
               'LS'  :0.15, # relative
               'VBS' :0.05, # relative
               'VExt':5e-6} # absolute
        if 'costVal' not in self.rsltBck[0] or forceχ2Calc:
            for i,rb in enumerate(self.rsltBck):
                rf = self.rsltFwd[0] if len(self.rsltFwd)==1 else self.rsltFwd[i]
                χΤοtal = np.array([])
                for measType in ['VExt', 'VBS', 'LS', 'I', 'QoI', 'UoI', 'Q', 'U']:
                    fitKey = 'fit_'+measType
                    if fitKey in rb:
                        DBck = rb[fitKey][~np.isnan(rb[fitKey])]
                        if fitKey in rf:
                            DFwd = rf[fitKey][~np.isnan(rf[fitKey])]
                        elif measType in ['QoI', 'UoI'] and fitKey[:-2] in rf: # rf has X while rb has XoI
                            DFwd = rf[fitKey[:-2]][~np.isnan(rf[fitKey[:-2]])]/rf['fit_I'][~np.isnan(rf[fitKey[:-2]])]
                        else:
                            assert False, 'Fwd lacked a rsltDict key for an observable that Bck had... Do you really need forceχ2Calc=True? If so, see issue #3 on GitHub' 
                        if measType in ['I', 'LS', 'VBS']: # relative errors
                            with np.errstate(divide='ignore'): # possible for DBck+DFwd=0, inf's will be removed below
                                χLocal = ((2*(DBck-DFwd)/(DBck+DFwd))/σ[measType])**2
                            χLocal[χLocal>100] = 100 # cap at 10σ (small values may produce huge relative errors)
                        else: # absolute errors
                            if measType in ['Q', 'U']: # we need to normalize by I, all Q[U] errors given in terms of q[u]
                                DBck = DBck/rb['fit_I'][~np.isnan(rb[fitKey])]
                                DFwd = DFwd/rf['fit_I'][~np.isnan(rf[fitKey])]
                            χLocal = ((DBck-DFwd)/σ[measType])**2
                        χΤοtal = np.r_[χΤοtal, χLocal]
                rb['costVal'] = np.sqrt(np.mean(χΤοtal))
        if χthresh and len(self.rsltBck) > 2: # we will always keep at least 2 entries
            validInd = np.array([rb['costVal']<=χthresh for rb in self.rsltBck])
            if verbose: print('%d/%d met χthresh' % (validInd.sum(), len(self.rsltBck)))
            if validInd.sum() < minSaved:
                validInd = np.argsort([rb['costVal'] for rb in self.rsltBck])[0:minSaved] # note validInd went from bool to array of ints
                if verbose:
                    print('Preserving the %d rsltBck elements with lowest χ scores, even though they did not meet χthresh.' % minSaved)
            if len(self.rsltFwd)==len(self.rsltBck): self.rsltFwd = self.rsltFwd[validInd]
            self.rsltBck = self.rsltBck[validInd]
        elif χthresh and np.sum([rb['costVal']<=χthresh for rb in self.rsltBck])<minSaved and verbose:
            print('rsltBck only has %d or fewer elements, no χthresh screening will be perofmed.' % minSaved)

    def analyzeSimProfile(self, wvlnthInd=0, fineModesFwd=[0,2], fineModesBck=[0,2], rsltRange=None, fwdSim=None):
        """
        rsltRange – range bins to use in polarimeter data
        We will assume:
        len(rsltFwd)==len(rsltBck) << CHECKED BELOW
        finemodes and pbl modes are same index in fwd and bck data
        We need error/bias estimats for:
        number concentration FT/PBL resolved
        effective radius FT/PBL resolved
            --- reff vertical averaging (reff_h=Rh, # conc.=Nh, vol_conc.=Vh, height_ind=h) ---
            reff = ∫ ΣNh*r^3*dr/∫ ΣNh*r^2*dr = 3/4/π*ΣVh/∫ Σnh*r^2*dr
            Rh = (3/4/π*Vh)/∫ nh*r^2*dr -> reff = 3/4/π*ΣVh/(Σ(3/4/π*Vh)/Rh) = ΣVh/Σ(Vh/Rh) = Σβh/Σ(βh/Rh)
        ssa FT/PBL resolved # ω=Σβh/Σαh & ωh*αh=βh => ω=Σωh*αh/Σαh
        lidar ratio FT/PBL resolved  # S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh) <<< not implemented yet
        """
        if fwdSim is None:
            rsltFwdLcl = self.rsltFwd
            rngBins = rsltRange
            wvlnthIndFwd = wvlnthInd
        else:
            rsltFwdLcl = fwdSim.rsltFwd
            rngBins = rsltFwdLcl[0]['range'][0,:] if rsltRange is None else rsltRange # this should be okay, lidar case will just ignore it
            wvlnthIndFwd = np.argmin(np.abs(rsltFwdLcl[0]['lambda']-self.rsltBck[0]['lambda'][wvlnthInd]))
        if len(rsltFwdLcl) > len(self.rsltBck):
            warnings.warn('More fwd rslts than bck, clipping extra fwd pixels. Safe in canoncical cases b/c no geomtry dependent vars are used in this method.')
            rsltFwdLcl = rsltFwdLcl[0:len(self.rsltBck)]
        assert len(rsltFwdLcl)==len(self.rsltBck), 'This method only works with sims where fwd and bck pair one-to-one'
        self.addProfFromGuassian(rsltRange=rngBins)
        pxKy = ['biasExt','trueExt','biasExtFine','trueExtFine', 'biasSSA', 'trueSSA', 'biasLR', 'trueLR']
        pxDct = {k: [] for k in pxKy}
        for rf,rb in zip(rsltFwdLcl, self.rsltBck):
            extFwd, scaFwd = self._findExtScaProf(rf, wvlnthIndFwd)
            extBck, scaBck = self._findExtScaProf(rb, wvlnthInd)
            pxDct['biasExt'].append(np.sum(extBck, axis=0) - np.sum(extFwd, axis=0))
            pxDct['trueExt'].append(np.sum(extFwd, axis=0)) # sum over all modes
            pxDct['biasExtFine'].append(np.sum(extBck[fineModesBck,:], axis=0) - np.sum(extFwd[fineModesFwd,:], axis=0))
            pxDct['trueExtFine'].append(np.sum(extFwd[fineModesFwd,:], axis=0)) # sum over all modes
            ssaFwd = np.sum(scaFwd, axis=0)/np.sum(extFwd, axis=0)
            ssaBck = np.sum(scaBck, axis=0)/np.sum(extBck, axis=0)
            pxDct['biasSSA'].append(ssaBck - ssaFwd)
            pxDct['trueSSA'].append(ssaFwd)
            LRModes = rf['LidarRatioMode'][:, wvlnthIndFwd][:,None]
            LRFwd = np.sum(scaFwd, axis=0)/np.sum(scaFwd/LRModes, axis=0)
            LRModes = rb['LidarRatioMode'][:, wvlnthInd][:,None]
            LRBck = np.sum(scaBck, axis=0)/np.sum(scaBck/LRModes, axis=0)
            pxDct['biasLR'].append(LRBck - LRFwd)
            pxDct['trueLR'].append(LRFwd)
        bias = {'βext':np.array(pxDct['biasExt']), 'βextFine':np.array(pxDct['biasExtFine']),
                'ssa':np.array(pxDct['biasSSA']), 'LR':np.array(pxDct['biasLR'])}
        βextMeans2 = [(prof**2).mean() for prof in pxDct['biasExt']] # profiles may not all be the same number of bins
        βextFineMeans2 = [(prof**2).mean() for prof in pxDct['biasExtFine']]
        ssaMeans2 = [(prof**2).mean() for prof in pxDct['biasSSA']]
        LRMeans2 = [(prof**2).mean() for prof in pxDct['biasLR']]
        rmse = {'βext':np.sqrt(βextMeans2),'βextFine':np.sqrt(βextFineMeans2),
                'ssa':np.sqrt(ssaMeans2), 'LR':np.sqrt(LRMeans2)}
        true = {'βext':np.array(pxDct['trueExt']), 'βextFine':np.array(pxDct['trueExtFine']),
                'ssa':np.array(pxDct['trueSSA']), 'LR':np.array(pxDct['trueLR'])}
        return rmse, bias, true # each w/ keys: βext, βextFine, ssa

    def _findExtScaProf(self, rslt, wvlnthInd):
        Nmodes = len(rslt['aodMode'][:,0])
        rng = rslt['range'][0,:]
        Nrange = len(rng)
        extPrf = np.empty([Nmodes, Nrange])
        scaPrf = np.empty([Nmodes, Nrange])
        for i in range(Nmodes):
            extPrf[i,:] = ms.norm2absExtProf(rslt['βext'][i,:], rng, rslt['aodMode'][i,wvlnthInd])
            scaPrf[i,:] = extPrf[i,:]*rslt['ssaMode'][i,wvlnthInd]
        return extPrf, scaPrf

    def analyzeSim(self, wvlnthInd=0, modeCut=None, hghtCut=None, fineModesFwd=None, fineModesBck=None):
        """ Returns the RMSE and bias (defined below) from the simulation results
                wvlngthInd - the index of the wavelength to calculate stats for
                modeCut - fine/coarse seperation radius in um, currenltly only applied to rEff (None -> do not calculate error's modal dependence)
                hghtCut - PBL/FT seperation in meters - can be list (None -> use rsltFwd['pblh'] if present, otherwise do not vertically resolve error)
                            if 'pblh' in fwd keys (OSSE case), fwd PBL(FT) value for VarX will pulled from VarX_PBL(FT), ignoring the value of hghtCut
                            in back OSSE case hghtCut argument is still used if provided, otherwise it is pulled from 'pblh' key
                fineModesFwd - [array-like] the indices of the fine modes in the foward calculation, set to None to use OSSE ..._Fine variables instead
                fineModesBck -  [array-like] the indices of the fine modes in the retrieval """
        # check on input and available variables
        assert (self.rsltBck is not None) and (self.rsltFwd is not None), 'You must call loadSim() or runSim(...,dryRun=False) before you can calculate statistics!'
        if type(self.rsltFwd) is dict: self.rsltFwd = [self.rsltFwd]
        assert type(self.rsltFwd) is list or type(self.rsltFwd) is np.ndarray, 'rsltFwd must be a list! Note that it was stored as a dict in older versions of the code.'
        fwdKys = self.rsltFwd[0].keys()
        bckKys = self.rsltBck[0].keys()
        msg = ' contains indices that are too high given the number of modes in '
        assert 'aodMode' not in fwdKys or fineModesFwd is None or self.rsltFwd[0]['aodMode'].shape[0] > max(fineModesFwd), 'fineModesFwd'+msg+'rsltFwd[aodMode]'
        assert 'aodMode' not in bckKys or fineModesBck is None or self.rsltBck[0]['aodMode'].shape[0] > max(fineModesBck), 'fineModesBck'+msg+'rsltBck[aodMode]'
        # define functions for calculating RMS and bias
        # rmsFun = lambda t,r: np.mean(r-t, axis=0) # formula for RMS output (true->t, retrieved->r)
        rmsFun = lambda t,r: np.sqrt(np.nanmedian((t-r)**2, axis=0)) # formula for RMS output (true->t, retrieved->r) – needs to handle nans! (hghtWghtedAvg returns nan when PBL<lowest_GRASP_bin)
        biasFun = lambda t,r: r-t if r.ndim > 1 else np.atleast_2d(r-t).T # formula for bias output – needs to handle nans! (hghtWghtedAvg returns nan when PBL<lowest_GRASP_bin)
        # variables we expect to see
        varsSpctrl = ['aod', 'aodMode', 'n', 'k', 'ssa', 'ssaMode', 'g','LidarRatio', 'LidarRatioMode']
        varsMorph = ['rv', 'sigma', 'sph', 'rEffMode', 'rEffCalc', 'rEff', 'height']
        varsAodAvg = ['n', 'k'] # modal variables for which we will append aod weighted average RMSE and BIAS value at the FIRST element (expected to be spectral quantity)
        modalVars = ['rv', 'sigma', 'sph', 'aodMode', 'ssaMode','rEffMode', 'n', 'k', 'LidarRatioMode'] # variables for which we find fine/coarse or FT/PBL errors seperately
        self._addReffMode() # calculate variables that weren't loaded (rEffMode)
        if hghtCut is not None:
            if not (type(hghtCut) is np.ndarray or type(hghtCut) is list):
                hghtCut = np.full(len(self.rsltBck), hghtCut)
        elif 'pblh' in fwdKys and len(self.rsltFwd)==len(self.rsltBck):
            hghtCut = [rd['pblh'] for rd in self.rsltFwd]
        varsSpctrl = [z for z in varsSpctrl if z in fwdKys and z in bckKys] # check that the variable is used in current configuration
        varsMorph = [z for z in varsMorph if z in fwdKys and z in bckKys]
        # loop through varsSpctrl and varsMorph calcualted RMS and bias
        rmsErr = dict()
        bias = dict()
        trueOut = dict()
        for av in varsSpctrl+varsMorph:
            rtrvd = np.array([rb[av] for rb in self.rsltBck])
            true = np.array([rf[av] for rf in self.rsltFwd])
            if av in varsSpctrl:
                rtrvd = rtrvd[...,wvlnthInd]
                true = true[...,wvlnthInd]
            if rtrvd.ndim==1: rtrvd = np.expand_dims(rtrvd,1) # we want [ipix, imode], we add a singleton dimension if only one mode/nonmodal
            if true.ndim==1: true = np.expand_dims(true,1) # we want [ipix, imode], we add a singleton dimension if only one mode/nonmodal
            if hghtCut is not None and av in modalVars and (av+'_PBL' in fwdKys or ('aodMode' in fwdKys and 'βext' in fwdKys)): # calculate vertical dependent RMS/BIAS [PBL, FT*]
                if av+'_PBL' in fwdKys:
                    trueBilayer = self.getStateVals(av+'_PBL', self.rsltFwd, varsSpctrl, wvlnthInd)
                    pblOnly = av+'_FT' not in fwdKys
                    if not pblOnly:
                        trueFT = self.getStateVals(av+'_FT', self.rsltFwd, varsSpctrl, wvlnthInd)
                        trueBilayer = np.block([[trueBilayer],[trueFT]]).T # trueBilayer[pixel,mode]
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut, av, pblOnly=pblOnly)
                else: # 'aodMode' in self.rsltFwd[0]
                    trueBilayer = self.hghtWghtedAvg(true, self.rsltFwd, wvlnthInd, hghtCut, av)
                    rtrvdBilayer = self.hghtWghtedAvg(rtrvd, self.rsltBck, wvlnthInd, hghtCut, av)
                if np.all(np.isnan(rtrvdBilayer[:,0])):
                    warnings.warn('Lowest GRASP bin below PBL at every pixel: All-NaN slice encountered -> can not calculate PBL uncertainties')
                rmsErr[av+'_PBLFT'] = rmsFun(trueBilayer, rtrvdBilayer) # PBL is 1st ind, FT (not total column!) is 2nd
                bias[av+'_PBLFT'] = biasFun(trueBilayer, rtrvdBilayer)
                trueOut[av+'_PBLFT'] = trueBilayer
            if fineModesBck is not None and av in modalVars: # calculate fine mode dependent RMS/BIAS
                fineCalculated = False
                if av+'_fine' in fwdKys and fineModesFwd is None and 'aodMode' in bckKys: # we have OSSE outputs, currently user provided fineModesFwd trumps this though
                    trueFine = self.getStateVals(av+'_fine', self.rsltFwd, varsSpctrl, wvlnthInd)
                    rtrvdFine = self.getStateVals(av, self.rsltBck, varsSpctrl, wvlnthInd, fineModesBck)
                    fineCalculated = True
                elif fineModesFwd is not None and 'aodMode' in fwdKys and 'aodMode' in bckKys: # user provided fwd and bck fine mode indices
                    trueFine = self.getStateVals(av, self.rsltFwd, varsSpctrl, wvlnthInd, fineModesFwd)
                    rtrvdFine = self.getStateVals(av, self.rsltBck, varsSpctrl, wvlnthInd, fineModesBck)
                    fineCalculated = True
                elif fineModesFwd is not None: # something went wrong...
                    assert False, 'If fineModeFwd and fineModeBck are provided aodMode must be present in rsltsBck and rsltsFwd!'
                if fineCalculated:
                    rmsErr[av+'_fine'] = rmsFun(trueFine, rtrvdFine) # PBL is 1st ind, FT (not total column!) is 2nd
                    bias[av+'_fine'] =  biasFun(trueFine, rtrvdFine)
                    trueOut[av+'_fine'] =  trueFine
            if av in varsAodAvg: # calculate the total, mode AOD weighted value of the variable (likely just CRI) -> [total, mode1, mode2,...]
                rtrvd = np.hstack([self.τWghtedAvg(rtrvd, self.rsltBck, wvlnthInd), rtrvd])
                true = np.hstack([self.τWghtedAvg(true, self.rsltFwd, wvlnthInd), true])
            if true.shape[1] == rtrvd.shape[1]: # truth and retrieved modes can be paired one-to-one
                rmsErr[av] = rmsFun(true, rtrvd) # BUG: fineModesFwd and fineModesBck and not taken into accoutn here, really we just shouldn't return n or k with more than one mode (we have n(k)_fine now)
                bias[av] = biasFun(true, rtrvd)
                trueOut[av] = true
            elif av in varsAodAvg and 'aodMode' in fwdKys and 'aodMode' in bckKys: # we at least know the first, total elements of CRI correspond to each other
                rmsErr[av] = rmsFun(true[:,0], rtrvd[:,0])
                bias[av] = biasFun(true[:,0], rtrvd[:,0])
                trueOut[av] = true[:,0]
            if modeCut: # calculate rEff, could be abstracted into above code but tricky b/c volume weighted mean will not give exactly correct results (code below is exact)
                rtrvd = np.squeeze([self.ReffMode(rb, modeCut) for rb in self.rsltBck])
                true = np.squeeze([self.ReffMode(rf, modeCut) for rf in self.rsltFwd])
                modeCut_nm = 1000*modeCut
                rmsErr['rEff_sub%dnm' % modeCut_nm] = rmsFun(true, rtrvd)
                bias['rEff_sub%dnm' % modeCut_nm] = biasFun(true, rtrvd)
                trueOut['rEff_sub%dnm' % modeCut_nm] = true
                # TODO: add sph fraction here? It would use volWghtedAvg, I think in its exact current form
            if av in rmsErr: rmsErr[av] = np.atleast_1d(rmsErr[av]) # HACK: n was coming back as scalar in some cases, we should do this right though
        return rmsErr, bias, trueOut

    def getStateVals(self, av, rslts, varsSpctrl, wvlnthInd, modeInd=None):
        assert av in rslts[0], 'Variable %s was not found in the rslts dictionary!' % av
        stateVals = np.array([rf[av] for rf in rslts])
        if av.replace('_PBL','').replace('_FT','').replace('_fine','').replace('_coarse','') in varsSpctrl:
            stateVals = stateVals[...,wvlnthInd]
        if modeInd is not None and 'aod' in av: # we will sum multiple fine modes to get total fine mode AOD
            stateVals = np.expand_dims(stateVals[:, modeInd].sum(axis=1),1)
        elif modeInd is not None: # we will perform AOD weighted averaging from all fine modes of an intensive property
            stateVals = self.τWghtedAvg(stateVals[:, modeInd], rslts, wvlnthInd, modeInd)
        return stateVals

    def τWghtedAvg(self, val, rslts, wvlnthInd, modeInd=slice(None)):
        avgVal = np.full(len(rslts), np.nan)
        if (val.shape[1]==1 and slice(None)==modeInd) or 'aodMode' not in rslts[0]: # there was only one mode to start OR can't calculate b/c we don't have aodMode
            return np.zeros([val.shape[0], 0]) # return an empty array
        for i,rslt in enumerate(rslts):
            ttlSum = np.sum(val[i]*rslt['aodMode'][modeInd,wvlnthInd])
            normDenom = np.sum(rslt['aodMode'][modeInd,wvlnthInd])
            avgVal[i] = ttlSum/normDenom
        return np.expand_dims(avgVal, 1)

    def addProfFromGuassian(self, rslts=None, rsltRange=None):
        if rslts is None:
            self.addProfFromGuassian(self.rsltFwd, rsltRange)
            self.addProfFromGuassian(self.rsltBck, rsltRange)
            return
        if 'range' in rslts[0]: return # profiles are already there
        if rsltRange is None: rsltRange = np.linspace(1, 2e4, int(1e3))
        for rslt in rslts:
            Nmodes = len(rslt['height'])
            rslt['range'] = np.empty([Nmodes,len(rsltRange)])
            rslt['βext'] = np.empty([Nmodes,len(rsltRange)])
            for i, (μ,σ) in enumerate(zip(rslt['height'], rslt['heightStd'])): # loop over modes
                rslt['range'][i,:] = rsltRange
                guasDist = norm(loc=μ, scale=σ)
                rslt['βext'][i,:] = guasDist.pdf(rsltRange)

    def hghtWghtedAvg(self, val, rslts, wvlnthInd, hghtCuts, av, pblOnly=False):
        """
        If no GRASP bins are at hghtCut or below we return nan for that pixel
        quantities in val could correspond to: av { ['rv', 'sigma', 'sph', 'aodMode', 'ssaMode','n','k']
            ω=Σβh/Σαh => ω=Σωh*αh/Σαh, i.e. aod weighting below is exact for SSA
            sph is vol weighted and also exact
            there is no write answer for rv,σ,n and k so they are AOD weighted
            lidar ratio is exact, calculated by: S=τ/F11[180] & F11h[180]=τh/Sh => S=Στh/Σ(τh/Sh)
            --- reff vertical averaging (reff_h=Rh, # conc.=Nh, vol_conc.=Vh, height_ind=h) ---
            reff = ∫ ΣNh*r^3*dr/∫ ΣNh*r^2*dr = 3/4/π*ΣVh/∫ Σnh*r^2*dr
            Rh = (3/4/π*Vh)/∫ nh*r^2*dr -> reff = 3/4/π*ΣVh/(Σ(3/4/π*Vh)/Rh) = ΣVh/Σ(Vh/Rh)
        """
        assert av not in 'gMode', 'We dont have a wghtVals setting for asymetry parameter (i.e. ssaMode*aodMode)!'
        self.addProfFromGuassian(rslts) # no impact if profiles already exist
        if av in ['aod']:
            wghtFun = lambda v,w: np.sum(w)
        elif av in ['rEffMode', 'LidarRatio']:
            wghtFun = lambda v,w: np.sum(w)/np.sum(w/v)
        else: # SSA, rv, σ, n, k, gMode
            wghtFun = lambda v,w: np.nan if np.all(w)==0 else np.sum(v*w)/np.sum(w)
        Bilayer = np.full([len(rslts), 2-pblOnly], np.nan)
        for i,(rslt,hghtCut) in enumerate(zip(rslts, hghtCuts)):
            if av in ['rEffMode', 'sph']: # weighting based on volume
                wghtVals = rslt['vol']
            else:  # weighting based on optical thickness
                wghtVals = rslt['aodMode'][:,wvlnthInd]
            wghtsPBL = []
            for β,h,τ in zip(rslt['βext'], rslt['range'], wghtVals): # loop over modes
                wghtsPBL.append(β[h <= hghtCut].sum()/β.sum()*τ) # this is τ contribution of each mode below hghtCut
            valPBL = wghtFun(val[i], wghtsPBL)
            if not pblOnly:
                wghtsFT = []
                for β,h,τ in zip(rslt['βext'], rslt['range'], wghtVals):
                    wghtsFT.append(β[h > hghtCut].sum()/β.sum()*τ)
                valFT = wghtFun(val[i], wghtsFT)
                Bilayer[i] = [valPBL, valFT]
            else:
                Bilayer[i] = [valPBL]
        return Bilayer

    def ReffMode(self, rs, modeCut=None):
        if 'rv' in rs and 'sigma' in rs:
            if modeCut is None:
                rv_dndr = rs['rv']/np.exp(3*rs['sigma']**2)
                return rv_dndr*np.exp(5/2*rs['sigma']**2) # eq. 60 from Grainger's "Useful Formulae for Aerosol Size Distributions"
            Vfc = self.volWghtedAvg(None, [rs], modeCut)
            Amode = rs['vol']/rs['rv']*np.exp(rs['sigma']**2/2) # convert N to rv and then ratio 1st and 4th rows of Table 1 of Grainger's "Useful Formulae for Aerosol Size Distributions"
            Afc = self.volWghtedAvg(None, [rs], modeCut, Amode)
            return Vfc/Afc # NOTE: ostensibly this should be Vfc/Afc/3 but we ommited the factor of 3 from Amode as well
        elif 'dVdlnr' in rs and 'r' in rs:
            fnCrsReff = []
            if modeCut is None: 
                for i in range(rs['r'].shape[0]):
                    fnCrsReff.append(ms.integratePSD([rs], moment='reff', sizeMode=i))
            else:
                fnCrsReff.append(ms.integratePSD([rs], moment='reff', upBnd=modeCut))
                fnCrsReff.append(ms.integratePSD([rs], moment='reff', lowBnd=modeCut))
            return np.squeeze(fnCrsReff) # integratePSD above was return 1 element np.arrays before each append to fnCrsReff
        else:
            assert False, 'Can not calculate effective radius without either rv & σ OR dVdlnr & r!'

    def volWghtedAvg(self, val, rslts, modeCut, vol=None, fineOnly=False):
        N = 200 # number of radii bins
        Bimode = np.full([len(rslts), 2-fineOnly], np.nan)
        for i,rslt in enumerate(rslts): # loop over each pixel/time
            sigma4 = np.exp(4*rslt['sigma'].max())
            minVal = rslt['rv'].min()/sigma4
            maxVal = rslt['rv'].max()*sigma4
            lower = [minVal] if fineOnly else [minVal, modeCut]
            upper = [modeCut] if fineOnly else [modeCut, maxVal]
            crsWght = [] # this will be [upr/lwr(N=2), mode]
            for upr, lwr, in zip(upper,lower):
                r = np.logspace(np.log10(lwr),np.log10(upr),N)
                crsWght.append([np.trapz(ms.logNormal(mu, σ, r)[0],r) for mu,σ in zip(rslt['rv'],rslt['sigma'])]) # integrated r 0->inf this will sum to unity
            if not np.isclose(np.sum(crsWght, axis=0), 1, rtol=0.001).all():
                warnings.warn('The sum of the crsWght values across all modes deviated more than 0.1% from unity') # probably need to adjust N or sigma4 above
            if val is None: # we just want volume (or area if vol arugment contains area) concentration of each mode
                if vol is None:
                    crsWght = np.array(crsWght)*rslt['vol']
                else:
                    crsWght = np.array(crsWght)*vol
                Bimode[i] = np.sum(crsWght, axis=1)
            else:
                Bimode[i] = np.sum(crsWght*val[i], axis=1)
        return Bimode
        
    def classifyAerosolType(self, verbose=False):
        """
        0 Dust
        1 PolDust (only detected if vol_DU is available)
        2 Marine
        3 DevUrb
        4 BB-White
        5 BB-Dark
        """
        
        mu = np.array([
             [0.930, 1.51, -0.10],  # Dust
             [999.9, 1.51, -0.10],  # Pol Dust XXX
             [1.000, 1.40,  0.25],  # Marine (we'll set pol dust using dust frac)
             [0.985, 1.37,  1.60],  # Urban/Industrial
             [0.930, 1.49,  1.80],  # BB-White
             [0.850, 1.53,  1.70]]) # BB-Dark 

        s = np.array([
             [0.015, 0.06, 0.05],  # Dust
             [0.015, 0.06, 0.05],  # pol. Dust
             [0.010, 0.02, 0.25],  # Marine
             [0.010, 0.02, 0.15],  # Urban/Industrial
             [0.025, 0.04, 0.20],  # BB-White
             [0.035, 0.04, 0.20]]) # BB-Dark
        
        modeInd = 0 # use these mode resolved parameters (this method will currently produce strange results for multimodal data)
        blui = np.abs(self.rsltFwd[0]['lambda']-0.41).argmin()
        redi = np.abs(self.rsltFwd[0]['lambda']-0.670).argmin()
        niri = np.abs(self.rsltFwd[0]['lambda']-0.870).argmin()
        assert redi is not niri, 'Red and NIR wavelengths were the same. Can not calculate AE!'
        for rf in self.rsltFwd:
            if 'vol_DU' in rf: # use this to ID dust types if available
                dustVolFrac = rf['vol_DU']/rf['vol']
                if dustVolFrac > 0.65:
                    rf['aeroType'] = 0
                    continue
                elif dustVolFrac > 0.16:
                    rf['aeroType'] = 1
                    continue
            ssa = rf['ssa'][blui]
            rri = np.atleast_2d(rf['n'])[modeInd, redi]
            ae  = np.log(rf['aod'][blui]/rf['aod'][niri])/np.log(rf['lambda'][niri]/rf['lambda'][blui])
            x = np.r_[ssa, rri, ae]
            typeDist = [np.sqrt((x-mu[ti,:])**2/s[ti,:]).sum() for ti in range(mu.shape[0])]
            rf['aeroType'] = np.argmin(typeDist)
            if verbose and rf['aeroType'] == 1 and 'vol_DU' in rf:
                print('Dust volume fraction was %f and we still IDed as dust' % (dustVolFrac))




















