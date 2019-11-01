import numpy as np
import os
import sys
import re
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import runGRASP as rg
import functools        

# DUMMY MEASUREMENTS (determined by architecture, should ultimatly move to seperate scripts)
#  For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
#  len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
#  msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
def returnPixel(archName, sza=30, landPrct=100, relPhi=0, nowPix=None): 
    if not nowPix: nowPix = rg.pixel(730123.0, 1, 1, 0, 0, 0, landPrct) # can be called for multiple instruments, BUT they must all have unqiue wavelengths
    assert nowPix.land_prct == landPrct, 'landPrct provided did not match land percentage in nowPix'
    if 'polar07' in archName.lower(): # CURRENTLY ONLY USING JUST 10 ANGLES IN RED
        msTyp = [41, 42, 43] # must be in ascending order
        thtv = np.tile([-57.0,  -44.0,  -32.0 ,  -19.0 ,  -6.0 ,  6.0,  19.0,  32.0,  44.0,  57.0], len(msTyp))
        wvls = [0.360, 0.380, 0.410, 0.550, 0.670, 0.870, 0.940, 1.230, 1.380, 1.550, 1.650] # Nλ=11
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] 
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas)
            nowPix.measVals[-1]['errorModel'] = functools.partial(addError, 'polar07') # this must link to an error model in addError() below
    if 'polar09' in archName.lower(): # CURRENTLY ONLY USING JUST 10 ANGLES IN RED
        msTyp = [41, 42, 43] # must be in ascending order
        thtv = np.tile([-60, 0.001, 60], len(msTyp))
        wvls = [0.380, 0.410, 0.550, 0.670, 0.865] #
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] 
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas)
            nowPix.measVals[-1]['errorModel'] = functools.partial(addError, 'polar09') # this must link to an error model in addError() below
    if 'modismisr01' in archName.lower(): # MISR with MODIS spectral coverage (no polarization)
        msTyp = [41] # must be in ascending order
        thtv = np.tile([-70.5, -60.0, -45.6, -26.1, 0, 26.1, 45.6, 60.0, 70.5], len(msTyp))
        wvls = [0.410, 0.469, 0.555, 0.645, 0.8585, 1.24, 1.64, 2.13]
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0])] 
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas)
            nowPix.measVals[-1]['errorModel'] = functools.partial(addError, 'modismisr01') # this must link to an error model in addError() below
    if 'lidar05' in archName.lower(): # TODO: this needs to be more complex, real lidar05 has backscatter at 1 wavelength
#        msTyp = [35, 36, 39] # must be in ascending order # HACK: we took out depol b/c GRASP was throwing error (& canonical cases are spherical)
        msTyp = [36, 39] # must be in ascending order
        botLayer = 100 # bottom layer in meters
        topLayer = 20000
        Nlayers = 45 #TODO: ultimatly this should be read from (or even better define) the YAML file
        nbvm = Nlayers*np.ones(len(msTyp), np.int)
        thtv = np.tile(np.logspace(np.log10(botLayer), np.log10(topLayer), Nlayers)[::-1], len(msTyp))
        wvls = [0.532, 1.064] # Nλ=2
#        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] 
        meas = np.r_[np.repeat(0.01, nbvm[0]), np.repeat(0.01, nbvm[1])] 
        phi = np.repeat(0, len(thtv)) # currently we assume all observations fall within a plane
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            nowPix.addMeas(wvl, msTyp, nbvm, 0, thtv, phi, meas) # sza=sounding_angle=0=nadir
            nowPix.measVals[-1]['errorModel'] = functools.partial(addError, 'lidar05') # this must link to an error model in addError() below
    return nowPix

def addError(measNm, l, rsltFwd, edgInd):
    # if the following check ever becomes a problem see commit #6793cf7
    assert (np.diff(edgInd)[0]==np.diff(edgInd)).all(), 'Current error models assume that each measurement type has the same number of measurements at each wavelength!'
    mtch = re.match('^([A-z]+)([0-9]+)$', measNm)
    if mtch.group(1).lower() == 'polar': # measNm should be string w/ format 'polarN', where N is polarimeter number
        if int(mtch.group(2)) in [4, 7, 8]: # S-Polar04 (a-d), S-Polar07, S-Polar08
            relErr = 0.03
            relDoLPErr = 0.005
        elif int(mtch.group(2)) in [700]: # "perfect" version of polarimeter 7
            relErr = 0.000003
            relDoLPErr = 0.0000005
        elif int(mtch.group(2)) in [1, 2, 3]: # S-Polar01, S-Polar2 (a-b), S-Polar3 [1st two state ΔI as "4% to 6%" in RFI]
            relErr = 0.05
            relDoLPErr = 0.005
        elif int(mtch.group(2)) in [9]: # S-Polar09
            relErr = 0.03
            relDoLPErr = 0.003
        elif int(mtch.group(2)) in [5]: # S-Polar05
            relErr = 0.02
            relDoLPErr = 0.003
        else:
            assert False, 'No error model found for %s!' % measNm # S-Polar06 has DoLP dependent ΔDoLP
        trueSimI = rsltFwd['fit_I'][:,l]
        trueSimQ = rsltFwd['fit_Q'][:,l]
        trueSimU = rsltFwd['fit_U'][:,l]
        noiseVctI = np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimI))
        fwdSimI = trueSimI*noiseVctI
        fwdSimQ = trueSimQ*noiseVctI # we scale Q and U too to keep q, u and DoLP inline with truth
        fwdSimU = trueSimU*noiseVctI
        dpRnd = np.random.normal(size=len(trueSimI))*relDoLPErr
        dPol = trueSimI*np.sqrt((trueSimQ**2+trueSimU**2)/(trueSimQ**4+trueSimU**4)) # true is fine b/c noiceVctI factors cancel themselves out
        dpRnd = np.random.normal(size=len(trueSimI))*relDoLPErr
        fwdSimQ = fwdSimQ*(1+dpRnd*dPol)
        dpRnd = np.random.normal(size=len(trueSimI))*relDoLPErr # Q and U errors are NOT correlated here (this is what we want)
        fwdSimU = fwdSimU*(1+dpRnd*dPol) 
        return np.r_[fwdSimI, fwdSimQ, fwdSimU] # safe because of ascending order check in simulateRetrieval.py 
    if mtch.group(1).lower() == 'lidar': # measNm should be string w/ format 'lidarN', where N is lidar number
        if int(mtch.group(2)) in [5]: # HSRL and depolarization 
            relErr = 0.03
#            dpolErr = 1/250
            trueSimβsca = rsltFwd['fit_VBS'][:,l] # measurement type: 39
            trueSimβext = rsltFwd['fit_VExt'][:,l] # 36
#            trueSimDPOL = rsltFwd['fit_DP'][:,l] # 35
            fwdSimβsca = relErr*trueSimβsca*np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimβsca))
            fwdSimβext = relErr*trueSimβext*np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimβext))
#            fwdSimDPOL = trueSimDPOL + dpolErr*np.random.lognormal(sigma=0.5, size=len(trueSimDPOL))
#            return np.r_[fwdSimDPOL, fwdSimβext, fwdSimβsca] # safe because of ascending order check in simulateRetrieval.py
            return np.r_[fwdSimβext, fwdSimβsca] # safe because of ascending order check in simulateRetrieval.py
#        elif int(mtch.group(2)) in [9]: # backscatter and depol
    if mtch.group(1).lower() == 'modismisr':
        relErr = 0.03
        trueSimI = rsltFwd['fit_I'][:,l]
        noiseVctI = np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimI))
        fwdSimI = trueSimI*noiseVctI
        return np.r_[fwdSimI] # safe because of ascending order check in simulateRetrieval.py 
    assert False, 'No error model found for %s!' % measNm # S-Polar06 has DoLP dependent ΔDoLP






        
