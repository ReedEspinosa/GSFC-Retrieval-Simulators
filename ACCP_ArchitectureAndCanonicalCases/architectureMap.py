import numpy as np
import os
import sys
import re
import datetime as dt
from scipy.integrate import simps
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import runGRASP as rg
from ACCP_functions import readKathysLidarσ
import functools
"""
DUMMY MEASUREMENTS: (determined by architecture, should ultimatly move to seperate scripts)
  For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
  len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
  msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
"""
def returnPixel(archName, sza=30, landPrct=100, relPhi=0, nowPix=None, concase=None, orbit=None, lidErrDir=None, lidarLayers=None):
    """Multiple instruments by archName='arch1+arch2+arch3+...' OR multiple calls with nowPix argument (tacking on extra instrument each time)"""
    if not nowPix: nowPix = rg.pixel(dt.datetime.now(), 1, 1, 0, 0, 0, landPrct) # can be called for multiple instruments, BUT they must all have unqiue wavelengths
    assert nowPix.land_prct == landPrct, 'landPrct provided did not match land percentage in nowPix'
    assert lidarLayers is None or np.all(np.diff(lidarLayers)<0), 'LidarLayers must be descending, starting with top of profile'
    if lidarLayers is None: # we use the below for lidar range bins
        botLayer = 10 # bottom layer in meters
        topLayer = 4510
        Nlayers = 10
        singProf = np.linspace(botLayer, topLayer, Nlayers)[::-1]
    else: # we use user input
        Nlayers = len(lidarLayers)
        singProf = lidarLayers
    if 'harp02' in archName.lower(): # CURRENTLY ONLY USING JUST 10 ANGLES IN RED
        msTyp = [41, 42, 43] # must be in ascending order
        thtv = np.tile([-57.0,  -44.0,  -32.0 ,  -19.0 ,  -6.0 ,  6.0,  19.0,  32.0,  44.0,  57.0], len(msTyp)) # BUG: the current values are at spacecraft not ground
        wvls = [0.441, 0.549, 0.669, 0.873]
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])]
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            errStr = 'polar0700' if 'harp0200' in archName.lower() else 'polar07'
            errModel = functools.partial(addError, errStr) # this must link to an error model in addError() below
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel)
    if 'modis' in archName.lower() or 'misr' in archName.lower(): # -- ModisMisrPolar --
        msTyp = [41, 42, 43] if 'polar' in archName.lower() else [41] # must be in ascending order
        if 'misr' in archName.lower():
            thtv = np.tile([-70.5,  -60.0,  -45.6 ,  -26.1 ,  0.1,  26.1,  45.6,  60.0,  70.5], len(msTyp))
        else: # we only use modis viewing angle
            thtv = np.tile([0.1], len(msTyp))
        if 'modis' in archName.lower():
            wvls = [0.41, 0.47, 0.55, 0.65, 0.87, 1.64, 2.13] # Nλ=7
        else: # we only use misr wavelengths
            wvls = [0.446, 0.558, 0.672, 0.867] # Nλ=7
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0])]
        if 'polar' in archName.lower(): meas = np.r_[meas, np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])]
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            errStr = 'polar07' if 'polar' in archName.lower() else 'modismisr01'
            errModel = functools.partial(addError, errStr) # this must link to an error model in addError() below
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel)
    if 'polarhemi' in archName.lower():
        msTyp = [41, 42, 43] # must be in ascending order
        azmthΑng = np.r_[0:180:10] # 0,10,...,170
        thtv = np.tile([57.0,  44.0,  32.0 ,  19.0 ,  6.0 ,  -6.0,  -19.0,  -32.0,  -44.0,  -57.0], len(msTyp)*len(azmthΑng))
        phi = np.concatenate([np.repeat(φ, 10) for φ in azmthΑng]) # 0,10,...,90
        phi = np.tile(phi, len(msTyp))
        wvls = [0.355, 0.360, 0.380, 0.410, 0.532, 0.550, 0.670, 0.870, 1.064, 1.550, 1.650] # Nλ=8
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])]
        errStr = 'polar07'
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            errModel = functools.partial(addError, errStr) # this must link to an error model in addError() below
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel)
    if 'polar07' in archName.lower(): # CURRENTLY ONLY USING JUST 10 ANGLES IN RED
        msTyp = [41, 42, 43] # must be in ascending order
        thtv = np.tile([-63.88,  -48.42,  -34.19 ,  -20.4 ,  -6.78 ,  6.78,  20.4,  34.19,  48.42,  63.88], len(msTyp)) # corresponds to 450 km orbit
        wvls = [0.360, 0.380, 0.410, 0.550, 0.670, 0.870, 1.550, 1.650] # Nλ=8
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])]
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        errStr = [y for y in archName.lower().split('+') if 'polar07' in y][0]
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            errModel = functools.partial(addError, errStr) # this must link to an error model in addError() below
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel)
    if 'polar09' in archName.lower(): # CURRENTLY ONLY USING JUST 10 ANGLES IN RED
        msTyp = [41, 42, 43] # must be in ascending order
        thtv = np.tile([-60, 0.001, 60], len(msTyp))
        wvls = [0.380, 0.410, 0.550, 0.670, 0.865] #
        nbvm = len(thtv)/len(msTyp)*np.ones(len(msTyp), np.int)
        meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])]
        phi = np.repeat(relPhi, len(thtv)) # currently we assume all observations fall within a plane
        errStr = [y for y in archName.lower().split('+') if 'polar09' in y][0]
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            errModel = functools.partial(addError, errStr) # this must link to an error model in addError() below
            nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas, errModel)
    if 'lidar05' in archName.lower() or 'lidar06' in archName.lower():
        msTyp = [[36, 39],[31]] # must be in ascending order # BUG: we took out depol (msTyp=35) b/c GRASP was throwing error (& canonical cases are spherical)
        wvls = [0.532, 1.064] # Nλ=2
        if 'lidar06' in archName.lower():
            msTyp.insert(0,[36, 39])
            wvls.insert(0,0.355)
        errStr = [y for y in archName.lower().split('+') if ('lidar05' in y or 'lidar06' in y)][0] # this must link to an error model in addError() below
        for wvl, msTyp in zip(wvls, msTyp): # This will be expanded for wavelength dependent measurement types/geometry
            nbvm = Nlayers*np.ones(len(msTyp), np.int)
            thtv = np.tile(singProf, len(msTyp))
            meas = np.block([np.repeat(n*0.001, n) for n in nbvm]) # measurement value should be type/1000
            phi = np.repeat(0, len(thtv))
            errModel = functools.partial(addError, errStr, concase=concase, orbit=orbit, lidErrDir=lidErrDir) # HSRL (LIDAR05/06)
            nowPix.addMeas(wvl, msTyp, nbvm, 0.01, thtv, phi, meas, errModel)
    if 'lidar09' in archName.lower(): # TODO: this needs to be more complex, real lidar09 has DEPOL
        msTyp = [31] # must be in ascending order # BUG: we took out depol b/c GRASP was throwing error (& canonical cases are spherical)
        nbvm = Nlayers*np.ones(len(msTyp), np.int)
        thtv = np.tile(singProf, len(msTyp))
        wvls = [0.532, 1.064] # Nλ=2
        meas = np.r_[np.repeat(0.007, nbvm[0])]
        phi = np.repeat(0, len(thtv)) # currently we assume all observations fall within a plane
        errStr = [y for y in archName.lower().split('+') if 'lidar09' in y][0]
        for wvl in wvls: # This will be expanded for wavelength dependent measurement types/geometry
            errModel = functools.partial(addError, errStr, concase=concase, orbit=orbit, lidErrDir=lidErrDir) # this must link to an error model in addError() below
            nowPix.addMeas(wvl, msTyp, nbvm, 0.01, thtv, phi, meas, errModel)
    assert nowPix.measVals, 'archName did not match any instruments!'
    return nowPix

def addError(measNm, l, rsltFwd, concase=None, orbit=None, lidErrDir=None, verbose=False):
    # if we ever want to simulate different number of measurements between the different types see commit #6793cf7
    wrngNumMeasMsg = 'Current error models assume that each measurement type has the same number of measurements at each wavelength!'
    βextLowLim = 0.01/1e6
    βscaLowLim = 2.0e-10
    βscaUprLim = 0.999
    mtch = re.match('^([A-z]+)([0-9]+)$', measNm)
    if mtch.group(1).lower() == 'polar': # measNm should be string w/ format 'polarN', where N is polarimeter number
        if int(mtch.group(2)) in [4, 7, 8]: # S-Polar04 (a-d), S-Polar07, S-Polar08
            relErr = 0.03
            relDoLPErr = 0.005
        elif int(mtch.group(2)) in [700]: # "perfect" version of polarimeter 7 (1e-4 standard noise)
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
        assert len(trueSimI)==len(trueSimQ) and len(trueSimI)==len(trueSimU), wrngNumMeasMsg
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
        vertRange = rsltFwd['RangeLidar'][:,l]
        if not np.isnan(rsltFwd['fit_LS'][:,l]).any(): # atten. backscatter
            trueSimβsca = rsltFwd['fit_LS'][:,l] # measurement type: 31
            if int(mtch.group(2)) in [500, 600, 900]:
                relErr = 0.000005 # else 1e-4 standard noise
            elif int(mtch.group(2)) in [5, 6, 9]:
                if np.isclose(rsltFwd['lambda'][l], 0.532): # must be lidar09
                    relErr = 0.07780750097002524
                elif np.isclose(rsltFwd['lambda'][l], 1.064) and int(mtch.group(2)) in [9]:
                    relErr = 0.16634947070811995
                elif np.isclose(rsltFwd['lambda'][l], 1.064) and int(mtch.group(2)) in [5, 6]:
                    relErr = 0.031179547685912617
                else:
                    assert False, 'No error values available for lidar %s wavelength %5.3 μm' % (mtch.group(2),rsltFwd['lmabda'][l])
            elif int(mtch.group(2)) in [50, 60, 90]: # Kathy's uncertainty models
                assert concase and orbit and lidErrDir, 'Canoncial case string, orbit and lidErrDir must all be provided to use Kathys models!'
                relErr = readKathysLidarσ(lidErrDir, orbit=orbit, wavelength=rsltFwd['lambda'][l], \
                                 instrument=int(mtch.group(2))/10, concase=concase, \
                                 LidarRange=vertRange, measType='Att', verbose=verbose)
                relErr = np.abs(relErr)
                relErr[relErr>10]=10
            else:
                assert False, 'Lidar ID number %d not recognized!' % mtch.group(2)
            fwdSimβsca = trueSimβsca*np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimβsca)) # works w/ relErr as scalar or vector
            fwdSimβsca[fwdSimβsca<βscaLowLim] = βscaLowLim
#            fwdSimβscaNrm = np.r_[fwdSimβsca]/simps(fwdSimβsca, x=-vertRange) # normalize profile to unity (GRASP requirement)
            fwdSimβscaNrm = np.r_[fwdSimβsca]
            return fwdSimβscaNrm # safe because of ascending order check in simulateRetrieval.py
        elif not (np.isnan(rsltFwd['fit_VBS'][:,l]).any() or np.isnan(rsltFwd['fit_VExt'][:,l]).any()): # HSRL
            trueSimβsca = rsltFwd['fit_VBS'][:,l] # measurement type: 39
            trueSimβext = rsltFwd['fit_VExt'][:,l] # 36
            assert len(trueSimβsca)==len(trueSimβext), wrngNumMeasMsg
            if int(mtch.group(2)) in [600, 500]:# 1e-4 standard noise
                relErrβsca = 0.000005 # fraction (unitless)
                absErrβext = 17/1e10 # m-1
            elif int(mtch.group(2)) in [50, 60]: # kathy's noise models
                assert concase and orbit and lidErrDir, 'Canoncial case string, orbit and lidErrDir must all be provided to use Kathys models!'
                absErrβext = readKathysLidarσ(lidErrDir, orbit=orbit, wavelength=rsltFwd['lambda'][l], \
                                 instrument=int(mtch.group(2))/10, concase=concase, \
                                 LidarRange=vertRange, measType='Ext', verbose=verbose)
                absErrβsca = readKathysLidarσ(lidErrDir, orbit=orbit, wavelength=rsltFwd['lambda'][l], \
                                 instrument=int(mtch.group(2))/10, concase=concase, \
                                 LidarRange=vertRange, measType='Bks', verbose=verbose)
                relErrβsca = absErrβsca/trueSimβsca # not sure why I didn't just read relative error from Kathy's files...
                relErrβsca = np.abs(relErrβsca)
                relErrβsca[relErrβsca>10]=10
            elif int(mtch.group(2)) in [5, 6]: # use normal noise model
                if np.isclose(rsltFwd['lambda'][l], 0.355):
                    relErrβsca = 0.08548621115220849 #
                    absErrβext = 0.14907060980676576*trueSimβext # m-1
                elif np.isclose(rsltFwd['lambda'][l], 0.532):
                    relErrβsca = 0.035656705986020394 #
                    absErrβext = 0.19721356280281252*trueSimβext # m-1
                else:
                    assert False, 'No error values available for lidar wavelength %5.3 μm' % rsltFwd['lmabda'][l]
            else:
                assert False, 'Lidar ID number %d not recognized!' % mtch.group(2)
            fwdSimβsca = trueSimβsca*np.random.lognormal(sigma=np.log(1+relErrβsca), size=len(trueSimβsca)) # works w/ relErrβsca as scalar or vector
            fwdSimβext = trueSimβext + absErrβext*np.random.normal(size=len(trueSimβext)) # works w/ absErrβext as scalar or vector
            fwdSimβext[fwdSimβext<βextLowLim] = βextLowLim
            fwdSimβsca[fwdSimβsca<βscaLowLim] = βscaLowLim
            fwdSimβsca[fwdSimβsca>βscaUprLim] = βscaUprLim
            return np.r_[fwdSimβext, fwdSimβsca] # safe because of ascending order check in simulateRetrieval.py
        else:
            assert False, 'Lidar data type not VBS, VExt or LS!' % mtch.group(2)
    if mtch.group(1).lower() == 'modismisr':
        relErr = 0.03
        trueSimI = rsltFwd['fit_I'][:,l]
        noiseVctI = np.random.lognormal(sigma=np.log(1+relErr), size=len(trueSimI))
        fwdSimI = trueSimI*noiseVctI
        return np.r_[fwdSimI] # safe because of ascending order check in simulateRetrieval.py
    assert False, 'No error model found for %s!' % measNm # S-Polar06 has DoLP dependent ΔDoLP
