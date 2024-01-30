#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulations on user defined scenes and instruments """

# import some basic stuff
import os
import sys
import pprint
import numpy as np
import time
import tempfile
import yaml

# add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths (assumes GRASP_scripts and GSFC-Retrieval-Simulators are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath("__file__"))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GSFC-Retrieval-Simulators – add it to Python path
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))
grandParentDir = os.path.dirname(parentDir)# THIS_FILE_PATH/../../ in POSIX (this is folder that contains GRASP_scripts and GSFC-Retrieval-Simulators
sys.path.append(os.path.join(grandParentDir, "GSFC-GRASP-Python-Interface"))

# import the random number generator (uniform dstribution in logscale)
from miscFunctions import loguniform

# import top level class that peforms the retrieval simulation, defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import returnPixel function with instrument definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel

# import setupConCaseYAML function with simulated scene definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/canonicalCaseMap.py
from canonicalCaseMap import setupConCaseYAML

# import selectGeometryEntry function that can read realsitic sun-satellite geometry
from ACCP_functions import selectGeometryEntry
#--------------------------------------------#
# Local functions
#--------------------------------------------#
def runMultiple(τFactor=1.0, SZA = 30, Phi = 0, 
                conCase='campex', instrument='polar07', ymlData='None'):
    '''
    This function will run a retrieval simulations on user defined scenes and instruments
    
    Parameters
    ----------
    Input:
        τFactor: scaling factor for total AOD
        SZA: solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
        Phi: relative azimuth angle, φsolar-φsensor
        conCase: Canonical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
        instrument: polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options
        ymlData: YAML file with the retrieval simulation configuration
    Output:
        None
    '''
    instrument = instrument # polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options
    # Full path to save simulation results ass a Python pickle
    savePath = os.path.join(ymlData['default']['run']['savePathParent'], ymlData['default']['run']['MmmYY'], ymlData['default']['run']['DD'],\
        ymlData['default']['forward']['geometry'], 'Geometry', 'CoarseMode%s' % ymlData['default']['forward']['coarseMode'], \
        ymlData['default']['forward']['surfaceType']+ymlData['default']['forward']['surface'], ymlData['default']['forward']['psdType'], \
        ymlData['default']['forward']['instrument'], ymlData['default']['run']['saveFN'])
    
    savePath = savePath %(
                        ymlData['default']['run']['tagName'],
                        str(τFactor).split('.')[0],
                        str(τFactor).split('.')[1][:3],
                        int(round(SZA, 2)*100),
                        int(round(Phi, 2)*100),
                        conCase)
    
    # Full path grasp binary
    # binGRASP = '/usr/local/bin/grasp'
    binGRASP = ymlData['default']['run']['graspBin']
        
    # Full path grasp precomputed single scattering kernels
    krnlPath = ymlData['default']['run']['krnlPath']

    # Directory containing the foward and inversion YAML files you would like to use
    ymlDir = os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases")
    print('one %s' %ymlData['default']['forward']['yaml'])
    fwdModelYAMLpath = os.path.join(ymlDir, ymlData['default']['forward']['yaml']) # foward YAML file
    bckYAMLpath = os.path.join(ymlDir, ymlData['default']['retrieval']['yaml'] % (ymlData['default']['forward']['psdType'],\
                                                                       ymlData['default']['forward']['surfaceType'],\
                                                                       ymlData['default']['forward']['surface'])) # inversion YAML file

    # Other non-path related settings
    Nsims = ymlData['default']['run']['nSims'] # the number of inversion to perform, each with its own random noise
    maxCPU = ymlData['default']['run']['maxCPU'] # the number of processes to launch, effectivly the # of CPU cores you want to dedicate to the simulation
    conCase = conCase #'camp_test' # Canonical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
    SZA = SZA # solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
    Phi = Phi # relative azimuth angle, φsolar-φsensor
    if τFactor > 0:
        τFactor = τFactor # scaling factor for total AOD
    else:
        # KEY: if τFactor is negative, then the retrieval simulation will use the tfactor that is
        #      randomly generated and equally spaced in logspace between the lower and upper limits
        τFactor = loguniform(0.1, 4) # lower and upper limits for AOD

    #  <><><> END BASIC CONFIGURATION SETTINGS <><><>

    # create a dummy pixel object, conveying the measurement geometry, wavelengths, etc. (i.e. information in a GRASP SDATA file)
    nowPix = returnPixel(instrument, sza=SZA, relPhi=Phi, nowPix=None)

    # generate a YAML file with the forward model "truth" state variable values for this simulated scene
    cstmFwdYAML = setupConCaseYAML(conCase, nowPix, fwdModelYAMLpath, caseLoadFctr=τFactor)

    # Define a new instance of the simulation class for the instrument defined by nowPix (an instance of the pixel class)
    simA = rs.simulation(nowPix)
    
    # run the simulation, see below the definition of runSIM in simulateRetrieval.py for more input argument explanations
    simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                binPathGRASP=binGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, \
                rndIntialGuess=ymlData['default']['retrieval']['randmGsOn'], dryRun=False, workingFileSave=False, \
                verbose=True, delTempFiles=True)

    # print some results to the console/terminal
    if ymlData['default']['run']['verbose']:
        wavelengthIndex = 2
        wavelengthValue = simA.rsltFwd[0]['lambda'][wavelengthIndex]
        print('RMS deviations (retrieved-truth) at wavelength of %5.3f μm:' % wavelengthValue)
        pprint.pprint(simA.analyzeSim(0)[0])

# definition for looping through different geometries
def loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=18, dryRun=False):
    '''
    This function will run a retrieval simulations on user defined scenes and instruments
    
    Parameters
    ----------
    
    Input:
        runMultiple: function to run the retrieval simulation
        tau: AOD at 550 nm
        instrument: polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options
        SZA: solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
        psd_type: '2modes' or '16bins'
        phi: relative azimuth angle, φsolar-φsensor
        nFlights: number of flights used for simulation (should be 18 for full camp2ex measurements)
        dryRun: True if you want to print the command line arguments without running the retrieval simulation
    Output:
        None
    '''
    # --------------------------------------------------------------------------- #
    # Loop through different AOD
    # --------------------------------------------------------------------------- #
    for i in tau:
        loop_start_time = time.time()
        tempVAR = 0
        for j in np.r_[1:nFlights+1]:
            flight_loop_start_time = time.time()
            for k in np.r_[0]: # 0 means use information from all layers, 1 means use only information from layer 1, so on
                if ymlData['default']['forward']['coarseMode']:
                    cmStr = 'addCoarse'
                else:
                    if ymlData['default']['forward']['fixedCoarseMode']:
                        cmStr = 'fixedCoarse'
                    else:
                        cmStr = 'noCoarse'

                conCase = '%s_%s_%s_%s_flight#%.2d_layer#%.2d' %(surface, cmStr,
                                                                psdMode,
                                                                spectral,j,k)
                print('<-->'*20)
                try:
                    print('<-->'*20)
                    print('Running runRetrievalSimulationSlurm.py for τ(550nm) = %0.3f' %i)
                    if dryRun:
                        print(conCase)
                    else:
                        print(conCase) 
                        runMultiple(τFactor=i, SZA=SZA, Phi=phi,
                                	conCase=conCase, instrument=instrument,
                                    ymlData=ymlData)
                except Exception as e:
                    print('<---->'*10)
                    print('Run error: Running runRetrievalSimulationSlurm.py for τ(550nm) = %0.3f' %i)
                    print('Error message: %s' %e)
            
            print('Time to complete one loop for flight: %s'%(time.time()-flight_loop_start_time))
            if deleteTemp:
                tempVAR+=1
                # Delete the temp files after each aod loops
                if 'borg' in os.uname()[1] and tempVAR==5:
                    tempFileDir = tempfile.gettempdir()
                    os.system('rm -rf temp%s' %tempFileDir)
                    print('Clearing the temp folder in discover and sleep for 1 second')
                    time.sleep(1)
                    tempVAR = 0
        print('Time to complete one loop for AOD: %s'%(time.time()-loop_start_time))
#--------------------------------------------#
# <> BEGIN BASIC CONFIGURATION SETTINGS <>
#--------------------------------------------#
start_time = time.time()
# %% Run multiple times
# Based on the input arguments modify the parameters
if len(sys.argv) > 0:
    # AOD range
    tau = [float(sys.argv[1])]
    if len(sys.argv) > 1:
        # Instrument details
        instrument = sys.argv[2]
        if len(sys.argv) > 2:
            # Solar Zenith angle
            SZA = float(sys.argv[3])
        else:
            SZA=30
    else:
        instrument = 'polar07'
else:
    SZA=30
    tau = [0.05]
    instrument = 'polar07'
    print('AOD not given as an argument so using the 0.05 at 550 nm')

phi = 0
useRealGeometry = False
# For real geometry
if len(sys.argv) > 3:
    useRealGeometry = bool(int(sys.argv[4]))
    # if using real geometry loop through different AOD in one run
    if len(sys.argv) > 4:
        tau = [float(sys.argv[5])]
    else:
        tau1 = np.logspace(np.log10(0.01), np.log10(4), 20)
        tau = tau1[0:5]
    # read the nPCA using the sys arg
    npca = [int(float(sys.argv[1]))]
    if len(sys.argv) > 5:
        conf_ = sys.argv[6]
    else:
        conf_ = 'BiModal'

# --------------------------------------------------------------------------- #
# Load the YAML settings files for the retrieval simulation configuration
# --------------------------------------------------------------------------- #
yamlFile = '../ACCP_ArchitectureAndCanonicalCases/camp2ex-configurations.yml'
with open(yamlFile, 'r') as f:
    ymlData = yaml.load(f, Loader=yaml.FullLoader)
if conf_ == 'Triangular':
    psdMode = 'campex_tria'
    ymlData['default']['forward']['yaml'] = ymlData['default']['forward']['psdMode']['tria']['yaml']
elif conf_ == 'BiModal':
    psdMode = 'campex_bi'
    ymlData['default']['forward']['yaml'] = ymlData['default']['forward']['psdMode']['bi']['yaml']
# See if running for all configurations
if ymlData['default']['run']['config'].lower() == 'all':
    
    ymlData['default']['run']['allConfig'] = True
    
    # create a list of configurations to run
    conf_lst = list(ymlData['configurations'].keys())
else:
    ymlData['default']['run']['allConfig'] = False
    
# --------------------------------------------------------------------------- #
# Define the retrieval simulation settings
# --------------------------------------------------------------------------- #
if ymlData is None:
    print('Loading YAML file: %s resulted in an error' % yamlFile)
    sys.exit()
else:
    # Properties of the run
    surface = ymlData['default']['run']['tagName'] + '_' + ymlData['default']['run']['config'] +\
        '_' + '##'+ ymlData['default']['forward']['surface']+'_RndmGsOn-'\
         + str(ymlData['default']['retrieval']['randmGsOn'])   # for ocean either open_ocean or dark_ocean
    spectral = '$$'# ymlData['forward']['spectralInfo']   # flatfine_flatcoarse for spectrally flat RI, urban for fine urban, use nothing if it is spectrally dependent
    
    # Controlling the coarse mode, `coarseMode` is True if coarse mode is used, `fixedCoarseMode` is True if coarse mode is fixed
    if ymlData['default']['forward']['fixedCoarseMode']:
        if not ymlData['default']['forward']['coarseMode']:
            spectral = spectral + '_zerocoarse'
        spectral = spectral + '_fixedcoarse'
    else:
        if not ymlData['default']['forward']['coarseMode']:
            spectral = spectral + '_zerocoarse'
    print(surface)
    psd_type = ymlData['default']['forward']['psdType']        # '2modes' or '16bins'
    nFlights =  ymlData['default']['run']['nFlights']          # number of flights used for simulation (should be 18 for full camp2ex measurements)
deleteTemp = ymlData['default']['run']['deleteTemp']           # Flag for deleting temp files regularly
#--------------------------------------------#
# Local functions   
#--------------------------------------------#

def updateConf(cnf_, ymlData, surface, spectral):
    '''
    This function will update the configuration file based on the input arguments
    
    Parameters
    ----------
    Input:
        cnf_: configuration name
        ymlData: YAML file with the retrieval simulation configuration
        surface: surface name
        spectral: spectral name
    Output:
        ymlData: updated YAML file with the retrieval simulation configuration
    '''
    ymlData['default']['retrieval']['yaml'] = ymlData['configurations'][cnf_]['retrieval']['yaml']
    ymlData['default']['forward']['surfaceType'] = ymlData['configurations'][cnf_]['forward']['surfaceType']
    ymlData['default']['forward']['spectralInfo'] = ymlData['configurations'][cnf_]['forward']['spectralInfo']
    surface =surface.replace('All', cnf_)
    surface = surface.replace('##', ymlData['configurations'][cnf_]['forward']['surfaceType'])
    spectral = spectral.replace('$$', ymlData['configurations'][cnf_]['forward']['spectralInfo'])
    loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)


# Use real geometry and loop through different nPCA. This file is based on the geometry selected by Feng and Lans
if useRealGeometry:
    print('Running retrieval simulations for real sun-satellite geometry')
    rawAngleDir = '../../../ACCDAM/onOrbitObservationGeometryACCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '../../../ACCDAM/onOrbitObservationGeometryACCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    orbit = 'SS'
    for nPCA in npca:
        SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nPCA, orbit=orbit)
        # run the function over multiple sun-satellite geometries
        if not ymlData['default']['run']['allConfig']:
            print('zero')
            '''
            cnf_ = ymlData['default']['run']['config']
            surface =surface.replace('All', cnf_)
            surface = surface.replace('##', ymlData['configurations'][cnf_]['forward']['surfaceType'])
            spectral = spectral.replace('$$', ymlData['configurations'][cnf_]['forward']['spectralInfo'])
            loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)
            '''
            updateConf(ymlData['default']['run']['config'], ymlData, surface, spectral)
        else:
            for cnf_ in conf_lst:
                ymlData['default']['retrieval']['yaml'] = ymlData['configurations'][cnf_]['retrieval']['yaml']
                ymlData['default']['forward']['surfaceType'] = ymlData['configurations'][cnf_]['forward']['surfaceType']
                ymlData['default']['forward']['spectralInfo'] = ymlData['configurations'][cnf_]['forward']['spectralInfo']
                surface =surface.replace('All', cnf_)
                surface = surface.replace('##', ymlData['configurations'][cnf_]['forward']['surfaceType'])
                spectral = spectral.replace('$$', ymlData['configurations'][cnf_]['forward']['spectralInfo'])
                loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)
    
else:
    print('Running retrieval simulations for principal plane with fixed SZA')
    if not ymlData['default']['run']['allConfig']:
        cnf_ = ymlData['default']['run']['config']
        surface =surface.replace('All', cnf_)
        surface = surface.replace('##', ymlData['configurations'][cnf_]['forward']['surfaceType'])
        spectral = spectral.replace('$$', ymlData['configurations'][cnf_]['forward']['spectralInfo'])
        loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)
    else:
        for cnf_ in conf_lst:
                ymlData['default']['retrieval']['yaml'] = ymlData['configurations'][cnf_]['retrieval']['yaml']
                ymlData['default']['forward']['surfaceType'] = ymlData['configurations'][cnf_]['forward']['surfaceType']
                ymlData['default']['forward']['spectralInfo'] = ymlData['configurations'][cnf_]['forward']['spectralInfo']
                surface =surface.replace('All', cnf_)
                surface = surface.replace('##', ymlData['configurations'][cnf_]['forward']['surfaceType'])
                spectral = spectral.replace('$$', ymlData['configurations'][cnf_]['forward']['spectralInfo'])
                loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)

# Total time
total_time = (time.time() - start_time)/60
print('*<><><><> %4.3f minutes for total run <><><><>*' %(total_time))