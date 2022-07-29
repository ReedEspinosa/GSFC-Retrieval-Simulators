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

# add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths (assumes GRASP_scripts and GSFC-Retrieval-Simulators are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath("__file__"))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GSFC-Retrieval-Simulators – add it to Python path
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))
grandParentDir = os.path.dirname(parentDir)# THIS_FILE_PATH/../../ in POSIX (this is folder that contains GRASP_scripts and GSFC-Retrieval-Simulators
sys.path.append(os.path.join(grandParentDir, "GSFC-GRASP-Python-Interface"))


# import top level class that peforms the retrieval simulation, defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import returnPixel function with instrument definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel

# import setupConCaseYAML function with simulated scene definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/canonicalCaseMap.py
from canonicalCaseMap import setupConCaseYAML

# import selectGeometryEntry function that can read realsitic sun-satellite geometry
from ACCP_functions import selectGeometryEntry

# <><><> BEGIN BASIC CONFIGURATION SETTINGS <><><>
def runMultiple(τFactor=1.0, SZA = 30, Phi = 0, psd_type='2modes',
                conCase='campex', instrument='polar07'):
    instrument = instrument # polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options

    # Full path to save simulation results as a Python pickle
    savePath = '../../../ACCDAM/2022/Campex_Simulations/Jul2022/29/'\
        'fullAuto/withCoarseMode/%s/'\
        '%sCamp2ex_%s_AOD_%sp%s_550nm_SZA_%s_PHI_%s_%s.pkl' %( psd_type,instrument,
                                                psd_type,
                                                str(τFactor).split('.')[0],
                                                str(τFactor).split('.')[1][:3],
                                                int(round(SZA, 2)*100),
                                                int(round(Phi, 2)*100),
                                                conCase)

    # Full path grasp binary
    # binGRASP = '/usr/local/bin/grasp'
    binGRASP = '../../GRASP_GSFC/build_megaharp01/bin/grasp_app'
        
    # Full path grasp precomputed single scattering kernels
    krnlPath = '../../GRASP_GSFC/src/retrieval/internal_files'

    # Directory containing the foward and inversion YAML files you would like to use
    ymlDir = os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases")
    fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_CustomBins.yml') # foward YAML file
    bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_%s_Campex.yml' %psd_type) # inversion YAML file

    # Other non-path related settings
    Nsims = 3 # the number of inversion to perform, each with its own random noise
    maxCPU = 1 # the number of processes to launch, effectivly the # of CPU cores you want to dedicate to the simulation
    conCase = conCase #'camp_test' # conanical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
    SZA = SZA # solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
    Phi = Phi # relative azimuth angle, φsolar-φsensor
    τFactor = τFactor # scaling factor for total AOD

    # %% <><><> END BASIC CONFIGURATION SETTINGS <><><>

    # create a dummy pixel object, conveying the measurement geometry, wavlengths, etc. (i.e. information in a GRASP SDATA file)
    nowPix = returnPixel(instrument, sza=SZA, relPhi=Phi, nowPix=None)

    # generate a YAML file with the forward model "truth" state variable values for this simulated scene
    cstmFwdYAML = setupConCaseYAML(conCase, nowPix, fwdModelYAMLpath, caseLoadFctr=τFactor)

    # Define a new instance of the simulation class for the instrument defined by nowPix (an instance of the pixel class)
    simA = rs.simulation(nowPix)

    # run the simulation, see below the definition of runSIM in simulateRetrieval.py for more input argument explanations
    simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                binPathGRASP=binGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, \
                rndIntialGuess=False, dryRun=False, workingFileSave=False, verbose=True, delTempFiles=True)

    # print some results to the console/terminal
    wavelengthIndex = 2
    wavelengthValue = simA.rsltFwd[0]['lambda'][wavelengthIndex]
    print('RMS deviations (retrieved-truth) at wavelength of %5.3f μm:' % wavelengthValue)
    pprint.pprint(simA.analyzeSim(0)[0])

    # save simulated truth data to a NetCDF file
    # simA.saveSim_netCDF(savePath[:-4], verbose=True)

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

psd_type = '2modes' # '2modes' or '16bins'
# conCase = 'campex_flight#16_layer#01'#'camp_test' # conanical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
phi = 0
start_time = time.time()
useRealGeometry = False
# For real geometry
if len(sys.argv) > 3:
    useRealGeometry = bool(int(sys.argv[4]))
    # if using real geometry loop through different AOD in one run
    #if len(sys.argv) > 4:
    #    tau = [float(sys.argv[5])]
    #else:
    tau1 = np.logspace(np.log10(0.01), np.log10(2), 20)
    tau = tau1[0:5]
    # read the nPCA using the sys arg
    npca = [int(float(sys.argv[1]))]
    
nFlights = 18 # number of flights used for simulation (should be 18 for full camp2ex measurements)
deleteTemp = False # Flag for deleting temp files regularly
 
def loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=18):
    for i in tau:
        loop_start_time = time.time()
        tempVAR = 0
        for j in np.r_[1:nFlights+1]:
            flight_loop_start_time = time.time()
            for k in np.r_[0]:
                conCase = 'Coarse_campex_flight#%.2d_layer#%.2d' %(j,k)
                print('<-->'*20)
                try:
                    print('<-->'*20)
                    print('Running runRetrievalSimulation.py for τ(550nm) = %0.3f' %i)
                    runMultiple(τFactor=i, psd_type=psd_type, SZA=SZA, Phi=phi,
                                conCase=conCase, instrument=instrument)
                except Exception as e:
                    print('<---->'*10)
                    print('Run error: Running runRetrievalSimulation.py for τ(550nm) = %0.3f' %i)
                    print('Error message: %s' %e)
            print('Time to comple one loop for flight: %s'%(time.time()-flight_loop_start_time))
            if deleteTemp:
                tempVAR+=1
                # Delete the temp files after each aod loop
                if 'borg' in os.uname()[1] and tempVAR==5:
                    tempFileDir = tempfile.gettempdir()
                    os.system('rm -rf temp%s' %tempFileDir)
                    print('Clearing the temp folder in discover and sleep for 1 second')
                    time.sleep(1)
                    tempVAR = 0
        print('Time to comple one loop for AOD: %s'%(time.time()-loop_start_time))

if useRealGeometry:
    print('Running retrieval simulations for real sun-satellite geometry')
    rawAngleDir = '../../../ACCDAM/onOrbitObservationGeometryACCP/angularSampling/colarco_20200520_g5nr_pdfs'
    PCAslctMatFilePath = '../../../ACCDAM/onOrbitObservationGeometryACCP/angularSampling/FengAndLans_PCA_geometry_May2020/FengAndLans_geometry_selected_by_PC.mat'
    orbit = 'SS'
    for nPCA in npca:
        SZA, phi = selectGeometryEntry(rawAngleDir, PCAslctMatFilePath, nPCA, orbit=orbit)
        # run the function over multiple sun-satellite geometries
        loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)
    
else:
    print('Running retrieval simulations for pricipal plane')
    loop_func(runMultiple, tau, instrument, SZA, psd_type, phi, nFlights=nFlights)

# Total time
total_time = (time.time() - start_time)/60
print('*<><><><> %4.3f minutes for total run <><><><>*' %(total_time))
