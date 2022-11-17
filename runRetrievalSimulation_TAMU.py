#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulations on user defined scenes and instruments for different spherical and non spherical GRASP Kernels and performs comparisions  """

# import some basic stuff
import os
import sys
import pprint
import numpy as np

# add GRASP_scripts, GSFC-Retrieval-Simulators and ACCP subfolder to paths (assumes GRASP_scripts and GSFC-Retrieval-Simulators are in the same parent folder)
parentDir = os.path.dirname(os.path.dirname(os.path.realpath("__file__"))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GSFC-Retrieval-Simulators – add it to Python path
sys.path.append(os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases"))

# import top level class that peforms the retrieval simulation, defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

# import returnPixel function with instrument definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py
from architectureMap import returnPixel

# import setupConCaseYAML function with simulated scene definitions from .../GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/canonicalCaseMap.py
from canonicalCaseMap import setupConCaseYAML


from glob import glob
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
from simulateRetrieval import simulation
from miscFunctions import matplotlibX11, norm2absExtProf
# matplotlibX11()
import matplotlib.pyplot as plt



AltbinPath = '/home/shared/GRASP_GSFC/build_megaharp01_mod/bin/grasp_app' #spheriodal kernels
binGRASP= '/home/shared/GRASP_GSFC/build_megaharp01/bin/grasp_app'
# Full path grasp precomputed single scattering kernels
krnlPath = '/home/shared/GRASP_GSFC/src/retrieval/internal_files'

# Directory containing the foward and inversion YAML files you would like to use
ymlDir = os.path.join(parentDir,"ACCP_ArchitectureAndCanonicalCases")
# fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_TAMU.yml') # foward YAML file # Add _TAMU if file name for new Kernels
bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex_TAMU.yml') # inversion YAML file # Add _TAMU if file name for new Kernels

fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_TAMU.yml') # foward YAML for spheriodal kernels
try:
    aa = int(sys.argv[1])-1 #file_number in the output file name
    nCases = int(sys.argv[2]) #no fo cases to be analyzed
    print (aa,nCases)
except:
    aa = 0
    nCases = 1 #no fo cases to be analyzed

ntypeSimulation = 3# no of different type of simulation.
for i in range (ntypeSimulation* nCases): # number of simulation will be performed
        
# <><><> BEGIN BASIC CONFIGURATION SETTINGS <><><>
    loop_no = i%ntypeSimulation
    if loop_no == 0: aa = aa +1
    if loop_no == 0:  #fwd = TAMU, #bwk = Spheriod
        defineRandom = np.random.random(9)
        
        print(loop_no,aa)
    # Full path to save simulation results as a Python pickle
        savePath = f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/Forward_TAMU_Back_Spheriod/fwd_Tamu_bck_Spheriod_variable_{aa}.pkl'
        print(savePath)
        bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex.yml') # inversion YAML file # Add _TAMU if file name for new Kernels
        fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_TAMU.yml') # foward YAML for spheriodal kernels
        # Running inverse GRASP with binGRASP# bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex.yml') # inversion YAML file # Add _TAMU if file name for new Kernels
        BwkKernel1 = None
        FwdKernel1 = "TAMU"
        conCase = 'tamu_variable' #nonspherical

    # savePath = './job/exampleSimulationTest#1.pkl'
    if loop_no == 1: #fwd = TAMU, #bwk = TAMU
        # Full path to save simulation results as a Python pickle
        savePath = f'./job/Forward_Back_TAMU/fwd_bck_TAMU_variable_{aa}.pkl'
        binGRASP= '/home/shared/GRASP_GSFC/build_megaharp01_mod/bin/grasp_app'
        bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex_TAMU.yml') # inversion YAML file # Add _TAMU if file name for new Kernels
        fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_TAMU.yml') # foward YAML for spheriodal kernels
        BwkKernel1 = None
        FwdKernel1 = None
        conCase = 'tamu_variable'

    if loop_no == 2: #fwd = Spheriod, #bwk = Spheriod
        # Full path to save simulation results as a Python pickle
        savePath = f'./job/Forward_Back_Spheriod/fwd_bck_Spheriod_variable_{aa}.pkl'
        binGRASP= '/home/shared/GRASP_GSFC/build_megaharp01/bin/grasp_app'
        bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex.yml') # inversion YAML file # Add _TAMU if file name for new Kernels
        fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda.yml') # foward YAML for spheriodal kernels
        BwkKernel1 = None
        FwdKernel1 = None

    if loop_no == 4: #fwd = TAMU, #bwk = Sphere
    # Full path to save simulation results as a Python pickle
        binGRASP= '/home/shared/GRASP_GSFC/build_megaharp01/bin/grasp_app'
        savePath = f'./job/Forward_TAMU_Back_Sphere/fwd_TAMU_bck_Sphere{aa}.pkl'
        bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex_Sphere.yml') # inversion YAML file # Add _TAMU if file name for new Kernels
        fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_TAMU.yml') # foward YAML for spheriodal kernels
        BwkKernel1 = None
        FwdKernel1 = "TAMU"
        conCase = 'tamu_variable'
    
    if loop_no == 3:  #fwd = Sphere, #bwk = Sphere
    # Full path to save simulation results as a Python pickle
        binGRASP= '/home/shared/GRASP_GSFC/build_megaharp01/bin/grasp_app'
        savePath = f'./job/Forward_Back_Sphere/fwd_bck_Sphere_variable{aa}.pkl'
        bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex_Sphere.yml') # inversion YAML file # Add _TAMU if file name for new Kernels
        fwdModelYAMLpath = os.path.join(ymlDir, 'settings_FWD_IQU_POLAR_1lambda_Sphere.yml') # foward YAML for spheriodal kernels
        BwkKernel1 = None
        FwdKernel1 = None
        conCase = 'tamu_variable_sphere' # this wll run GRASP with spherical fraction of 0.999999
 
    
    # bckYAMLpath = os.path.join(ymlDir, 'settings_BCK_POLAR_2modes_Campex.yml') # inversion YAML file # Add _TAMU if file name for new Kernels

    # Other non-path related settingss
    Nsims = 1 # the number of inversions to sperform, each with its own random noise
    maxCPU = 1 # the number of processes to launch, effectivly the # of CPU cores you want to dedicate to the simulation
    
    #camp_test_flight#01_layer#00' # conanical case scene to run, case06a-k should work (see all defintions in setupConCaseYAML function)
    SZA = 30 # solar zenith (Note GRASP doesn't seem to be wild about θs=0; θs=0.1 is fine though)
    Phi = 0 # relative azimuth angle, φsolar-φsensor
    τFactor = 1 # scaling factor for total AOD
    instrument = 'megaharp01' # polar0700 has (almost) no noise, polar07 has ΔI=3%, ΔDoLP=0.5%; see returnPixel function for more options

    # %% <><><> END BASIC CONFIGURATION SETTINGS <><><>

    # create a dummy pixel object, conveying the measurement geometry, wavlengths, etc. (i.e. information in a GRASP SDATA file)
    nowPix = returnPixel(instrument, sza=SZA, relPhi=Phi, nowPix=None)

    # generate a YAML file with the forward model "truth" state variable values for this simulated scene
    cstmFwdYAML = setupConCaseYAML(conCase, nowPix, fwdModelYAMLpath, caseLoadFctr=τFactor,defineRandom=defineRandom)

    # Define a new instance of the simulation class for the instrument defined by nowPix (an instance of the pixel class)
    simA = rs.simulation(nowPix)

    try:
        # run the simulation, see below the definition of runSIM in simulateRetrieval.py for more input argument explanations
        simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, \
                    binPathGRASP=binGRASP, BwkKernel= BwkKernel1, FwdKernel= FwdKernel1, AltbinPath= AltbinPath,  intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, \
                    rndIntialGuess=False, dryRun=False, workingFileSave=True, verbose=True)

        # print some results to the console/terminal
        wavelengthIndex = 3
        wavelengthValue = simA.rsltFwd[0]['lambda'][wavelengthIndex]
        print('RMS deviations (retrieved-truth) at wavelength of %5.3f μm:' % wavelengthValue)
        pprint.pprint(simA.analyzeSim(0)[0])

    except: print(f" No Simulation was performed for{aa} , {defineRandom} ")


    # save simulated truth data to a NetCDF file
    #simA.saveSim_netCDF(savePath[:-4], verbose=True)

    # Plotting an comparing forward values from GRASP and TAMU

    # n_mode = 2
    # if loop_no:
    #     simRsltFile_bSph = f'./job/Forward_Back_Spheriod/fwd_bck_Spheriod_variable_{aa}.pkl'
    #     simRsltFile_bTAMU = f'./job/Forward_Back_TAMU/fwd_bck_TAMU_variable_{aa}.pkl'
    #     posFiles1 = glob(simRsltFile_bSph)
    #     posFiles2 = glob(simRsltFile_bTAMU)

    #     simA_bsph = simulation(picklePath=posFiles1[0])
    #     simA_bTAMU = simulation(picklePath=posFiles2[0])

    #     Varfwd_y = ["p11", 'p12','p22','p33'] # Phase function variables 

    #     fig, ax = plt.subplots(nrows=n_mode,ncols = len(Varfwd_y), figsize = (20,10),dpi=330)
    #     for wl in range( len(simA_bsph.rsltFwd[0]['lambda'])):
            

    #         for j in range(len(Varfwd_y)):
    #             for i in range(n_mode): #plot size distribution for two modes
    #                 ax[i,j].plot(simA_bsph.rsltFwd[0]['angle'][:,i,0], simA_bsph.rsltFwd[0][f'{Varfwd_y[j]}'][:,i,wl], label = f"Fwd:TAMU, bwk:Spheriod {simA_bsph.rsltFwd[0]['lambda'][wl]}")
    #                 ax[i,j].plot(simA_bTAMU.rsltFwd[0]['angle'][:,i,0], simA_bTAMU.rsltFwd[0][f'{Varfwd_y[j]}'][:,i,wl], ls = "--", label = f"Fwd: TAMU, bwk:TAMU {simA_bsph.rsltFwd[0]['lambda'][wl]}")        
    #                 ax[i,j].set_title(f'{ Varfwd_y[j]}')   
    #                 ax[i,j].set_xlabel(f'Scattering Angle')
    #                 ax[i,j].set_ylabel(f'{Varfwd_y[j]}')
    #                 if j != 1: #log scale for all ohase function ecept p12
    #                     ax[i,j].set_xscale("log")
    #     ax[i,j].legend()

    #     plt.savefig(f'/job/Forward_Back_TAMU/fwd_bck_TAMU_variable_{aa}.png')

    # if i ==3:
    #     simRsltFile_bTAMU = f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/exampleSimulationTest#1_TAMUvariable_{i-1}.pkl'
    #     simRsltFile_bSph = f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/exampleSimulationTest#1_TAMU_Spheriod{i-1}.pkl'
    #     posFiles1 = glob(simRsltFile_bSph)
    #     posFiles2 = glob(simRsltFile_bTAMU)

    #     simA_bsph = simulation(picklePath=posFiles1[0])
    #     simA_bTAMU = simulation(picklePath=posFiles2[0])

    #     fig, ax = plt.subplots(nrows=2,ncols = 5, figsize = (20,10),dpi=330)
    #     n_mode = 2
    #     Var_y = ['dVdlnr','aodMode','ssaMode','n','k']

    #     for j in range (0,5):
    #         if j ==0: 
    #             val_x = 'r'
    #             for i in range(n_mode): #plot size distribution for two modes
    #                 ax[i,j].plot(simA_bsph.rsltBck[0][f'{val_x}'][i], simA_bsph.rsltBck[0][f'{Var_y[j]}'][i], label = "Fwd:TAMU, bwk:Spheriod")
    #                 ax[i,j].plot(simA_bTAMU.rsltBck[0][f'{val_x}'][i], simA_bTAMU.rsltBck[0][f'{Var_y[j]}'][i], ls = "--", label = f"Fwd: TAMU, bwk:TAMU")
                    
    #                 ax[i,j].set_title(f'Mode - {i}')

    #                 ax[i,j].set_xscale("log")
    #                 ax[i,j].set_xlabel(f'{val_x}')
    #                 ax[i,j].set_ylabel(f'{Var_y[j]}')
    #         else: 
    #             val_x = 'lambda' #plot other microphysicsal properties
    #             for i in range(n_mode):
    #                 ax[i,j].plot(simA_bsph.rsltBck[0][f'{val_x}'], simA_bsph.rsltBck[0][f'{Var_y[j]}'][i], label = "Fwd:TAMU, bwk:Spheriod")
    #                 ax[i,j].plot(simA_bTAMU.rsltBck[0][f'{val_x}'], simA_bTAMU.rsltBck[0][f'{Var_y[j]}'][i], ls = "--", label = f"Fwd: TAMU, bwk:TAMU") 
    #                 ax[i,j].set_title(f'Mode - {i}')
    #                 ax[i,j].set_xlabel(f'{val_x}')
    #                 ax[i,j].set_ylabel(f'{Var_y[j]}')
    #     ax[i,j].legend()




