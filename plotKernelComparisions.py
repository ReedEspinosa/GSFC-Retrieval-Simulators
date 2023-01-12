
# import some basic stuff
import os
import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.autolayout"] = True

parentDir = os.path.dirname(os.path.dirname(os.path.realpath("__file__"))) # obtain THIS_FILE_PATH/../ in POSIX
sys.path.append(parentDir) # that should be GSFC-Retrieval-Simulators – add it to Python path

# import top level class that peforms the retrieval simulation, defined in THIS_FILE_PATH/../simulateRetrieval.py
import simulateRetrieval as rs

from glob import glob
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
from simulateRetrieval import simulation
# matplotlibX11()
import matplotlib.pyplot as plt



Folder_names = ['Forward_Back_Spheriod',"Forward_Back_Sphere", "Forward_Back_TAMU"] #FIle names to be replace in the file path

Varfwd_y = ["p11", 'p12','p22','p33'] 
Markers = ["$O$","H","o"]
# Plotting an comparing forward values from GRASP and TAMU
# Case 1: plotting
n_mode = 2
n_files = 8
for k in range(1,n_files):
        fig1, ax = plt.subplots(nrows=n_mode,ncols = len(Varfwd_y), figsize = (25,10),dpi=330)
        fig2, ax2 = plt.subplots(nrows=2,ncols = 2,figsize = (12,4),dpi=330)
        Filenames = [f'fwd_bck_Spheriod_variable_{k}.pkl',f'fwd_bck_Sphere_variable{k}.pkl',f'fwd_bck_TAMU_variable_{k}.pkl']
        for l in range (3):
                simRsltFile_bSph = f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/{Folder_names[l]}/{Filenames[l]}'
                #simRsltFile_bSph = f'./job/Forward_Back_{File_names[l]}/fwd_bck_{File_names[l]}_variable_{k}.pkl' #path of the file 
                posFiles1 = glob(simRsltFile_bSph)
                simA_bsph = simulation(picklePath=posFiles1[0])
                # Phase function variables 
                n_mode = 2
               
                for wl in range( len(simA_bsph.rsltFwd[0]['lambda'][:2])):

                        for j in range(len(Varfwd_y)):
                                for i in range(n_mode): #plot size distribution for two modes
                                        ax[i,j].plot(simA_bsph.rsltFwd[0]['angle'][:,i,0], simA_bsph.rsltFwd[0][f'{Varfwd_y[j]}'][:,i,wl],  label = f" {Filenames[l], simA_bsph.rsltFwd[0]['lambda'][wl]}")
                        
                                        if i ==0: ax[i,j].set_title(f'{ Varfwd_y[j]}',fontsize =14)   
                                        ax[i,j].set_xlabel(f'Scattering Angle', fontsize =14)
                                        ax[i,j].set_ylabel(f'{Varfwd_y[j]}', fontsize =14)
                                        if j != 1: #log scale for all ohase function ecept p12
                                                ax[i,j].set_yscale("log")
                                        if j ==1:
                                                ax2[i,0].plot(simA_bsph.rsltFwd[0]['angle'][:,i,0], -simA_bsph.rsltFwd[0]['p12'][:,i,wl]/simA_bsph.rsltFwd[0]['p11'][:,i,wl],  label = f" {Filenames[l], simA_bsph.rsltFwd[0]['lambda'][wl]}")
                        
                                                ax2[i,1].plot(simA_bsph.rsltFwd[0]['angle'][:,i,0], simA_bsph.rsltFwd[0]['p11'][:,i,wl],label = f"{Filenames[l]}")
                                                ax2[i,1].set_yscale('log')
                                                ax2[i,0].set_xlabel(f'Scattering Angle', fontsize =14)
                                                ax2[i,1].set_xlabel(f'Scattering Angle', fontsize =14)
                                                ax2[i,1].set_ylabel(f'{Varfwd_y[j]}', fontsize =14)
                                                ax2[i,0].set_ylabel(f'-P12/P11', fontsize =14)
                                        plt.tight_layout()

        ldg1 = ax2[i,0].legend(bbox_to_anchor=(2.5, 1.5),loc='upper left',fontsize =12)
        ldg2 = ax[i,j].legend(bbox_to_anchor=(0.1, -0.35),ncol=3,fontsize =12)
                
                
        fig1.savefig(f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/FwdComparision/FwkCompare{k}.png')
#ax.savefig(f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/FwdComparison/{simRsltFile_bSph}.png')
#  plt.savefig(f'/job/Forward_Back_TAMU/fwd_bck_TAMU_variable_{i}.png')
        fig2.savefig(f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/FwdComparision/FwkCompareP{k}.png')
Folder_names_bwk = ['Forward_TAMU_Back_Spheriod',"Forward_TAMU_Back_Sphere", "Forward_Back_TAMU"] #FIle names to be replace in the file path
Filenames_bwk = [f'fwd_Tamu_bck_Spheriod_variable_{k}.pkl',f'fwd_TAMU_bck_Sphere{k}.pkl',f'fwd_bck_TAMU_variable_{k}.pkl']
color_list = ["#008080",  '#c85a53', '#DB7210' ]
for k in range(1,n_files):
        fig, ax = plt.subplots(nrows=2,ncols = 5, figsize = (30,10),dpi=330)
        for l in range(3):
                
                # Performing the inverse retrivals
                simRsltFile_bSph = f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/{Folder_names_bwk[l]}/{Filenames_bwk[l]}'
                posFiles1 = glob(simRsltFile_bSph)
                simA_bsph = simulation(picklePath=posFiles1[0])



                n_mode = 2
                Var_y = ['dVdlnr','aodMode','ssaMode','n','k']
                

                for j in range (0,5):
                        if j ==0: 
                                val_x = 'r'
                                for i in range(n_mode): #plot size distribution for two modes
                                        ax[i,j].plot(simA_bsph.rsltBck[0][f'{val_x}'][i], simA_bsph.rsltBck[0][f'{Var_y[j]}'][i], color = color_list[l],marker = f"{Markers[l]}",label = f'{Filenames_bwk[l]}')
                                        if l ==2:
                                                ax[i,j].plot(simA_bsph.rsltFwd[0][f'{val_x}'][1-i], simA_bsph.rsltFwd[0][f'{Var_y[j]}'][1-i], color = "k", marker = ".",label = "Fwd:TAMU")
                                        if i ==0:ax[i,j].set_title(f'Size Distribution', fontsize =14)
                                        ax[i,j].set_xscale("log")
                                        ax[i,j].set_xlabel(f'Radius $\mu$m ', fontsize =14)
                                        ax[i,j].set_ylabel(f'{Var_y[j]}', fontsize =14)
                                        
                                ldg3 = ax[i,j].legend(bbox_to_anchor=(3, -0.35),loc = "lower center",ncol=4,fontsize =14)
                        else: 
                                val_x = 'lambda' #plot other microphysicsal properties
                                for i in range(n_mode):
                                        ax[i,j].plot(simA_bsph.rsltBck[0][f'{val_x}'], simA_bsph.rsltBck[0][f'{Var_y[j]}'][i],color = color_list[l], marker= f"{Markers[l]}", label = f'{Filenames_bwk[l]}')
                                        if l ==2:
                                                ax[i,j].plot(simA_bsph.rsltFwd[0][f'{val_x}'], simA_bsph.rsltFwd[0][f'{Var_y[j]}'][1-i], color = "k", marker = ".", label = "Fwd:TAMU")
                                        if i ==0:ax[i,j].set_title(f'{Var_y[j]}', fontsize =14)
                                        ax[i,j].set_xlabel(f"$\lambda  \mu$m ", fontsize =14)
                                        # ax[i,j].set_xticks(simA_bsph.rsltBck[0][f'{val_x}'])
                                        ax[i,j].set_xticklabels(simA_bsph.rsltBck[0][f'{val_x}'])
                                        ax[i,j].set_ylabel(f'{Var_y[j]}', fontsize =14)
                                        #
                        plt.tight_layout()
        plt.savefig(f'/home/shared/git/GSFC-Retrieval-Simulators/Examples/job/InverseComparision/InversionCompare{k}.png')
