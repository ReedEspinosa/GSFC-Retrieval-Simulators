import os
from pprint import pprint
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams.update({'xtick.direction': 'in'}); mpl.rcParams.update({'ytick.direction': 'in'});mpl.rcParams.update({'ytick.right': 'True'});mpl.rcParams.update({'xtick.top': 'True'});plt.rcParams["font.family"] = "Latin Modern Math"; plt.rcParams["mathtext.fontset"] = "cm"; plt.rcParams["figure.dpi"]=330
from scipy import interpolate
from scipy.stats import norm, gaussian_kde, ncx2, moyal
from simulateRetrieval import simulation
from glob import glob


# load the pickles

simBase = simulation(picklePath='./Examples/job/exampleSimulationTest#4_TAMU.pkl')

# wavelengths
bands = simBase.rsltFwd[0]['lambda']
bandStr = ['']+[str(i) for i in bands]

# # =============================================================================
# # Filter data
# # =============================================================================

# plot k
# loop through bands to calculate the error and bias in individual variables
aodLst = []
kFMLst = []
kCMLst = []
nFMLst = []
nCMLst = []
wlRadLst = []
fig2, ax2 = plt.subplots(6,1, figsize=(4,8))
aodWght = lambda x,τ : np.sum(x*τ)/np.sum(τ)
for i in range(len(bands)):
    # =============================================================================
    # AOD
    # =============================================================================
    true = np.asarray([rf['aod'][i] for rf in simBase.rsltFwd])
    rtrv = np.asarray([rf['aod'][i] for rf in simBase.rsltBck])
    
    Rcoef = np.corrcoef(true, rtrv)[0,1]
    RMSE = np.sqrt(np.median((true - rtrv)**2))
    bias = np.mean((rtrv-true))
    
    aodLst.append((true-rtrv).T)
    ax2[0].plot(bands[i], true, 'k*')
    ax2[0].plot(bands[i], rtrv, 'rs')
    
    # true = np.asarray([rf['k'][i] for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([aodWght(rf['k'][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltBck])[keepInd]
    # Modifying the true value based on the NDIM
    # if 5 modes present, for the case of ACCDAM-CAMP2EX four modes have one refractive index
    # and the coarse mode 'sea salt' have different value. So based on the dimension of the var
    # We can distinguish each run type and generalize the code
    tempInd = 0
    for nMode_ in [0,4]:        
        rtrv = np.asarray([rf['k'][:,i] for rf in simBase.rsltBck])[:,tempInd]
        
        if nMode_ == 0:
            # true = np.asarray([aodWght(rf['k'][:,i], rf['aodMode'][:,i]) for rf in simBase.rsltFwd])
            true = np.asarray([rf['k'][:,i] for rf in simBase.rsltFwd])[:,nMode_]
            Rcoef = np.corrcoef(true, rtrv)[0,1]
            RMSE = np.sqrt(np.median((true - rtrv)**2))
            bias = np.mean((rtrv-true))
            
            kFMLst.append((true-rtrv).T)
            ax2[1].plot(bands[i], true, 'k*')
            ax2[1].plot(bands[i], rtrv, 'rs')

        else:
            true = np.asarray([rf['k'][:,i] for rf in simBase.rsltFwd])[:,nMode_]
            kCMLst.append((true-rtrv).T)
            ax2[2].plot(bands[i], true, 'k*')
            ax2[2].plot(bands[i], rtrv, 'rs')
        tempInd += 1
    
    tempInd = 0
    for nMode_ in [0,4]:        
        rtrv = np.asarray([rf['n'][:,i] for rf in simBase.rsltBck])[:,tempInd]
        
        if nMode_ == 0:
            true = np.asarray([rf['n'][:,i] for rf in simBase.rsltFwd])[:,nMode_]  
            nFMLst.append((true-rtrv).T)
            ax2[3].plot(bands[i], true, 'k*')
            ax2[3].plot(bands[i], rtrv, 'rs')

        else:
            true = np.asarray([rf['n'][:,i] for rf in simBase.rsltFwd])[:,nMode_]
            nCMLst.append((true-rtrv).T)
            ax2[4].plot(bands[i], true, 'k*')
            ax2[4].plot(bands[i], rtrv, 'rs')
        tempInd += 1
    
    true = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltFwd])
    rtrv = np.asarray([rf['wtrSurf'][0,i] for rf in simBase.rsltBck])
    
    aodLst.append((true-rtrv).T)
    ax2[5].plot(bands[i], true, 'k*')
    ax2[5].plot(bands[i], rtrv, 'rs')
    
ax2[0].set_ylabel('AOD')
ax2[1].set_ylabel('k_fine')
ax2[2].set_ylabel('k_coarse')
ax2[3].set_ylabel('n_fine')
ax2[4].set_ylabel('n_coarse')
ax2[5].set_ylabel('WL_RAD')
ax2[5].set_xlabel('wavelength(um)')
        
  