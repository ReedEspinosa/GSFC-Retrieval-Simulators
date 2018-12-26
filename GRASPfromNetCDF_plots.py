#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

#pltInd = 94 # pixel index to plot (especially bad one)
pltInd = 48 # pixel index to plot
polVar = 'PoI' #'P' or 'PoI'
#titleStr = 'BRDF + Rayleigh'
#titleStr = 'Rayleigh Only'
titleStr = 'BRDF + BPDF + Rayleigh'

outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/harp_inversionRslts_16bin_looseControl.txt'
#outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/harp_inversionRslts_16bin_looseSlwRT.txt'
#outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/harp_inversionRslts_LN_bestContol.txt'
#outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/harp_inversionRslts_LN_looseContol.txt'
#outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/harp_inversionRslts_LN_looseContol_1RI.txt'
#outFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/graspWorking/harp_inversionRslts_LN_looseslwRT.txt'

rslts = graspObjs[0].readOutput(outFile)

# PLOTTING CODE
Nwvlth = 1 if np.isscalar(rslts[pltInd]['lambda']) else len(rslts[pltInd]['lambda'])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
objs = ax[0].plot(rslts[pltInd]['sca_ang'], rslts[pltInd]['fit_I']) #, linestyle='none')
clrMat = np.tile([obj.get_color() for obj in objs], Nwvlth)
ax[0].scatter(rslts[pltInd]['sca_ang'], rslts[pltInd]['meas_I'], c=clrMat, marker='x')
ax[0].set_xlabel("Scattering Angle (deg)")
ax[0].set_ylabel("I")
if Nwvlth>1:
    ax[0].legend(['%4.2f μm' % l for l in rslts[pltInd]['lambda']],ncol=2, prop={'size': 12}, loc = 'upper left')
else:
    ax[0].legend(['GRASP', 'OSSE'])
    plt.suptitle(titleStr + ' (%4.2f μm)' % rslts[pltInd]['lambda'])
objs = ax[1].plot(rslts[pltInd]['sca_ang'], rslts[pltInd]['fit_'+polVar]) #, linestyle='none')
ax[1].scatter(rslts[pltInd]['sca_ang'], rslts[pltInd]['meas_'+polVar], c=clrMat, marker='x')
ax[1].set_xlabel("Scattering Angle (deg)")
ax[1].set_ylabel("DoLP")
plt.tight_layout()

#minVal = np.array([])
#minValFit = np.array([])
#for rslt in rslts:
#    ind = np.argmin(np.abs(rslt['sca_ang'][:,0]-180))
#    minVal = np.append(minVal, rslt['meas_'+polVar][ind,3])
#    minValFit = np.append(minValFit, rslt['fit_'+polVar][ind,3])
#plt.figure()
#plt.hist(minVal, 60)
#plt.hist(minValFit, 60)


