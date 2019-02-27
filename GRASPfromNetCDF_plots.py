#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.join("..", "GRASP_PythonUtils"))
from runGRASP import graspDB

pltInd = 4 # pixel index to plot
polVar = 'PoI' #'P' (absolute) or 'PoI' (relative)
titleStr = 'BRDF + BPDF + Rayleigh'
rsltsFile = '/Users/wrespino/Desktop/MADCAP_test.pkl'

yVarNm = 'n'
xVarNm = 'k'
yInd = [0]
xInd = [0]
#cVarNm = 'vol'
cVarNm= False
cInd = [0]

filterSites = False
#frbdnSites = np.r_[179,1146,535,840,175] # need to uncomment/comment below to activate
keepSites = np.r_[9] #9,518,77,1,285,514,946,961


gDB = graspDB()
gDB.loadResults(rsltsFile)

gDB = graspDB()
gDB.loadResults(rsltsFile)
pltRslt = gDB.rslts[pltInd]

# PLOTTING CODE
Nwvlth = 1 if np.isscalar(pltRslt['lambda']) else len(pltRslt['lambda'])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
objs = ax[0].plot(pltRslt['sca_ang'], pltRslt['fit_I']) #, linestyle='none')
clrMat = np.tile([obj.get_color() for obj in objs], Nwvlth)
ax[0].scatter(pltRslt['sca_ang'], pltRslt['meas_I'], c=clrMat, marker='x')
ax[0].set_xlabel("Scattering Angle (deg)")
ax[0].set_ylabel("I")
if Nwvlth>1:
    ax[0].legend(['%4.2f μm' % l for l in pltRslt['lambda']],ncol=2, prop={'size': 12}, loc = 'upper left')
else:
    ax[0].legend(['GRASP', 'OSSE'])
    plt.suptitle(titleStr + ' (%4.2f μm)' % pltRslt['lambda'])
objs = ax[1].plot(pltRslt['sca_ang'], pltRslt['fit_'+polVar]) #, linestyle='none')
ax[1].scatter(pltRslt['sca_ang'], pltRslt['meas_'+polVar], c=clrMat, marker='x')
ax[1].set_xlabel("Scattering Angle (deg)")
ax[1].set_ylabel("DoLP")
plt.tight_layout()

vldInd = slice(None)
# MANUAL SCAT PLOT
plt.figure(figsize=(6,5))
gDB.scatterPlot(xVarNm, yVarNm, 1, 1, cVarNm, 1, one2oneScale=False, 
                    logScl=False, customAx=plt.gca(), Rstats=False, rsltInds=vldInd, pltLabel=os.path.basename(rsltsFile))


#minVal = np.array([])
#minValFit = np.array([])
#for rslt in rslts:
#    ind = np.argmin(np.abs(rslt['sca_ang'][:,0]-180))
#    minVal = np.append(minVal, rslt['meas_'+polVar][ind,3])
#    minValFit = np.append(minValFit, rslt['fit_'+polVar][ind,3])
#plt.figure()
#plt.hist(minVal, 60)
#plt.hist(minValFit, 60)


