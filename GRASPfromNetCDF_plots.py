#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.join("..", "GRASP_Scripts"))
from runGRASP import graspDB

pltInd = 2 # pixel index to plot
polVar = 'DOLP' #'P' (absolute), 'PoI' (relative), 'Q', 'U', 'DOLP' (only when retrieving Q & U)
titleStr = 'PLOT TITLE'
rsltsFile = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/sept01_testCase_regenerated/TEST_6Lambda_YAMLa565f773.pkl'

# Scatter Plot Parameters
showScatter = False
yVarNm = 'n'
xVarNm = 'k'
yInd = [0]
xInd = [0]
cVarNm= False
cInd = [0]


# LOAD DATA
gDB = graspDB()
gDB.loadResults(rsltsFile)
pltRslt = gDB.rslts[pltInd]

# PLOTTING CODE
Nwvlth = 1 if np.isscalar(pltRslt['lambda']) else len(pltRslt['lambda'])
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# Plot total intensity
objs = ax[0,0].plot(pltRslt['sca_ang'], pltRslt['fit_I']) #, linestyle='none')
clrMat = np.tile([obj.get_color() for obj in objs], Nwvlth)
ax[0,0].scatter(pltRslt['sca_ang'], pltRslt['meas_I'], c=clrMat, marker='.')
ax[0,0].set_ylabel("I")
if Nwvlth>1:
    ax[0,0].legend(['%4.2f μm' % l for l in pltRslt['lambda']],ncol=2, prop={'size': 12}, loc = 'upper left')
    plt.suptitle(titleStr)
else:
    ax[0,0].legend(['GRASP', 'OSSE'])
    plt.suptitle(titleStr + ' (%4.2f μm)' % pltRslt['lambda'])
# Plot total polization
if polVar.upper() == 'DOLP':
    assert 'fit_Q' in pltRslt, 'DOLP only valid when fitting absolute_polarization_components (Q,U). Use PoI or P instead.'
    fitP = np.sqrt(pltRslt['fit_Q']**2 + pltRslt['fit_U']**2)/pltRslt['fit_I']
    measP = np.sqrt(pltRslt['meas_Q']**2 + pltRslt['meas_U']**2)/pltRslt['meas_I']
else:
    fitP = pltRslt['fit_'+polVar]
    measP = pltRslt['meas_'+polVar]
objs = ax[0,1].plot(pltRslt['sca_ang'], fitP) #, linestyle='none')
ax[0,1].scatter(pltRslt['sca_ang'], measP, c=clrMat, marker='.')
ax[0,1].set_ylabel(polVar + " [absolute]")
# Plot intensity difference
relErrI = 200*(pltRslt['fit_I']-pltRslt['meas_I'])/(pltRslt['fit_I']+pltRslt['meas_I'])
objs = ax[1,0].plot(pltRslt['sca_ang'], relErrI)
ax[1,0].set_xlabel("Scattering Angle (deg)")
ax[1,0].set_ylabel("$2(I_{grasp} - I_{vildort})/(I_{grasp} + I_{vildort})$ [%]")
# Plot polarization difference
relErrI = 100*(fitP - measP)
ax[1,1].plot([pltRslt['sca_ang'].min(), pltRslt['sca_ang'].max()], [0.5,0.5], '--', color=[0.5,0.5,0.5])
ax[1,1].plot([pltRslt['sca_ang'].min(), pltRslt['sca_ang'].max()], [-0.5,-0.5], '--', color=[0.5,0.5,0.5])
objs = ax[1,1].plot(pltRslt['sca_ang'], relErrI)
ax[1,1].set_xlabel("Scattering Angle (deg)")
ax[1,1].set_ylabel("$" + polVar + "_{grasp} - " + polVar + "_{vildort}$ [%]")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# MANUAL SCAT PLOT
if not showScatter: sys.exit()
vldInd = slice(None)
plt.figure(figsize=(6,5))
gDB.scatterPlot(xVarNm, yVarNm, 1, 1, cVarNm, 1, one2oneScale=False, 
                    logScl=False, customAx=plt.gca(), Rstats=False, rsltInds=vldInd, pltLabel=os.path.basename(rsltsFile))

