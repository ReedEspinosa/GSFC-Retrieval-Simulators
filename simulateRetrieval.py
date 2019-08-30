#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:40:41 2019

@author: wrespino
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join("..", "GRASP_scripts"))
import pickle
from runGRASP import graspRun, pixel

fwdModelYAMLpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/canonical_cases/settings_FWD_IQU_5lambda_Template.yml'
bckYAMLpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/A-CCP/canonical_cases/settings_BCK_IQU_5lambda_Template.yml'
savePath = '/Users/wrespino/Desktop/testFwd_SPH90.pkl'
Nsims = 30

# DUMMY MEASUREMENTS (determined by architecture, should ultimatly move to seperate scripts)
#  For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
#  len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
#  msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
measNm = ['I', 'Q', 'U'] # should match GRASP output (e.g. fit_I -> 'I')
msTyp = [41, 42, 43]
nbvm = [9, 9, 9]
sza = 30
thtv = np.tile([70.5, 60.0, 45.6, 26.1, 0, 26.1, 45.6, 60.0, 70.5], len(nbvm))
phi = np.tile([0, 0, 0, 0, 0, 180, 180, 180, 180], len(nbvm))
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] 
nowPix = pixel(730123.0, 1, 1, 0, 0, 0, 100)
nowPix.addMeas(0.350, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(0.50, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(0.70, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(1.0, msTyp, nbvm, sza, thtv, phi, meas)
nowPix.addMeas(1.5, msTyp, nbvm, sza, thtv, phi, meas)
nbvm = np.tile(nbvm, [len(nowPix.measVals),1]) # this assumes nbvm(wvlth,msType) is same at all lambda
measNm = np.tile(measNm, [len(nowPix.measVals),1]) # this assumes nbvm(wvlth,msType) is same at all lambda

def addError(meas, measNm):
    if measNm=='I':
        return meas*(1+np.random.normal()*0.03)
    elif measNm=='Q' or measNm=='U':
        return meas*(1+np.random.normal()*0.005)
    else:
        assert False, 'Unkown measurement string, can not add error!'

# RUN THE FOWARD MODEL
gObjFwd = graspRun(fwdModelYAMLpath)
gObjFwd.addPix(nowPix)
gObjFwd.runGRASP()
rsltFwd = gObjFwd.readOutput()

# ADD NOISE AND PERFORM RETRIEVALS
gObjBck = graspRun(bckYAMLpath)
for i in range(Nsims):
    nowPix.dtNm = copy.copy(nowPix.dtNm)
    for l, msDct in enumerate(nowPix.measVals):
        edgInd = np.r_[0, np.cumsum(nbvm[l,:])]
        msDct['measurements'] = copy.copy(msDct['measurements']) # we are going to write to this
        for i in range(len(nbvm[l,:])):
            fwdSim = rsltFwd[0]['fit_'+measNm[l,i]][:,l]
            fwdSim = copy.copy(addError(fwdSim,measNm[l,i]))
            msDct['measurements'][edgInd[i]:edgInd[i+1]] = fwdSim
    nowPix.dtNm = nowPix.dtNm+1 # otherwise GRASP will whine
    gObjBck.addPix(nowPix)
gObjBck.runGRASP()
rsltBck = gObjBck.readOutput()

rsltBck.append(rsltFwd[0]) # the last element in the list corresponds to the "truth"
with open(savePath, 'wb') as f:
    pickle.dump(rsltBck, f, pickle.HIGHEST_PROTOCOL)