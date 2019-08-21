#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:40:41 2019

@author: wrespino
"""

import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", "GRASP_scripts"))
from runGRASP import graspDB, graspRun, pixel

gObj = graspRun('/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables settings_optics_SU.v1_5_gpmom_sizedist.yml')
nowPix = pixel(730123.0, 1, 1, 0, 0, 0, 100)
nowPix.addMeas(0.350, [41], 1, 30, 0, 0, 0.123)
nowPix.addMeas(0.50, [41], 1, 30, 0, 0, 0.123)
nowPix.addMeas(0.70, [41], 1, 30, 0, 0, 0.123)
nowPix.addMeas(1.0, [41], 1, 30, 0, 0, 0.123)
nowPix.addMeas(1.5, [41], 1, 30, 0, 0, 0.123)
gObj.addPix(nowPix)
gObj.writeSDATA()


##ang = gDB.rslts[0]['angle'].squeeze()*np.pi/180
##p11Int = intrp.interp1d(ang, gDB.rslts[0]['p11'].squeeze()*np.sin(ang), 'linear')
##print(scipy.integrate.quadrature(p11Int,0,np.pi,maxiter=500))
#
#mu = 0.12
#sig = 0.55
#r = np.logspace(-2,0,1000)
#nrmFct = 1/(sig*np.sqrt(2*np.pi))
#fx = nrmFct*(r**-1)*np.exp(-((np.log(r)-np.log(mu))**2)/(2*sig**2))
#
#plt.figure
#plt.plot(r,fx)
#plt.xscale('log')