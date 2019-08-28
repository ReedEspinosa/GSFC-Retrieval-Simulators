#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:40:41 2019

@author: wrespino
"""

import numpy as np
import os
import sys
sys.path.append(os.path.join("..", "GRASP_scripts"))
import pickle
from runGRASP import graspDB, graspRun, pixel

fwdModelYAMLpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/settings_optics_SU.v1_5_gpmom_sizedist.yml'
fwdTruthSavePath = '/Users/wrespino/Desktop/testFwd.pkl'

# FAKE MEASUREMENTS (determined by architecture)
nowPix = pixel(730123.0, 1, 1, 0, 0, 0, 100)
#nowPix.addMeas(0.350, [41], 1, 30, [0, 0, 10, 20], [0, 10, 10, 10], np.repeat(0.123,4)) # wl, msTyp, nbvm, sza, thtv, phi, msrmnts
nowPix.addMeas(0.350, [41], 2, 30, [0, 10], [0, 0], np.repeat(0.123,2)) # wl, msTyp, nbvm, sza, thtv, phi, msrmnts
nowPix.addMeas(0.50, [41], 2, 30, [0, 10], [0, 0], np.repeat(0.123,2))
nowPix.addMeas(0.70, [41], 2, 30, [0, 10], [0, 0], np.repeat(0.123,2))
nowPix.addMeas(1.0, [41], 2, 30, [0, 10], [0, 0], np.repeat(0.123,2))
nowPix.addMeas(1.5, [41], 2, 30, [0, 10], [0, 0], np.repeat(0.123,2))

# THIS COULD PROBABLY MOVE TO A FUNCTION WITHIN runGRASP
gObjFwd = graspRun(fwdModelYAMLpath)
gObjFwd.addPix(nowPix)
gObjFwd.writeSDATA()
gObjFwd.runGRASP()
rsltFwd = gObjFwd.readOutput()
with open(fwdTruthSavePath, 'wb') as f:
    pickle.dump(rsltFwd, f, pickle.HIGHEST_PROTOCOL)




