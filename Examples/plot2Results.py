#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Compare the PSD from two GRASP out """

# import the library
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append('/home/shared/git/GSFC-GRASP-Python-Interface')
from runGRASP import graspRun
gr = graspRun()

# lognNormal
result_1 = gr.readOutput(customOUT='/tmp/tmp04eswwq5'+'/bench_FWD_IQU_rslts.txt')
# triangularBins
result_2 = gr.readOutput(customOUT='/tmp/tmp0d6yk1t5'+'/bench_FWD_IQU_rslts.txt')

# Plot
fig, ax = plt.subplots(nrows=2, dpi=330)
for i in range(0,5):
    #PSD
    ax[0].plot(result_1[0]['r'][i],
            result_1[0]['dVdlnr'][i], #/np.max(result_1[0]['dVdlnr'][i]),
            'C%d-' %i, label= 'LN-mode#%d' %(i+1))
    ax[0].plot(result_2[0]['r'][i],
            result_2[0]['dVdlnr'][i],#/np.max(result_2[0]['dVdlnr'][i]),
            'C%d--' %i)
    
    # AOD
    ax[1].plot(result_2[0]['lambda'],
            result_1[0]['aodMode'][i], 
            'C%d-' %i, label= 'LN-mode#%d' %(i+1))
    ax[1].plot(result_2[0]['lambda'],
            result_2[0]['aodMode'][i],
            'C%d--' %i)
ax[1].set_title('CMF = %0.3f, %0.3f, AOD = %0.3f, %0.3f' \
        %(result_1[0]['aodMode'][4][2]/result_1[0]['aod'][2],
        result_2[0]['aodMode'][4][2]/result_2[0]['aod'][2],
        result_1[0]['aod'][2], result_2[0]['aod'][2]))

ax[0].set_xscale('log')
ax[0].set_xlabel(r'Radius ($\mu m$)')
ax[0].set_ylabel('dVdlnr')
ax[1].set_ylabel('aodMode')
ax[1].set_xlabel('Wavelength')
ax[0].legend()
plt.tight_layout()
plt.savefig('Comparison_LN_Tria.png', dpi=330)