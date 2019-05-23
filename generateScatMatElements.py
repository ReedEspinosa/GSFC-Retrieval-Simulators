#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:40:28 2019

@author: wrespino
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join("..", "GRASP_scripts"))
from runGRASP import graspRun, pixel

# Paths to files
outFile = '/var/folders/lt/3kt1ddms7211cvcclg7yq3pc5f5bnm/T/tmpigb7yhca/bench_inversionRslts.txt'

gr = graspRun()
rslt = gr.readOutput(outFile)

plt.figure()
plt.subplot(1,2,1)
plt.plot(rslt[0]['angle'],rslt[0]['ph11osca'])
plt.yscale('log')
plt.ylabel('S11')
plt.xlabel('scattering angle (deg)')

plt.subplot(1,2,2)
plt.plot(rslt[0]['angle'],-rslt[0]['ph12osca']/rslt[0]['ph11osca'])
plt.ylabel('-S12/S11')
plt.xlabel('scattering angle (deg)')

plt.suptitle('simple aerosol case GRASP output (λ = 0.865 μm)')





