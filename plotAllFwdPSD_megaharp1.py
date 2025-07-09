import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation
import os
import glob

# --- Configuration ---
waveInd = 3  # Wavelength index to analyze (e.g., 2 for 550 nm)
basePath = '/Users/wrespino/Synced/AOS/Pre-Phase-A/Polarimeter_Simulations/V2/'
filePatterns = [
    'V2megaharp1_pollutionVariable+smokeVariableLand_tFctrandLogNrm5.0_n*_nAng0.pkl',
    # Add more megaharp1 patterns here if needed
]

# --- Load all megaharp1 forward PSDs ---
all_r = []
all_dVdlnr = []

for pattern in filePatterns:
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        continue
    for rslt in sim.rsltFwd:
        # Each rslt['r'] is (n_modes, n_radii), rslt['dVdlnr'] is (n_modes, n_radii)
        # We'll plot all modes for all pixels
        r = rslt.get('r')
        dVdlnr = rslt.get('dVdlnr')
        if r is None or dVdlnr is None:
            continue
        # r and dVdlnr are arrays of shape (n_modes, n_radii)
        for mode in range(r.shape[0]):
            all_r.append(r[mode])
            all_dVdlnr.append(dVdlnr[mode])

# --- Plotting ---
plt.figure(figsize=(7,5))
for r, dVdlnr in zip(all_r, all_dVdlnr):
    plt.plot(r, dVdlnr, color='tab:blue', alpha=0.15)
plt.xscale('log')
plt.xlabel('Radius (μm)')
plt.ylabel('dV/dlnr (μm³/μm²)')
plt.title('All Forward Modeled Size Distributions (megaharp1)')
plt.tight_layout()
plt.savefig('AllFwdPSD_megaharp1.png')
plt.show() 