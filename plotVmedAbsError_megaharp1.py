import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation
import os

# --- Configuration ---
waveInd = 3  # Wavelength index to analyze (e.g., 2 for 550 nm)
basePath = '/Users/wrespino/Synced/AOS/Pre-Phase-A/Polarimeter_Simulations/V2/'
filePatterns = [
    'V2megaharp1_pollutionVariable+smokeVariableLand_tFctrandLogNrm5.0_n*_nAng0.pkl',
]

# --- Helper function: volume median radius ---
def volume_median_radius(r, dVdlnr):
    # Integrate in log(r)
    logr = np.log(r)
    vol_cum = np.cumsum(dVdlnr * np.gradient(logr))
    vol_tot = vol_cum[-1]
    if vol_tot <= 0:
        return np.nan
    idx = np.searchsorted(vol_cum, 0.5 * vol_tot)
    if idx == 0:
        return r[0]
    elif idx >= len(r):
        return r[-1]
    # Linear interpolation
    x0, x1 = r[idx-1], r[idx]
    y0, y1 = vol_cum[idx-1], vol_cum[idx]
    return x0 + (0.5*vol_tot - y0) * (x1 - x0) / (y1 - y0)

# --- Second approach: mode-based fine/coarse summing ---
def get_mode_indices(n_modes):
    if n_modes == 4:
        return {'fine': [0,2], 'coarse': [1,3]}
    elif n_modes == 2:
        return {'fine': [0], 'coarse': [1]}
    else:
        return None

def sum_modes(r, dVdlnr, vol, indices):
    # r: (n_modes, n_radii), dVdlnr: (n_modes, n_radii), vol: (n_modes,)
    # returns r[0], sum over selected modes of dVdlnr (no vol multiplication)
    dists = np.zeros_like(r[0])
    for i in indices:
        dists += dVdlnr[i]
    return r[0], dists

# --- Main ---
fine_errors = []
coarse_errors = []
cutoff_radii = []

for pattern in filePatterns:
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        continue
    for fwd, bck in zip(sim.rsltFwd, sim.rsltBck):
        # Sum all modes for total PSD
        r = fwd['r'][0]  # assume all modes share the same grid
        total_fwd = np.sum(fwd['dVdlnr'], axis=0)
        total_bck = np.sum(bck['dVdlnr'], axis=0)
        # Find cutoff: minimum of total_fwd in 0.4 < r < 1.5 μm
        mask = (r > 0.4) & (r < 1.5)
        if not np.any(mask):
            continue
        min_idx = np.argmin(total_fwd[mask])
        cutoff_idx = np.arange(len(r))[mask][min_idx]
        cutoff = r[cutoff_idx]
        cutoff_radii.append(cutoff)
        # Fine: r <= cutoff, Coarse: r > cutoff
        fine_mask = r <= cutoff
        coarse_mask = r > cutoff
        # Compute vmed for each mode, both fwd and bck
        vmed_fine_fwd = volume_median_radius(r[fine_mask], total_fwd[fine_mask])
        vmed_fine_bck = volume_median_radius(r[fine_mask], total_bck[fine_mask])
        vmed_coarse_fwd = volume_median_radius(r[coarse_mask], total_fwd[coarse_mask])
        vmed_coarse_bck = volume_median_radius(r[coarse_mask], total_bck[coarse_mask])
        # Store absolute errors
        if not np.isnan(vmed_fine_fwd) and not np.isnan(vmed_fine_bck):
            fine_errors.append(abs(vmed_fine_bck - vmed_fine_fwd))
        if not np.isnan(vmed_coarse_fwd) and not np.isnan(vmed_coarse_bck):
            coarse_errors.append(abs(vmed_coarse_bck - vmed_coarse_fwd))

# --- Second approach: mode-based fine/coarse summing ---
fine_errors_mode = []
coarse_errors_mode = []

for pattern in filePatterns:
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        continue
    for fwd, bck in zip(sim.rsltFwd, sim.rsltBck):
        n_modes_fwd = fwd['dVdlnr'].shape[0]
        n_modes_bck = bck['dVdlnr'].shape[0]
        idxs_fwd = get_mode_indices(n_modes_fwd)
        idxs_bck = get_mode_indices(n_modes_bck)
        if idxs_fwd is None or idxs_bck is None:
            continue
        # Fine
        r_fine_fwd, dist_fine_fwd = sum_modes(fwd['r'], fwd['dVdlnr'], fwd['vol'], idxs_fwd['fine'])
        r_fine_bck, dist_fine_bck = sum_modes(bck['r'], bck['dVdlnr'], bck['vol'], idxs_bck['fine'])
        vmed_fine_fwd = volume_median_radius(r_fine_fwd, dist_fine_fwd)
        vmed_fine_bck = volume_median_radius(r_fine_bck, dist_fine_bck)
        # Coarse
        r_coarse_fwd, dist_coarse_fwd = sum_modes(fwd['r'], fwd['dVdlnr'], fwd['vol'], idxs_fwd['coarse'])
        r_coarse_bck, dist_coarse_bck = sum_modes(bck['r'], bck['dVdlnr'], bck['vol'], idxs_bck['coarse'])
        vmed_coarse_fwd = volume_median_radius(r_coarse_fwd, dist_coarse_fwd)
        vmed_coarse_bck = volume_median_radius(r_coarse_bck, dist_coarse_bck)
        # Store absolute errors
        if not np.isnan(vmed_fine_fwd) and not np.isnan(vmed_fine_bck):
            fine_errors_mode.append(abs(vmed_fine_bck - vmed_fine_fwd))
        if not np.isnan(vmed_coarse_fwd) and not np.isnan(vmed_coarse_bck):
            coarse_errors_mode.append(abs(vmed_coarse_bck - vmed_coarse_fwd))

# --- Collect coarse PSDs for plotting by error group (mode-based) ---
coarse_psd_fwd_lowerr = []
coarse_psd_bck_lowerr = []
coarse_psd_fwd_higherr = []
coarse_psd_bck_higherr = []
coarse_r_grid = None

for pattern in filePatterns:
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        continue
    for fwd, bck in zip(sim.rsltFwd, sim.rsltBck):
        n_modes_fwd = fwd['dVdlnr'].shape[0]
        n_modes_bck = bck['dVdlnr'].shape[0]
        idxs_fwd = get_mode_indices(n_modes_fwd)
        idxs_bck = get_mode_indices(n_modes_bck)
        if idxs_fwd is None or idxs_bck is None:
            continue
        # Coarse
        r_coarse_fwd, dist_coarse_fwd = sum_modes(fwd['r'], fwd['dVdlnr'], fwd['vol'], idxs_fwd['coarse'])
        r_coarse_bck, dist_coarse_bck = sum_modes(bck['r'], bck['dVdlnr'], bck['vol'], idxs_bck['coarse'])
        vmed_coarse_fwd = volume_median_radius(r_coarse_fwd, dist_coarse_fwd)
        vmed_coarse_bck = volume_median_radius(r_coarse_bck, dist_coarse_bck)
        abs_err = abs(vmed_coarse_bck - vmed_coarse_fwd)
        # Save r grid for plotting
        if coarse_r_grid is None:
            coarse_r_grid = r_coarse_fwd
        # Group by error
        if abs_err < 1.0:
            coarse_psd_fwd_lowerr.append(dist_coarse_fwd)
            coarse_psd_bck_lowerr.append(dist_coarse_bck)
        elif abs_err > 2.5:
            coarse_psd_fwd_higherr.append(dist_coarse_fwd)
            coarse_psd_bck_higherr.append(dist_coarse_bck)

# --- New: Histogram of 'rv' for all coarse mode pixels in fwd and bck ---
coarse_rv_fwd = []
coarse_rv_bck = []

for pattern in filePatterns:
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        continue
    for fwd, bck in zip(sim.rsltFwd, sim.rsltBck):
        # Fwd
        n_modes_fwd = fwd['rv'].shape[0]
        if n_modes_fwd == 4:
            coarse_rv_fwd.append(fwd['rv'][1])
            coarse_rv_fwd.append(fwd['rv'][3])
        elif n_modes_fwd == 2:
            coarse_rv_fwd.append(fwd['rv'][1])
        # Bck
        n_modes_bck = bck['rv'].shape[0]
        if n_modes_bck == 4:
            coarse_rv_bck.append(bck['rv'][1])
            coarse_rv_bck.append(bck['rv'][3])
        elif n_modes_bck == 2:
            coarse_rv_bck.append(bck['rv'][1])

# --- Plotting ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(fine_errors, bins=30, color='tab:blue', alpha=0.7)
plt.xlabel('Absolute Error in Fine Mode $r_{v,med}$ (μm)')
plt.ylabel('Count')
plt.title('Fine Mode')
plt.subplot(1,2,2)
plt.hist(coarse_errors, bins=30, color='tab:orange', alpha=0.7)
plt.xlabel('Absolute Error in Coarse Mode $r_{v,med}$ (μm)')
plt.ylabel('Count')
plt.title('Coarse Mode')
plt.tight_layout()
plt.savefig('AbsError_VolMedRadius_megaharp1.png')
plt.show()

# --- Plot cutoff radii histogram ---
plt.figure(figsize=(6,4))
plt.hist(cutoff_radii, bins=30, color='tab:green', alpha=0.7)
plt.xlabel('Cutoff Radius Between Modes (μm)')
plt.ylabel('Count')
plt.title('Distribution of Fine/Coarse Mode Cutoff Radii')
plt.tight_layout()
plt.savefig('CutoffRadius_Hist_megaharp1.png')
plt.show()

# --- Plotting for mode-based approach ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(fine_errors_mode, bins=30, color='tab:purple', alpha=0.7)
plt.xlabel('Abs. Error in Fine Mode $r_{v,med}$ (μm) [mode-based]')
plt.ylabel('Count')
plt.title('Fine Mode (mode-based)')
plt.subplot(1,2,2)
plt.hist(coarse_errors_mode, bins=30, color='tab:brown', alpha=0.7)
plt.xlabel('Abs. Error in Coarse Mode $r_{v,med}$ (μm) [mode-based]')
plt.ylabel('Count')
plt.title('Coarse Mode (mode-based)')
plt.tight_layout()
plt.savefig('AbsError_VolMedRadius_megaharp1_modebased.png')
plt.show()

# --- Plot coarse PSDs by error group ---
plt.figure(figsize=(12,5))
# Left: abs error < 1 μm
plt.subplot(1,2,1)
N = min(len(coarse_psd_fwd_lowerr), len(coarse_psd_bck_lowerr))
for i in range(N):
    plt.plot(coarse_r_grid, coarse_psd_fwd_lowerr[i], color='tab:blue', alpha=0.15, label='Fwd' if i == 0 else "")
    plt.plot(coarse_r_grid, coarse_psd_bck_lowerr[i], color='tab:red', alpha=0.15, label='Bck' if i == 0 else "")
plt.xscale('log')
plt.xlim(0.1, 15)
plt.xlabel('Radius (μm)')
plt.yscale('log')
plt.ylim(0.0001, 10)
plt.ylabel('dV/dlnr × vol (μm³/μm²)')
plt.title('Coarse PSDs (|Δrv| < 1 μm)')
plt.legend(loc='upper right', frameon=False)
# Right: abs error > 2.5 μm
plt.subplot(1,2,2)
N = min(len(coarse_psd_fwd_higherr), len(coarse_psd_bck_higherr))
for i in range(N):
    plt.plot(coarse_r_grid, coarse_psd_fwd_higherr[i], color='tab:blue', alpha=0.15, label='Fwd' if i == 0 else "")
    plt.plot(coarse_r_grid, coarse_psd_bck_higherr[i], color='tab:red', alpha=0.15, label='Bck' if i == 0 else "")
plt.xscale('log')
plt.xlim(0.1, 15)
plt.xlabel('Radius (μm)')
plt.yscale('log')
plt.ylim(0.0001, 10)
plt.ylabel('dV/dlnr × vol (μm³/μm²)')
plt.title('Coarse PSDs (|Δrv| > 2.5 μm)')
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig('CoarsePSD_byError_megaharp1.png')
plt.show()

# --- New: Histogram of 'rv' for all coarse mode pixels in fwd and bck ---
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.hist(coarse_rv_fwd, bins=30, color='tab:blue', alpha=0.7)
plt.xlabel('Fwd Coarse Mode $r_v$ (μm)')
plt.ylabel('Count')
plt.title('Fwd Coarse Mode $r_v$')
plt.subplot(1,3,2)
plt.hist(coarse_rv_bck, bins=30, color='tab:red', alpha=0.7)
plt.xlabel('Bck Coarse Mode $r_v$ (μm)')
plt.ylabel('Count')
plt.title('Bck Coarse Mode $r_v$')
plt.subplot(1,3,3)
plt.hist(coarse_rv_fwd, bins=30, color='tab:blue', alpha=0.5, label='Fwd')
plt.hist(coarse_rv_bck, bins=30, color='tab:red', alpha=0.5, label='Bck')
plt.xlabel('Coarse Mode $r_v$ (μm)')
plt.ylabel('Count')
plt.title('Overlay: Fwd & Bck Coarse Mode $r_v$')
plt.legend()
plt.tight_layout()
plt.savefig('CoarseMode_rv_hist_megaharp1.png')
plt.show() 