import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation
import os

# ===== CONFIGURATION =====
# Update these parameters for different analyses
basePath = '/Users/wrespino/Synced/AOS/Pre-Phase-A/Polarimeter_Simulations/V3/'
filePattern = 'V3Amegaharp4_*Land*_nAng0.pkl'  # Use glob wildcards
outputPrefix = 'megaharp4_land_VA'  # Prefix for output files
outputDir = 'plots'  # Directory to save plots

# Create output directory if it doesn't exist
os.makedirs(outputDir, exist_ok=True)

# --- Helper functions ---
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

def calculate_effective_radius(r, dVdlnr):
    """Calculate effective radius from PSD using same method as aosPolarimeterCompare.py"""
    # Calculate moments
    logr = np.log(r)
    # Second moment (area-weighted)
    moment2 = np.trapz(dVdlnr / r, logr)  # ∫(dV/dlnr)/r * dlnr = ∫N*r²*dr 
    # Third moment (volume-weighted) 
    moment3 = np.trapz(dVdlnr, logr)       # ∫dV/dlnr * dlnr = ∫N*r³*dr
    
    if moment2 <= 0:
        return np.nan
    return (3.0/4.0) * moment3 / moment2  # reff = 3V/4A = (3/4) * ∫N*r³*dr / ∫N*r²*dr

def combine_modes_effective(mode_values, mode_aods, method='aod_weighted'):
    """
    Combine multiple mode values into effective value using specified method
    
    Parameters:
    -----------
    mode_values : list or array
        Values for each mode (rv, sigma, n, k, etc.)
    mode_aods : list or array  
        AOD values for each mode (for weighting)
    method : str
        'aod_weighted' - AOD weighted average
        'sum' - simple sum (for volume concentration)
        'bulk_ssa' - exact bulk SSA calculation
    
    Returns:
    --------
    float : Combined effective value
    """
    mode_values = np.array(mode_values)
    mode_aods = np.array(mode_aods)
    
    # Filter out invalid values
    valid_mask = ~(np.isnan(mode_values) | np.isnan(mode_aods) | (mode_aods <= 0))
    if not np.any(valid_mask):
        return np.nan
        
    mode_values = mode_values[valid_mask]
    mode_aods = mode_aods[valid_mask]
    
    if method == 'sum':
        return np.sum(mode_values)
    elif method == 'aod_weighted':
        total_aod = np.sum(mode_aods)
        if total_aod <= 0:
            return np.nan
        return np.sum(mode_values * mode_aods) / total_aod
    elif method == 'bulk_ssa':
        # Bulk SSA = sum(SSA_i * AOD_i) / sum(AOD_i)
        # This is exact for SSA
        total_aod = np.sum(mode_aods)
        if total_aod <= 0:
            return np.nan
        return np.sum(mode_values * mode_aods) / total_aod
    else:
        raise ValueError(f"Unknown method: {method}")

# --- Load Data ---
print(f"Loading data with pattern: {filePattern}")
full_path_pattern = os.path.join(basePath, filePattern)
sim = simulation(picklePath=full_path_pattern)

if not sim.rsltFwd:
    print("No data loaded!")
    exit(1)

print(f"Loaded {len(sim.rsltFwd)} pixels from simulation.")

# Initialize all data containers
fine_errors = []
coarse_errors = []
cutoff_radii = []
fine_errors_mode = []
coarse_errors_mode = []
coarse_psd_fwd_lowerr = []
coarse_psd_bck_lowerr = []
coarse_psd_fwd_higherr = []
coarse_psd_bck_higherr = []
coarse_r_grid = None
coarse_rv_fwd = []
coarse_rv_bck = []
coarse_sigma_fwd = []
coarse_sigma_bck = []
fine_reff_fwd = []
fine_reff_bck = []
coarse_reff_fwd = []
coarse_reff_bck = []

# Individual mode containers
fine_rv_fwd_mode0 = []
fine_rv_fwd_mode2 = []
fine_rv_bck_mode0 = []
fine_rv_bck_mode2 = []
coarse_rv_fwd_mode1 = []
coarse_rv_fwd_mode3 = []
coarse_rv_bck_mode1 = []
coarse_rv_bck_mode3 = []

fine_sigma_fwd_mode0 = []
fine_sigma_fwd_mode2 = []
fine_sigma_bck_mode0 = []
fine_sigma_bck_mode2 = []
coarse_sigma_fwd_mode1 = []
coarse_sigma_fwd_mode3 = []
coarse_sigma_bck_mode1 = []
coarse_sigma_bck_mode3 = []

fine_vol_fwd_mode0 = []
fine_vol_fwd_mode2 = []
fine_vol_bck_mode0 = []
fine_vol_bck_mode2 = []
coarse_vol_fwd_mode1 = []
coarse_vol_fwd_mode3 = []
coarse_vol_bck_mode1 = []
coarse_vol_bck_mode3 = []

fine_n_fwd_mode0 = []
fine_n_fwd_mode2 = []
fine_n_bck_mode0 = []
fine_n_bck_mode2 = []
coarse_n_fwd_mode1 = []
coarse_n_fwd_mode3 = []
coarse_n_bck_mode1 = []
coarse_n_bck_mode3 = []

fine_aod_fwd_mode0 = []
fine_aod_fwd_mode2 = []
fine_aod_bck_mode0 = []
fine_aod_bck_mode2 = []
coarse_aod_fwd_mode1 = []
coarse_aod_fwd_mode3 = []
coarse_aod_bck_mode1 = []
coarse_aod_bck_mode3 = []

fine_ssa_fwd_mode0 = []
fine_ssa_fwd_mode2 = []
fine_ssa_bck_mode0 = []
fine_ssa_bck_mode2 = []
coarse_ssa_fwd_mode1 = []
coarse_ssa_fwd_mode3 = []
coarse_ssa_bck_mode1 = []
coarse_ssa_bck_mode3 = []

fine_k_fwd_mode0 = []
fine_k_fwd_mode2 = []
fine_k_bck_mode0 = []
fine_k_bck_mode2 = []
coarse_k_fwd_mode1 = []
coarse_k_fwd_mode3 = []
coarse_k_bck_mode1 = []
coarse_k_bck_mode3 = []

# Combined mode containers for effective values
fine_combined_fwd = []
fine_combined_bck = []
coarse_combined_fwd = []
coarse_combined_bck = []

saved_plots = []  # Track saved plot files

print("Processing pixel data...")
for fwd, bck in zip(sim.rsltFwd, sim.rsltBck):
    # === CUTOFF-BASED APPROACH ===
    r = fwd['r'][0]
    total_fwd = np.sum(fwd['dVdlnr'], axis=0)
    total_bck = np.sum(bck['dVdlnr'], axis=0)
    
    mask = (r > 0.4) & (r < 1.5)
    if np.any(mask):
        min_idx = np.argmin(total_fwd[mask])
        cutoff_idx = np.arange(len(r))[mask][min_idx]
        cutoff = r[cutoff_idx]
        cutoff_radii.append(cutoff)
        
        fine_mask = r <= cutoff
        coarse_mask = r > cutoff
        
        vmed_fine_fwd = volume_median_radius(r[fine_mask], total_fwd[fine_mask])
        vmed_fine_bck = volume_median_radius(r[fine_mask], total_bck[fine_mask])
        vmed_coarse_fwd = volume_median_radius(r[coarse_mask], total_fwd[coarse_mask])
        vmed_coarse_bck = volume_median_radius(r[coarse_mask], total_bck[coarse_mask])
        
        if not np.isnan(vmed_fine_fwd) and not np.isnan(vmed_fine_bck):
            fine_errors.append(abs(vmed_fine_bck - vmed_fine_fwd))
        if not np.isnan(vmed_coarse_fwd) and not np.isnan(vmed_coarse_bck):
            coarse_errors.append(abs(vmed_coarse_bck - vmed_coarse_fwd))

    # === MODE-BASED APPROACH ===
    n_modes_fwd = fwd['dVdlnr'].shape[0]
    n_modes_bck = bck['dVdlnr'].shape[0]
    idxs_fwd = get_mode_indices(n_modes_fwd)
    idxs_bck = get_mode_indices(n_modes_bck)
    
    if idxs_fwd is not None and idxs_bck is not None:
        # Fine modes
        r_fine_fwd, dist_fine_fwd = sum_modes(fwd['r'], fwd['dVdlnr'], fwd['vol'], idxs_fwd['fine'])
        r_fine_bck, dist_fine_bck = sum_modes(bck['r'], bck['dVdlnr'], bck['vol'], idxs_bck['fine'])
        vmed_fine_fwd = volume_median_radius(r_fine_fwd, dist_fine_fwd)
        vmed_fine_bck = volume_median_radius(r_fine_bck, dist_fine_bck)
        
        # Coarse modes
        r_coarse_fwd, dist_coarse_fwd = sum_modes(fwd['r'], fwd['dVdlnr'], fwd['vol'], idxs_fwd['coarse'])
        r_coarse_bck, dist_coarse_bck = sum_modes(bck['r'], bck['dVdlnr'], bck['vol'], idxs_bck['coarse'])
        vmed_coarse_fwd = volume_median_radius(r_coarse_fwd, dist_coarse_fwd)
        vmed_coarse_bck = volume_median_radius(r_coarse_bck, dist_coarse_bck)
        
        # Store mode-based errors
        if not np.isnan(vmed_fine_fwd) and not np.isnan(vmed_fine_bck):
            fine_errors_mode.append(abs(vmed_fine_bck - vmed_fine_fwd))
        if not np.isnan(vmed_coarse_fwd) and not np.isnan(vmed_coarse_bck):
            coarse_errors_mode.append(abs(vmed_coarse_bck - vmed_coarse_fwd))
            
        # Group coarse PSDs by error magnitude
        abs_err = abs(vmed_coarse_bck - vmed_coarse_fwd)
        if coarse_r_grid is None:
            coarse_r_grid = r_coarse_fwd
        if abs_err < 1.0:
            coarse_psd_fwd_lowerr.append(dist_coarse_fwd)
            coarse_psd_bck_lowerr.append(dist_coarse_bck)
        elif abs_err > 2.5:
            coarse_psd_fwd_higherr.append(dist_coarse_fwd)
            coarse_psd_bck_higherr.append(dist_coarse_bck)

        # === EFFECTIVE RADIUS ANALYSIS ===
        reff_fine_fwd = calculate_effective_radius(r_fine_fwd, dist_fine_fwd)
        reff_fine_bck = calculate_effective_radius(r_fine_bck, dist_fine_bck)
        reff_coarse_fwd = calculate_effective_radius(r_coarse_fwd, dist_coarse_fwd)
        reff_coarse_bck = calculate_effective_radius(r_coarse_bck, dist_coarse_bck)
        
        if not np.isnan(reff_fine_fwd) and not np.isnan(reff_fine_bck):
            fine_reff_fwd.append(reff_fine_fwd)
            fine_reff_bck.append(reff_fine_bck)
        if not np.isnan(reff_coarse_fwd) and not np.isnan(reff_coarse_bck):
            coarse_reff_fwd.append(reff_coarse_fwd)
            coarse_reff_bck.append(reff_coarse_bck)

    # === INDIVIDUAL MODE DATA COLLECTION ===
    n_modes_fwd_rv = fwd['rv'].shape[0]
    wv_idx = 3  # 4th wavelength index (0-based)
    if n_modes_fwd_rv == 4:
        # Fine modes (0, 2)
        fine_rv_fwd_mode0.append(fwd['rv'][0])
        fine_rv_fwd_mode2.append(fwd['rv'][2])
        fine_sigma_fwd_mode0.append(fwd['sigma'][0])
        fine_sigma_fwd_mode2.append(fwd['sigma'][2])
        fine_vol_fwd_mode0.append(fwd['vol'][0])
        fine_vol_fwd_mode2.append(fwd['vol'][2])
        fine_n_fwd_mode0.append(fwd['n'][0, wv_idx])
        fine_n_fwd_mode2.append(fwd['n'][2, wv_idx])
        fine_aod_fwd_mode0.append(fwd['aodMode'][0, wv_idx])
        fine_aod_fwd_mode2.append(fwd['aodMode'][2, wv_idx])
        fine_ssa_fwd_mode0.append(fwd['ssaMode'][0, wv_idx])
        fine_ssa_fwd_mode2.append(fwd['ssaMode'][2, wv_idx])
        fine_k_fwd_mode0.append(fwd['k'][0, wv_idx])
        fine_k_fwd_mode2.append(fwd['k'][2, wv_idx])
        # Coarse modes (1, 3)
        coarse_rv_fwd_mode1.append(fwd['rv'][1])
        coarse_rv_fwd_mode3.append(fwd['rv'][3])
        coarse_sigma_fwd_mode1.append(fwd['sigma'][1])
        coarse_sigma_fwd_mode3.append(fwd['sigma'][3])
        coarse_vol_fwd_mode1.append(fwd['vol'][1])
        coarse_vol_fwd_mode3.append(fwd['vol'][3])
        coarse_n_fwd_mode1.append(fwd['n'][1, wv_idx])
        coarse_n_fwd_mode3.append(fwd['n'][3, wv_idx])
        coarse_aod_fwd_mode1.append(fwd['aodMode'][1, wv_idx])
        coarse_aod_fwd_mode3.append(fwd['aodMode'][3, wv_idx])
        coarse_ssa_fwd_mode1.append(fwd['ssaMode'][1, wv_idx])
        coarse_ssa_fwd_mode3.append(fwd['ssaMode'][3, wv_idx])
        coarse_k_fwd_mode1.append(fwd['k'][1, wv_idx])
        coarse_k_fwd_mode3.append(fwd['k'][3, wv_idx])
        coarse_rv_fwd.extend([fwd['rv'][1], fwd['rv'][3]])
        coarse_sigma_fwd.extend([fwd['sigma'][1], fwd['sigma'][3]])
        
        # Calculate combined effective values for fine modes
        fine_modes_aod = [fwd['aodMode'][0, wv_idx], fwd['aodMode'][2, wv_idx]]
        fine_combined_rv_fwd = combine_modes_effective([fwd['rv'][0], fwd['rv'][2]], fine_modes_aod, 'aod_weighted')
        fine_combined_sigma_fwd = combine_modes_effective([fwd['sigma'][0], fwd['sigma'][2]], fine_modes_aod, 'aod_weighted')
        fine_combined_vol_fwd = combine_modes_effective([fwd['vol'][0], fwd['vol'][2]], fine_modes_aod, 'sum')
        fine_combined_n_fwd = combine_modes_effective([fwd['n'][0, wv_idx], fwd['n'][2, wv_idx]], fine_modes_aod, 'aod_weighted')
        fine_combined_aod_fwd = combine_modes_effective([fwd['aodMode'][0, wv_idx], fwd['aodMode'][2, wv_idx]], fine_modes_aod, 'sum')
        fine_combined_ssa_fwd = combine_modes_effective([fwd['ssaMode'][0, wv_idx], fwd['ssaMode'][2, wv_idx]], fine_modes_aod, 'bulk_ssa')
        fine_combined_k_fwd = combine_modes_effective([fwd['k'][0, wv_idx], fwd['k'][2, wv_idx]], fine_modes_aod, 'aod_weighted')
        
        # Calculate combined effective values for coarse modes
        coarse_modes_aod = [fwd['aodMode'][1, wv_idx], fwd['aodMode'][3, wv_idx]]
        coarse_combined_rv_fwd = combine_modes_effective([fwd['rv'][1], fwd['rv'][3]], coarse_modes_aod, 'aod_weighted')
        coarse_combined_sigma_fwd = combine_modes_effective([fwd['sigma'][1], fwd['sigma'][3]], coarse_modes_aod, 'aod_weighted')
        coarse_combined_vol_fwd = combine_modes_effective([fwd['vol'][1], fwd['vol'][3]], coarse_modes_aod, 'sum')
        coarse_combined_n_fwd = combine_modes_effective([fwd['n'][1, wv_idx], fwd['n'][3, wv_idx]], coarse_modes_aod, 'aod_weighted')
        coarse_combined_aod_fwd = combine_modes_effective([fwd['aodMode'][1, wv_idx], fwd['aodMode'][3, wv_idx]], coarse_modes_aod, 'sum')
        coarse_combined_ssa_fwd = combine_modes_effective([fwd['ssaMode'][1, wv_idx], fwd['ssaMode'][3, wv_idx]], coarse_modes_aod, 'bulk_ssa')
        coarse_combined_k_fwd = combine_modes_effective([fwd['k'][1, wv_idx], fwd['k'][3, wv_idx]], coarse_modes_aod, 'aod_weighted')
        
        fine_combined_fwd.append({
            'rv': fine_combined_rv_fwd, 'sigma': fine_combined_sigma_fwd, 'vol': fine_combined_vol_fwd,
            'n': fine_combined_n_fwd, 'aod': fine_combined_aod_fwd, 'ssa': fine_combined_ssa_fwd, 'k': fine_combined_k_fwd
        })
        coarse_combined_fwd.append({
            'rv': coarse_combined_rv_fwd, 'sigma': coarse_combined_sigma_fwd, 'vol': coarse_combined_vol_fwd,
            'n': coarse_combined_n_fwd, 'aod': coarse_combined_aod_fwd, 'ssa': coarse_combined_ssa_fwd, 'k': coarse_combined_k_fwd
        })
        
    elif n_modes_fwd_rv == 2:
        fine_rv_fwd_mode0.append(fwd['rv'][0])
        fine_sigma_fwd_mode0.append(fwd['sigma'][0])
        fine_vol_fwd_mode0.append(fwd['vol'][0])
        fine_n_fwd_mode0.append(fwd['n'][0, wv_idx])
        fine_aod_fwd_mode0.append(fwd['aodMode'][0, wv_idx])
        fine_ssa_fwd_mode0.append(fwd['ssaMode'][0, wv_idx])
        fine_k_fwd_mode0.append(fwd['k'][0, wv_idx])
        coarse_rv_fwd_mode1.append(fwd['rv'][1])
        coarse_sigma_fwd_mode1.append(fwd['sigma'][1])
        coarse_vol_fwd_mode1.append(fwd['vol'][1])
        coarse_n_fwd_mode1.append(fwd['n'][1, wv_idx])
        coarse_aod_fwd_mode1.append(fwd['aodMode'][1, wv_idx])
        coarse_ssa_fwd_mode1.append(fwd['ssaMode'][1, wv_idx])
        coarse_k_fwd_mode1.append(fwd['k'][1, wv_idx])
        coarse_rv_fwd.append(fwd['rv'][1])
        coarse_sigma_fwd.append(fwd['sigma'][1])
        
        # For 2-mode case, combined values are just the single mode values
        fine_combined_fwd.append({
            'rv': fwd['rv'][0], 'sigma': fwd['sigma'][0], 'vol': fwd['vol'][0],
            'n': fwd['n'][0, wv_idx], 'aod': fwd['aodMode'][0, wv_idx], 'ssa': fwd['ssaMode'][0, wv_idx], 'k': fwd['k'][0, wv_idx]
        })
        coarse_combined_fwd.append({
            'rv': fwd['rv'][1], 'sigma': fwd['sigma'][1], 'vol': fwd['vol'][1],
            'n': fwd['n'][1, wv_idx], 'aod': fwd['aodMode'][1, wv_idx], 'ssa': fwd['ssaMode'][1, wv_idx], 'k': fwd['k'][1, wv_idx]
        })
        
    # Bck data collection
    n_modes_bck_rv = bck['rv'].shape[0]
    if n_modes_bck_rv == 4:
        fine_rv_bck_mode0.append(bck['rv'][0])
        fine_rv_bck_mode2.append(bck['rv'][2])
        fine_sigma_bck_mode0.append(bck['sigma'][0])
        fine_sigma_bck_mode2.append(bck['sigma'][2])
        fine_vol_bck_mode0.append(bck['vol'][0])
        fine_vol_bck_mode2.append(bck['vol'][2])
        fine_n_bck_mode0.append(bck['n'][0, wv_idx])
        fine_n_bck_mode2.append(bck['n'][2, wv_idx])
        fine_aod_bck_mode0.append(bck['aodMode'][0, wv_idx])
        fine_aod_bck_mode2.append(bck['aodMode'][2, wv_idx])
        fine_ssa_bck_mode0.append(bck['ssaMode'][0, wv_idx])
        fine_ssa_bck_mode2.append(bck['ssaMode'][2, wv_idx])
        fine_k_bck_mode0.append(bck['k'][0, wv_idx])
        fine_k_bck_mode2.append(bck['k'][2, wv_idx])
        coarse_rv_bck_mode1.append(bck['rv'][1])
        coarse_rv_bck_mode3.append(bck['rv'][3])
        coarse_sigma_bck_mode1.append(bck['sigma'][1])
        coarse_sigma_bck_mode3.append(bck['sigma'][3])
        coarse_vol_bck_mode1.append(bck['vol'][1])
        coarse_vol_bck_mode3.append(bck['vol'][3])
        coarse_n_bck_mode1.append(bck['n'][1, wv_idx])
        coarse_n_bck_mode3.append(bck['n'][3, wv_idx])
        coarse_aod_bck_mode1.append(bck['aodMode'][1, wv_idx])
        coarse_aod_bck_mode3.append(bck['aodMode'][3, wv_idx])
        coarse_ssa_bck_mode1.append(bck['ssaMode'][1, wv_idx])
        coarse_ssa_bck_mode3.append(bck['ssaMode'][3, wv_idx])
        coarse_k_bck_mode1.append(bck['k'][1, wv_idx])
        coarse_k_bck_mode3.append(bck['k'][3, wv_idx])
        coarse_rv_bck.extend([bck['rv'][1], bck['rv'][3]])
        coarse_sigma_bck.extend([bck['sigma'][1], bck['sigma'][3]])
        
        # Calculate combined effective values for fine modes
        fine_modes_aod = [bck['aodMode'][0, wv_idx], bck['aodMode'][2, wv_idx]]
        fine_combined_rv_bck = combine_modes_effective([bck['rv'][0], bck['rv'][2]], fine_modes_aod, 'aod_weighted')
        fine_combined_sigma_bck = combine_modes_effective([bck['sigma'][0], bck['sigma'][2]], fine_modes_aod, 'aod_weighted')
        fine_combined_vol_bck = combine_modes_effective([bck['vol'][0], bck['vol'][2]], fine_modes_aod, 'sum')
        fine_combined_n_bck = combine_modes_effective([bck['n'][0, wv_idx], bck['n'][2, wv_idx]], fine_modes_aod, 'aod_weighted')
        fine_combined_aod_bck = combine_modes_effective([bck['aodMode'][0, wv_idx], bck['aodMode'][2, wv_idx]], fine_modes_aod, 'sum')
        fine_combined_ssa_bck = combine_modes_effective([bck['ssaMode'][0, wv_idx], bck['ssaMode'][2, wv_idx]], fine_modes_aod, 'bulk_ssa')
        fine_combined_k_bck = combine_modes_effective([bck['k'][0, wv_idx], bck['k'][2, wv_idx]], fine_modes_aod, 'aod_weighted')
        
        # Calculate combined effective values for coarse modes
        coarse_modes_aod = [bck['aodMode'][1, wv_idx], bck['aodMode'][3, wv_idx]]
        coarse_combined_rv_bck = combine_modes_effective([bck['rv'][1], bck['rv'][3]], coarse_modes_aod, 'aod_weighted')
        coarse_combined_sigma_bck = combine_modes_effective([bck['sigma'][1], bck['sigma'][3]], coarse_modes_aod, 'aod_weighted')
        coarse_combined_vol_bck = combine_modes_effective([bck['vol'][1], bck['vol'][3]], coarse_modes_aod, 'sum')
        coarse_combined_n_bck = combine_modes_effective([bck['n'][1, wv_idx], bck['n'][3, wv_idx]], coarse_modes_aod, 'aod_weighted')
        coarse_combined_aod_bck = combine_modes_effective([bck['aodMode'][1, wv_idx], bck['aodMode'][3, wv_idx]], coarse_modes_aod, 'sum')
        coarse_combined_ssa_bck = combine_modes_effective([bck['ssaMode'][1, wv_idx], bck['ssaMode'][3, wv_idx]], coarse_modes_aod, 'bulk_ssa')
        coarse_combined_k_bck = combine_modes_effective([bck['k'][1, wv_idx], bck['k'][3, wv_idx]], coarse_modes_aod, 'aod_weighted')
        
        fine_combined_bck.append({
            'rv': fine_combined_rv_bck, 'sigma': fine_combined_sigma_bck, 'vol': fine_combined_vol_bck,
            'n': fine_combined_n_bck, 'aod': fine_combined_aod_bck, 'ssa': fine_combined_ssa_bck, 'k': fine_combined_k_bck
        })
        coarse_combined_bck.append({
            'rv': coarse_combined_rv_bck, 'sigma': coarse_combined_sigma_bck, 'vol': coarse_combined_vol_bck,
            'n': coarse_combined_n_bck, 'aod': coarse_combined_aod_bck, 'ssa': coarse_combined_ssa_bck, 'k': coarse_combined_k_bck
        })
        
    elif n_modes_bck_rv == 2:
        fine_rv_bck_mode0.append(bck['rv'][0])
        fine_sigma_bck_mode0.append(bck['sigma'][0])
        fine_vol_bck_mode0.append(bck['vol'][0])
        fine_n_bck_mode0.append(bck['n'][0, wv_idx])
        fine_aod_bck_mode0.append(bck['aodMode'][0, wv_idx])
        fine_ssa_bck_mode0.append(bck['ssaMode'][0, wv_idx])
        fine_k_bck_mode0.append(bck['k'][0, wv_idx])
        coarse_rv_bck_mode1.append(bck['rv'][1])
        coarse_sigma_bck_mode1.append(bck['sigma'][1])
        coarse_vol_bck_mode1.append(bck['vol'][1])
        coarse_n_bck_mode1.append(bck['n'][1, wv_idx])
        coarse_aod_bck_mode1.append(bck['aodMode'][1, wv_idx])
        coarse_ssa_bck_mode1.append(bck['ssaMode'][1, wv_idx])
        coarse_k_bck_mode1.append(bck['k'][1, wv_idx])
        coarse_rv_bck.append(bck['rv'][1])
        coarse_sigma_bck.append(bck['sigma'][1])
        
        # For 2-mode case, combined values are just the single mode values
        fine_combined_bck.append({
            'rv': bck['rv'][0], 'sigma': bck['sigma'][0], 'vol': bck['vol'][0],
            'n': bck['n'][0, wv_idx], 'aod': bck['aodMode'][0, wv_idx], 'ssa': bck['ssaMode'][0, wv_idx], 'k': bck['k'][0, wv_idx]
        })
        coarse_combined_bck.append({
            'rv': bck['rv'][1], 'sigma': bck['sigma'][1], 'vol': bck['vol'][1],
            'n': bck['n'][1, wv_idx], 'aod': bck['aodMode'][1, wv_idx], 'ssa': bck['ssaMode'][1, wv_idx], 'k': bck['k'][1, wv_idx]
        })

print(f"Processed {len(sim.rsltFwd)} pixels total.")

# Calculate RMSE for effective radius
if fine_reff_fwd and fine_reff_bck:
    fine_reff_rmse = np.sqrt(np.mean([(f-b)**2 for f,b in zip(fine_reff_fwd, fine_reff_bck)]))
    print(f"Fine Mode Effective Radius RMSE: {fine_reff_rmse:.4f} μm")

if coarse_reff_fwd and coarse_reff_bck:
    coarse_reff_rmse = np.sqrt(np.mean([(f-b)**2 for f,b in zip(coarse_reff_fwd, coarse_reff_bck)]))
    print(f"Coarse Mode Effective Radius RMSE: {coarse_reff_rmse:.4f} μm")

# === PLOTTING SECTION ===
print("Generating plots...")

# 1. Volume median radius absolute error histograms (cutoff-based)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
if fine_errors:
    plt.hist(fine_errors, bins=30, color='tab:blue', alpha=0.7)
plt.xlabel(r'|$r_{v,fine}^{bck} - r_{v,fine}^{fwd}$| (μm)')
plt.ylabel('Count')
plt.title('Fine Mode rv Absolute Error (Cutoff-based)')

plt.subplot(1,2,2)
if coarse_errors:
    plt.hist(coarse_errors, bins=30, color='tab:orange', alpha=0.7)
plt.xlabel(r'|$r_{v,coarse}^{bck} - r_{v,coarse}^{fwd}$| (μm)')
plt.ylabel('Count')
plt.title('Coarse Mode rv Absolute Error (Cutoff-based)')
plt.tight_layout()
plot_file = os.path.join(outputDir, f'rv_abs_error_cutoff_hist_{outputPrefix}.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
saved_plots.append(plot_file)

# 2. Volume median radius absolute error histograms (mode-based)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
if fine_errors_mode:
    plt.hist(fine_errors_mode, bins=30, color='tab:blue', alpha=0.7)
plt.xlabel(r'|$r_{v,fine}^{bck} - r_{v,fine}^{fwd}$| (μm)')
plt.ylabel('Count')
plt.title('Fine Mode rv Absolute Error (Mode-based)')

plt.subplot(1,2,2)
if coarse_errors_mode:
    plt.hist(coarse_errors_mode, bins=30, color='tab:orange', alpha=0.7)
plt.xlabel(r'|$r_{v,coarse}^{bck} - r_{v,coarse}^{fwd}$| (μm)')
plt.ylabel('Count')
plt.title('Coarse Mode rv Absolute Error (Mode-based)')
plt.tight_layout()
plot_file = os.path.join(outputDir, f'rv_abs_error_mode_hist_{outputPrefix}.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
saved_plots.append(plot_file)

# 3. Cutoff radius histogram
if cutoff_radii:
    plt.figure(figsize=(8,6))
    plt.hist(cutoff_radii, bins=30, color='tab:green', alpha=0.7)
    plt.xlabel('Cutoff Radius (μm)')
    plt.ylabel('Count')
    plt.title('Distribution of Cutoff Radii')
    plt.tight_layout()
    plot_file = os.path.join(outputDir, f'cutoff_radius_hist_{outputPrefix}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_file)

# 4. Effective radius comparison
if fine_reff_fwd and fine_reff_bck and coarse_reff_fwd and coarse_reff_bck:
    plt.figure(figsize=(18,6))
    
    # Fine mode histogram
    plt.subplot(1,3,1)
    plt.hist(fine_reff_fwd, bins=30, alpha=0.7, label='Fwd', color='tab:blue')
    plt.hist(fine_reff_bck, bins=30, alpha=0.7, label='Bck', color='tab:red')
    plt.xlabel('Effective Radius (μm)')
    plt.ylabel('Count')
    plt.title('Fine Mode Effective Radius')
    plt.legend()
    
    # Coarse mode histogram
    plt.subplot(1,3,2)
    plt.hist(coarse_reff_fwd, bins=30, alpha=0.7, label='Fwd', color='tab:blue')
    plt.hist(coarse_reff_bck, bins=30, alpha=0.7, label='Bck', color='tab:red')
    plt.xlabel('Effective Radius (μm)')
    plt.ylabel('Count')
    plt.title('Coarse Mode Effective Radius')
    plt.legend()
    
    # Scatter plot
    plt.subplot(1,3,3)
    plt.loglog(fine_reff_fwd, fine_reff_bck, 'o', alpha=0.5, label='Fine', color='tab:blue')
    plt.loglog(coarse_reff_fwd, coarse_reff_bck, 'o', alpha=0.5, label='Coarse', color='tab:orange')
    min_val = min(min(fine_reff_fwd + coarse_reff_fwd), min(fine_reff_bck + coarse_reff_bck))
    max_val = max(max(fine_reff_fwd + coarse_reff_fwd), max(fine_reff_bck + coarse_reff_bck))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    plt.xlabel('Forward Effective Radius (μm)')
    plt.ylabel('Retrieved Effective Radius (μm)')
    plt.title('Effective Radius: Retrieved vs Forward')
    plt.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(outputDir, f'effective_radius_comparison_{outputPrefix}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_file)

# 5. Effective radius error histogram
if fine_reff_fwd and fine_reff_bck and coarse_reff_fwd and coarse_reff_bck:
    fine_reff_errors = [abs(b-f) for f,b in zip(fine_reff_fwd, fine_reff_bck)]
    coarse_reff_errors = [abs(b-f) for f,b in zip(coarse_reff_fwd, coarse_reff_bck)]
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.hist(fine_reff_errors, bins=30, color='tab:blue', alpha=0.7)
    plt.xlabel(r'|$r_{eff,fine}^{bck} - r_{eff,fine}^{fwd}$| (μm)')
    plt.ylabel('Count')
    plt.title('Fine Mode Effective Radius Error')
    
    plt.subplot(1,2,2)
    plt.hist(coarse_reff_errors, bins=30, color='tab:orange', alpha=0.7)
    plt.xlabel(r'|$r_{eff,coarse}^{bck} - r_{eff,coarse}^{fwd}$| (μm)')
    plt.ylabel('Count')
    plt.title('Coarse Mode Effective Radius Error')
    
    plt.tight_layout()
    plot_file = os.path.join(outputDir, f'effective_radius_error_hist_{outputPrefix}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_file)

# 6. Coarse PSDs grouped by error magnitude
if coarse_psd_fwd_lowerr and coarse_psd_bck_lowerr and coarse_psd_fwd_higherr and coarse_psd_bck_higherr:
    plt.figure(figsize=(12,4))
    
    # Low error pixels
    plt.subplot(1,2,1)
    for i, (fwd_psd, bck_psd) in enumerate(zip(coarse_psd_fwd_lowerr, coarse_psd_bck_lowerr)):
        plt.plot(coarse_r_grid, fwd_psd, color='tab:blue', alpha=0.15)
        plt.plot(coarse_r_grid, bck_psd, color='tab:red', alpha=0.15)
    plt.xscale('log')
    plt.xlim(0.3, 15)
    plt.xlabel('Radius (μm)')
    plt.ylabel('dV/dlnr')
    plt.title(f'Coarse Mode PSDs (Low Error, n={len(coarse_psd_fwd_lowerr)})')
    
    # High error pixels
    plt.subplot(1,2,2)
    for i, (fwd_psd, bck_psd) in enumerate(zip(coarse_psd_fwd_higherr, coarse_psd_bck_higherr)):
        plt.plot(coarse_r_grid, fwd_psd, color='tab:blue', alpha=0.15)
        plt.plot(coarse_r_grid, bck_psd, color='tab:red', alpha=0.15)
    plt.xscale('log')
    plt.xlim(0.3, 15)
    plt.xlabel('Radius (μm)')
    plt.ylabel('dV/dlnr')
    plt.title(f'Coarse Mode PSDs (High Error, n={len(coarse_psd_fwd_higherr)})')
    
    plt.tight_layout()
    plot_file = os.path.join(outputDir, f'coarse_PSD_by_error_{outputPrefix}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_file)

# Function to create three-panel plots with effective combined values
def create_three_panel_plot(prop_name, ylabel, fine_data, coarse_data, fine_combined_fwd, fine_combined_bck, coarse_combined_fwd, coarse_combined_bck):
    """Create three-panel plot with individual modes, scatter, and effective combined values"""
    
    # Fine modes
    plt.figure(figsize=(18,6))
    
    # Fine mode individual histograms
    plt.subplot(1,3,1)
    if fine_data[0]:  # mode0 fwd
        plt.hist(fine_data[0], bins=30, color='tab:blue', alpha=0.7, label='Fwd Mode 0')
    if fine_data[1]:  # mode0 bck
        plt.hist(fine_data[1], bins=30, color='tab:red', alpha=0.7, label='Bck Mode 0')
    if fine_data[2]:  # mode2 fwd
        plt.hist(fine_data[2], bins=30, color='tab:cyan', alpha=0.7, label='Fwd Mode 2')
    if fine_data[3]:  # mode2 bck
        plt.hist(fine_data[3], bins=30, color='tab:pink', alpha=0.7, label='Bck Mode 2')
    plt.xlabel(f'Individual Mode {ylabel}')
    plt.ylabel('Count')
    plt.title(f'Fine Mode Individual {prop_name.upper()}')
    plt.legend()

    # Fine mode effective scatter plot
    plt.subplot(1,3,2)
    fine_combined_fwd_vals = [item[prop_name] for item in fine_combined_fwd if not np.isnan(item[prop_name])]
    fine_combined_bck_vals = [item[prop_name] for item in fine_combined_bck if not np.isnan(item[prop_name])]
    if fine_combined_fwd_vals and fine_combined_bck_vals:
        plt.scatter(fine_combined_fwd_vals, fine_combined_bck_vals, color='tab:blue', alpha=0.6, s=15)
        min_val = min(min(fine_combined_fwd_vals), min(fine_combined_bck_vals))
        max_val = max(max(fine_combined_fwd_vals), max(fine_combined_bck_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1)
    plt.xlabel(f'Fwd Effective Fine Mode {ylabel}')
    plt.ylabel(f'Bck Effective Fine Mode {ylabel}')
    plt.title(f'Fine Mode Effective {prop_name.upper()}: Bck vs Fwd')

    # Fine mode effective values
    plt.subplot(1,3,3)
    fine_combined_fwd_vals = [item[prop_name] for item in fine_combined_fwd if not np.isnan(item[prop_name])]
    fine_combined_bck_vals = [item[prop_name] for item in fine_combined_bck if not np.isnan(item[prop_name])]
    if fine_combined_fwd_vals and fine_combined_bck_vals:
        plt.hist(fine_combined_fwd_vals, bins=30, color='tab:blue', alpha=0.7, label='Fwd Effective')
        plt.hist(fine_combined_bck_vals, bins=30, color='tab:red', alpha=0.7, label='Bck Effective')
    plt.xlabel(f'Effective Fine Mode {ylabel}')
    plt.ylabel('Count')
    plt.title(f'Fine Mode Effective {prop_name.upper()}')
    plt.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(outputDir, f'FineMode_{prop_name}_hist_{outputPrefix}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_file)

    # Coarse modes
    plt.figure(figsize=(18,6))
    
    # Coarse mode individual histograms
    plt.subplot(1,3,1)
    if coarse_data[0]:  # mode1 fwd
        plt.hist(coarse_data[0], bins=30, color='tab:orange', alpha=0.7, label='Fwd Mode 1')
    if coarse_data[1]:  # mode1 bck
        plt.hist(coarse_data[1], bins=30, color='tab:green', alpha=0.7, label='Bck Mode 1')
    if coarse_data[2]:  # mode3 fwd
        plt.hist(coarse_data[2], bins=30, color='tab:brown', alpha=0.7, label='Fwd Mode 3')
    if coarse_data[3]:  # mode3 bck
        plt.hist(coarse_data[3], bins=30, color='tab:olive', alpha=0.7, label='Bck Mode 3')
    plt.xlabel(f'Individual Mode {ylabel}')
    plt.ylabel('Count')
    plt.title(f'Coarse Mode Individual {prop_name.upper()}')
    plt.legend()

    # Coarse mode effective scatter plot
    plt.subplot(1,3,2)
    coarse_combined_fwd_vals = [item[prop_name] for item in coarse_combined_fwd if not np.isnan(item[prop_name])]
    coarse_combined_bck_vals = [item[prop_name] for item in coarse_combined_bck if not np.isnan(item[prop_name])]
    if coarse_combined_fwd_vals and coarse_combined_bck_vals:
        plt.scatter(coarse_combined_fwd_vals, coarse_combined_bck_vals, color='tab:orange', alpha=0.6, s=15)
        min_val = min(min(coarse_combined_fwd_vals), min(coarse_combined_bck_vals))
        max_val = max(max(coarse_combined_fwd_vals), max(coarse_combined_bck_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1)
    plt.xlabel(f'Fwd Effective Coarse Mode {ylabel}')
    plt.ylabel(f'Bck Effective Coarse Mode {ylabel}')
    plt.title(f'Coarse Mode Effective {prop_name.upper()}: Bck vs Fwd')

    # Coarse mode effective values
    plt.subplot(1,3,3)
    coarse_combined_fwd_vals = [item[prop_name] for item in coarse_combined_fwd if not np.isnan(item[prop_name])]
    coarse_combined_bck_vals = [item[prop_name] for item in coarse_combined_bck if not np.isnan(item[prop_name])]
    if coarse_combined_fwd_vals and coarse_combined_bck_vals:
        plt.hist(coarse_combined_fwd_vals, bins=30, color='tab:blue', alpha=0.7, label='Fwd Effective')
        plt.hist(coarse_combined_bck_vals, bins=30, color='tab:red', alpha=0.7, label='Bck Effective')
    plt.xlabel(f'Effective Coarse Mode {ylabel}')
    plt.ylabel('Count')
    plt.title(f'Coarse Mode Effective {prop_name.upper()}')
    plt.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(outputDir, f'CoarseMode_{prop_name}_hist_{outputPrefix}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_plots.append(plot_file)

# 7-13. Individual mode histograms for all properties
all_properties = [
    ('rv', 'Volume Median Radius (μm)', 
     [(fine_rv_fwd_mode0, fine_rv_bck_mode0, fine_rv_fwd_mode2, fine_rv_bck_mode2),
      (coarse_rv_fwd_mode1, coarse_rv_bck_mode1, coarse_rv_fwd_mode3, coarse_rv_bck_mode3)]),
    ('sigma', 'Standard Deviation',
     [(fine_sigma_fwd_mode0, fine_sigma_bck_mode0, fine_sigma_fwd_mode2, fine_sigma_bck_mode2),
      (coarse_sigma_fwd_mode1, coarse_sigma_bck_mode1, coarse_sigma_fwd_mode3, coarse_sigma_bck_mode3)]),
    ('vol', r'Volume Concentration (μm³/μm²)',
     [(fine_vol_fwd_mode0, fine_vol_bck_mode0, fine_vol_fwd_mode2, fine_vol_bck_mode2),
      (coarse_vol_fwd_mode1, coarse_vol_bck_mode1, coarse_vol_fwd_mode3, coarse_vol_bck_mode3)]),
    ('n', 'Refractive Index n (4th wavelength)',
     [(fine_n_fwd_mode0, fine_n_bck_mode0, fine_n_fwd_mode2, fine_n_bck_mode2),
      (coarse_n_fwd_mode1, coarse_n_bck_mode1, coarse_n_fwd_mode3, coarse_n_bck_mode3)]),
    ('aod', 'AOD (4th wavelength)', 
     [(fine_aod_fwd_mode0, fine_aod_bck_mode0, fine_aod_fwd_mode2, fine_aod_bck_mode2),
      (coarse_aod_fwd_mode1, coarse_aod_bck_mode1, coarse_aod_fwd_mode3, coarse_aod_bck_mode3)]),
    ('ssa', 'SSA (4th wavelength)',
     [(fine_ssa_fwd_mode0, fine_ssa_bck_mode0, fine_ssa_fwd_mode2, fine_ssa_bck_mode2),
      (coarse_ssa_fwd_mode1, coarse_ssa_bck_mode1, coarse_ssa_fwd_mode3, coarse_ssa_bck_mode3)]),
    ('k', 'Imaginary Refractive Index k (4th wavelength)',
     [(fine_k_fwd_mode0, fine_k_bck_mode0, fine_k_fwd_mode2, fine_k_bck_mode2),
      (coarse_k_fwd_mode1, coarse_k_bck_mode1, coarse_k_fwd_mode3, coarse_k_bck_mode3)])
]

for prop_name, ylabel, data_sets in all_properties:
    fine_data, coarse_data = data_sets
    create_three_panel_plot(prop_name, ylabel, fine_data, coarse_data, fine_combined_fwd, fine_combined_bck, coarse_combined_fwd, coarse_combined_bck)

print(f"\nPlot generation complete! Saved {len(saved_plots)} plots:")
for plot_file in saved_plots:
    print(f"  {plot_file}") 