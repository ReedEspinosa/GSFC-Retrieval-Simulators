import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation
import os
import glob
from scipy.stats import gaussian_kde

def calculate_pm25(rslt_dict, layer_height_m=1000, density_g_cm3=1.0):
    """
    Calculates PM2.5 from a GRASP result dictionary assuming water density and a fixed layer height.
    PM2.5 is returned in units of μg/m^3. This version interpolates to the 2.5 um diameter cutoff for accuracy.
    """
    r_um = rslt_dict.get('r')
    dVdlnr = rslt_dict.get('dVdlnr')
    cutoff_radius = 1.25 # PM2.5 -> diameter < 2.5 um

    if r_um is None or dVdlnr is None:
        return np.nan

    total_volume_col = 0
    # Loop over each aerosol mode, as each can have different radius bins
    for i in range(r_um.shape[0]):
        r_mode = r_um[i, :]
        dVdlnr_mode = dVdlnr[i, :]

        # Find where the cutoff radius falls within the mode's radius bins
        cutoff_idx_search = np.where(r_mode > cutoff_radius)[0]

        if cutoff_idx_search.size > 0:
            # --- Interpolation needed ---
            cutoff_idx = cutoff_idx_search[0]
            prev_idx = cutoff_idx - 1

            if prev_idx < 0: # All radii are larger than cutoff
                continue

            # Linearly interpolate dVdlnr at the cutoff radius
            interp_r = np.array([r_mode[prev_idx], r_mode[cutoff_idx]])
            interp_dVdlnr = np.array([dVdlnr_mode[prev_idx], dVdlnr_mode[cutoff_idx]])
            interp_val = np.interp(cutoff_radius, interp_r, interp_dVdlnr)
            
            # Create new arrays for integration up to the exact cutoff
            r_pm25 = np.append(r_mode[:cutoff_idx], cutoff_radius)
            dVdlnr_pm25 = np.append(dVdlnr_mode[:cutoff_idx], interp_val)
        
        else:
            # --- No interpolation needed, all radii are within the PM2.5 range ---
            r_pm25 = r_mode
            dVdlnr_pm25 = dVdlnr_mode
            
        # Integrate dV/dlnr over lnr to get total volume for this mode
        if len(r_pm25) > 1:
            total_volume_col += np.trapz(dVdlnr_pm25, np.log(r_pm25))

    # Convert total column volume (μm^3/μm^2) to mass concentration (μg/m^3)
    # V_col[um] * (1e-6 m/um) * DENSITY[kg/m^3] / H[m] * (1e9 ug/kg) = V_col * DENSITY * 1e3 / H
    density_kg_m3 = density_g_cm3 * 1000
    pm25_ug_m3 = (total_volume_col * density_kg_m3 * 1e3) / layer_height_m
    
    return pm25_ug_m3

# --- Configuration ---
waveInd = 0  # Wavelength index to analyze (e.g., 2 for 550 nm)
aodThresh = 0.0  # Set to 0 to disable AOD filtering
basePath = '/Users/wrespino/Synced/AOS/Pre-Phase-A/Polarimeter_Simulations/V3/'
# filePatterns = [
#     'V2megaharp1_pollutionVariable+smokeVariableLand_tFctrandLogNrm5.0_n*_nAng0.pkl',
#     'V2option2_pollutionVariable+smokeVariableLand_tFctrandLogNrm5.0_n*_nAng0.pkl',
#     'V2megaharp4_pollutionVariable+smokeVariableLand_tFctrandLogNrm5.0_n*_nAng0.pkl'
# ]
# filePatterns = [
#     'V3megaharp1_marineVariable+dustVariableOcean_tFctrandLogNrm0.*_n*_nAng0.pkl',
#     'V3megaharp1_pollutionVariable+dustVariableLand_tFctrandLogNrm0.*_n*_nAng0.pkl',
#     'V3option2_marineVariable+dustVariableOcean_tFctrandLogNrm0.*_n*_nAng0.pkl',
#     'V3option2_pollutionVariable+dustVariableLand_tFctrandLogNrm0.*_n*_nAng0.pkl',
#     'V3megaharp4_marineVariable+dustVariableOcean_tFctrandLogNrm0.*_n*_nAng0.pkl',
#     'V3megaharp4_pollutionVariable+dustVariableLand_tFctrandLogNrm0.*_n*_nAng0.pkl'
# ]
filePatterns = [
    'V3Amegaharp1_marineVariable+smokeVariableOcean_tFctrandLogNrm*.*_n*_nAng0.pkl',
    'V3Amegaharp1_pollutionVariable+smokeVariableLand_tFctrandLogNrm*.*_n*_nAng0.pkl',
    'V3Aoption2_marineVariable+smokeVariableOcean_tFctrandLogNrm*.*_n*_nAng0.pkl',
    'V3Aoption2_pollutionVariable+smokeVariableLand_tFctrandLogNrm*.*_n*_nAng0.pkl',
    'V3Amegaharp4_marineVariable+smokeVariableOcean_tFctrandLogNrm*.*_n*_nAng0.pkl',
    'V3Amegaharp4_pollutionVariable+smokeVariableLand_tFctrandLogNrm*.*_n*_nAng0.pkl'
]


instrument_pairs = ['megaharp1', 'option2', 'megaharp4']
colors = ['blue', 'red']  # Ocean, Land
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12 # Increase font size for better readability

recalcChi = True # recalculate chi2 for convergence filter
σx={'I'   :0.030, # relative
    'QoI' :0.005, # absolute
    'UoI' :0.005, # absolute
    'Q'   :0.005, # absolute in terms of Q/I
    'U'   :0.005, # absolute in terms of U/I
    }

# --- Data Loading and Analysis (Pass 1) ---
print("Loading and analyzing simulation data...")
processed_data = []
case_types = []  # Track if each pattern is 'land' or 'ocean'
instrument_case_map = []  # Track (instrument, case_type) for each pattern
for pattern in filePatterns:
    # Determine case type from pattern
    if 'Land' in pattern or 'land' in pattern:
        case_type = 'land'
    elif 'Ocean' in pattern or 'ocean' in pattern:
        case_type = 'ocean'
    else:
        case_type = 'unknown'
    case_types.append(case_type)
    # Determine instrument from pattern
    instrument = None
    for inst in instrument_pairs:
        if inst in pattern:
            instrument = inst
            break
    instrument_case_map.append((instrument, case_type))

    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    sim.conerganceFilter(χthresh=1.5, forceχ2Calc=recalcChi, σ=σx, verbose=True)
    # sim._addReffMode(modeCut=0.5, Force=True)

    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        processed_data.append(None)
        continue

    # Filter pixels by AOD threshold
    initial_pixel_count = len(sim.rsltFwd)
    if aodThresh > 0:
        inds2keep = [i for i, rslt in enumerate(sim.rsltFwd) if rslt['aod'][waveInd] >= aodThresh]
    else:
        inds2keep = list(range(len(sim.rsltFwd)))
    
    # Filter out very high AOD values (>10)
    high_aod_count = 0
    final_inds = []
    for i in inds2keep:
        if sim.rsltFwd[i]['aod'][waveInd] < 10.0:
            final_inds.append(i)
        else:
            high_aod_count += 1
    
    if high_aod_count > 0:
        print(f"  WARNING: Removed {high_aod_count} pixels with AOD >= 10 from {pattern.split('_')[0]}")
    
    sim.rsltFwd = [sim.rsltFwd[i] for i in final_inds]
    sim.rsltBck = [sim.rsltBck[i] for i in final_inds]
    
    if aodThresh > 0 and len(final_inds) < initial_pixel_count:
        print(f"  Filtered {initial_pixel_count - len(final_inds)} pixels from {pattern.split('_')[0]} (AOD < {aodThresh} or AOD >= 10). Kept {len(sim.rsltFwd)}.")

    if not sim.rsltFwd:
        print(f"Warning: No pixels left for {pattern} after AOD filtering.")
        processed_data.append(None)
        continue
        
    # Calculate PM2.5 MAD
    true_pm25 = np.array([calculate_pm25(rs) for rs in sim.rsltFwd])
    print(np.nanmean(true_pm25))
    retrieved_pm25 = np.array([calculate_pm25(rs) for rs in sim.rsltBck])
    mad_pm25 = np.nanmedian(np.abs(true_pm25 - retrieved_pm25))
    
    # Analyze other variables
    analysis_results = sim.analyzeSim(waveInd)[0]
    analysis_results['PM2.5'] = mad_pm25
    processed_data.append(analysis_results)

# --- Data Structuring (Pass 2) ---
# Build nested results: results[var][instrument][case_type] = value
results = {}
all_keys = set()
for data in processed_data:
    if data:
        all_keys.update(data.keys())

if 'rEffMode' in all_keys:
    all_keys.remove('rEffMode')
    all_keys.add('rEffMode (fine)')
    all_keys.add('rEffMode (coarse)')

for key in sorted(list(all_keys)):
    results[key] = {inst: {} for inst in instrument_pairs}

for idx, data in enumerate(processed_data):
    instrument, case_type = instrument_case_map[idx]
    for key in results:
        if data is None or instrument is None or case_type == 'unknown':
            continue
        if key == 'rEffMode (fine)':
            val = data.get('rEffMode', [np.nan, np.nan])[0]
            results[key][instrument][case_type] = val
        elif key == 'rEffMode (coarse)':
            val = data.get('rEffMode', [np.nan, np.nan])[1]
            results[key][instrument][case_type] = val
        else:
            val = data.get(key, np.nan)
            results[key][instrument][case_type] = np.mean(val) if key != 'PM2.5' else val

if not results:
    print("Error: No data was successfully processed. Halting.")
    exit()
    
# --- Plotting ---
print("Generating and saving figures...")

# Need to load one file to get a sample lambda
simBase = simulation(picklePath=os.path.join(basePath, filePatterns[0]))
bckLambda = simBase.rsltBck[0]['lambda'][waveInd]

for var, inst_dict in results.items():
    fig, ax = plt.subplots(figsize=(6, 5)) # Make plot narrower
    n_pairs = len(instrument_pairs)
    bar_width = 0.35
    index = np.arange(n_pairs)
    
    # Gather available data for each instrument
    ocean_vals = []
    land_vals = []
    bar_positions_ocean = []
    bar_positions_land = []
    for i, inst in enumerate(instrument_pairs):
        has_ocean = 'ocean' in inst_dict[inst] and not np.isnan(inst_dict[inst].get('ocean', np.nan))
        has_land = 'land' in inst_dict[inst] and not np.isnan(inst_dict[inst].get('land', np.nan))
        if var == 'PM2.5':
            # Only plot land bars for PM2.5
            if has_land:
                land_vals.append(inst_dict[inst]['land'])
                bar_positions_land.append(index[i])
        else:
            if has_ocean:
                ocean_vals.append(inst_dict[inst]['ocean'])
                bar_positions_ocean.append(index[i] - (bar_width/2 if has_land else 0))
            if has_land:
                land_vals.append(inst_dict[inst]['land'])
                bar_positions_land.append(index[i] + (bar_width/2 if has_ocean else 0))
    
    # Plot bars only for available data
    if var == 'PM2.5':
        if land_vals:
            ax.bar(bar_positions_land, land_vals, bar_width, label='Land', color=colors[1])
        ax.set_xticks(index)
        ax.set_xticklabels(instrument_pairs)
        # No legend for PM2.5
    else:
        if ocean_vals:
            ax.bar(bar_positions_ocean, ocean_vals, bar_width, label='Ocean', color=colors[0])
        if land_vals:
            ax.bar(bar_positions_land, land_vals, bar_width, label='Land', color=colors[1])
        ax.set_xticks(index)
        ax.set_xticklabels(instrument_pairs)
        # Only show legend if both present
        if ocean_vals and land_vals:
            ax.legend()
        elif ocean_vals:
            ax.legend(['Ocean'])
        elif land_vals:
            ax.legend(['Land'])
    
    ax.set_xlabel('Instrument')
    if 'rEff' in var:
        ax.set_ylabel('RMSE (μm)')
        ax.set_title(f'RMSE for {var}')
    elif var == 'PM2.5':
        ax.set_ylabel('MAD (μg/m³)')
        ax.set_title(f'Median Absolute Deviation for {var}')
    else:
        ax.set_ylabel('RMSE')
        ax.set_title(f'RMSE for {var} at {bckLambda:.3f} μm')
    
    fig.tight_layout()
    
    aod_str = f"_aod{aodThresh}" if aodThresh > 0 else ""
    # Determine output directory based on file patterns
    case_version = "V3A" if "V3A" in filePatterns[0] else "V3"
    output_dir = f"./{case_version}_results/"
    save_filename = f'{output_dir}RMSE_{var.replace(" ", "_")}_comparison_wv{waveInd}{aod_str}.png'
    plt.savefig(save_filename)
    print(f"Saved figure: {save_filename}")
    plt.close(fig)

# --- AOD Scatter Plots ---
print("Generating AOD scatter plots...")

# Load data again to get individual pixel values for scatter plots
scatter_data = {}  # scatter_data[instrument][case_type] = {'fwd_aod': [], 'diff_aod': []}

for idx, pattern in enumerate(filePatterns):
    instrument, case_type = instrument_case_map[idx]
    
    if instrument not in scatter_data:
        scatter_data[instrument] = {}
    if case_type not in scatter_data[instrument]:
        scatter_data[instrument][case_type] = {'fwd_aod': [], 'diff_aod': []}
    
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    sim.conerganceFilter(χthresh=1.5, forceχ2Calc=recalcChi, σ=σx, verbose=False)
    
    if not sim.rsltFwd:
        continue
    
    # Filter pixels by AOD threshold (same as before)
    initial_pixel_count = len(sim.rsltFwd)
    if aodThresh > 0:
        inds2keep = [i for i, rslt in enumerate(sim.rsltFwd) if rslt['aod'][waveInd] >= aodThresh]
    else:
        inds2keep = list(range(len(sim.rsltFwd)))
    
    # Filter out very high AOD values (>10)
    high_aod_count = 0
    final_inds = []
    for i in inds2keep:
        if sim.rsltFwd[i]['aod'][waveInd] < 10.0:
            final_inds.append(i)
        else:
            high_aod_count += 1
    
    sim.rsltFwd = [sim.rsltFwd[i] for i in final_inds]
    sim.rsltBck = [sim.rsltBck[i] for i in final_inds]
    
    if not sim.rsltFwd:
        continue
    
    # Extract AOD values for each pixel
    for i, (fwd_rslt, bck_rslt) in enumerate(zip(sim.rsltFwd, sim.rsltBck)):
        fwd_aod = fwd_rslt['aod'][waveInd]
        bck_aod = bck_rslt['aod'][waveInd]
        diff_aod = bck_aod - fwd_aod
        
        scatter_data[instrument][case_type]['fwd_aod'].append(fwd_aod)
        scatter_data[instrument][case_type]['diff_aod'].append(diff_aod)

# Create scatter plots (3 instruments x 2 surface types = 6 plots)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Find global limits for consistent axis scaling
max_fwd_aod = 0
all_fwd_aod = []
for inst in instrument_pairs:
    for case in ['ocean', 'land']:
        if (inst in scatter_data and 
            case in scatter_data[inst] and 
            len(scatter_data[inst][case]['fwd_aod']) > 0):
            fwd_vals = np.array(scatter_data[inst][case]['fwd_aod'])
            all_fwd_aod.extend(fwd_vals)
            max_fwd_aod = max(max_fwd_aod, np.max(fwd_vals))

# Set global axis limits
x_max = max_fwd_aod * 1.1
y_lim = max_fwd_aod * 0.25  # ±25% of highest AOD

for col, instrument in enumerate(instrument_pairs):
    for row, case_type in enumerate(['ocean', 'land']):
        ax = axes[row, col]
        
        # Check if data exists for this combination
        if (instrument in scatter_data and 
            case_type in scatter_data[instrument] and 
            len(scatter_data[instrument][case_type]['fwd_aod']) > 0):
            
            fwd_aod = np.array(scatter_data[instrument][case_type]['fwd_aod'])
            diff_aod = np.array(scatter_data[instrument][case_type]['diff_aod'])
            
            # Calculate point density for coloring
            if len(fwd_aod) > 1:
                try:
                    # Create a 2D density estimate
                    xy = np.vstack([fwd_aod, diff_aod])
                    kde = gaussian_kde(xy)
                    density = kde(xy)
                except:
                    # Fallback if KDE fails
                    density = np.ones(len(fwd_aod))
            else:
                density = np.ones(len(fwd_aod))
            
            # Create scatter plot with density-based coloring
            scatter = ax.scatter(fwd_aod, diff_aod, c=density, s=10, alpha=0.7, 
                               cmap='viridis', edgecolors='none')
            
            # Add 1-sigma error envelope (0.03 + 10% of true AOD)
            x_range = np.linspace(0, x_max, 100)
            error_envelope_upper = 0.03 + 0.1 * x_range
            error_envelope_lower = -(0.03 + 0.1 * x_range)
            
            ax.fill_between(x_range, error_envelope_lower, error_envelope_upper, 
                           color='lightgray', alpha=0.3, label='±1σ envelope')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Calculate statistics
            n_points = len(fwd_aod)
            
            # Calculate percentage within error envelope
            envelope_upper_at_points = 0.03 + 0.1 * fwd_aod
            envelope_lower_at_points = -(0.03 + 0.1 * fwd_aod)
            within_envelope = np.logical_and(diff_aod >= envelope_lower_at_points, 
                                           diff_aod <= envelope_upper_at_points)
            pct_within_envelope = np.sum(within_envelope) / len(diff_aod) * 100
            
            # Display statistics
            ax.text(0.05, 0.95, f'N: {n_points}\nWithin ±1σ: {pct_within_envelope:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # No data available
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
        
        # Set consistent axis limits for all subplots
        ax.set_xlim(0, x_max)
        ax.set_ylim(-y_lim, y_lim)
        
        # Set labels and titles
        if row == 1:  # Bottom row
            ax.set_xlabel('True AOD (Forward)')
        if col == 0:  # Left column
            ax.set_ylabel('AOD Difference (Retrieved - True)')
        
        # Title
        surface_name = case_type.capitalize()
        ax.set_title(f'{instrument} - {surface_name}')
        
        # Grid
        ax.grid(True, alpha=0.3)

# Add main title
fig.suptitle(f'AOD Retrieval Performance at {bckLambda:.3f} μm', fontsize=16, y=0.98)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save figure
aod_str = f"_aod{aodThresh}" if aodThresh > 0 else ""
case_version = "V3A" if "V3A" in filePatterns[0] else "V3"
output_dir = f"./{case_version}_results/"
scatter_filename = f'{output_dir}AOD_scatter_comparison_wv{waveInd}{aod_str}.png'
plt.savefig(scatter_filename, dpi=150, bbox_inches='tight')
print(f"Saved figure: {scatter_filename}")
plt.close(fig)

# --- rEff Scatter Plots ---
print("Generating rEff scatter plots...")

# Load data again to get individual pixel values for rEff scatter plots
reff_scatter_data = {}  # reff_scatter_data[instrument][case_type] = {'fwd_aod': [], 'diff_reff': []}

for idx, pattern in enumerate(filePatterns):
    instrument, case_type = instrument_case_map[idx]
    
    if instrument not in reff_scatter_data:
        reff_scatter_data[instrument] = {}
    if case_type not in reff_scatter_data[instrument]:
        reff_scatter_data[instrument][case_type] = {'fwd_aod': [], 'diff_reff': []}
    
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    sim.conerganceFilter(χthresh=1.5, forceχ2Calc=recalcChi, σ=σx, verbose=False)
    
    if not sim.rsltFwd:
        continue
    
    # Filter pixels by AOD threshold (same as before)
    initial_pixel_count = len(sim.rsltFwd)
    if aodThresh > 0:
        inds2keep = [i for i, rslt in enumerate(sim.rsltFwd) if rslt['aod'][waveInd] >= aodThresh]
    else:
        inds2keep = list(range(len(sim.rsltFwd)))
    
    # Filter out very high AOD values (>10)
    high_aod_count = 0
    final_inds = []
    for i in inds2keep:
        if sim.rsltFwd[i]['aod'][waveInd] < 10.0:
            final_inds.append(i)
        else:
            high_aod_count += 1
    
    sim.rsltFwd = [sim.rsltFwd[i] for i in final_inds]
    sim.rsltBck = [sim.rsltBck[i] for i in final_inds]
    
    if not sim.rsltFwd:
        continue
    
    # Extract AOD and rEff values for each pixel
    for i, (fwd_rslt, bck_rslt) in enumerate(zip(sim.rsltFwd, sim.rsltBck)):
        fwd_aod = fwd_rslt['aod'][waveInd]
        fwd_reff = fwd_rslt['rEff']  # rEff is a scalar, not wavelength-dependent
        bck_reff = bck_rslt['rEff']  # rEff is a scalar, not wavelength-dependent
        diff_reff = bck_reff - fwd_reff
        
        reff_scatter_data[instrument][case_type]['fwd_aod'].append(fwd_aod)
        reff_scatter_data[instrument][case_type]['diff_reff'].append(diff_reff)

# Create rEff scatter plots (3 instruments x 2 surface types = 6 plots)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Find global limits for consistent axis scaling
max_fwd_aod_reff = 0
all_diff_reff = []
for inst in instrument_pairs:
    for case in ['ocean', 'land']:
        if (inst in reff_scatter_data and 
            case in reff_scatter_data[inst] and 
            len(reff_scatter_data[inst][case]['fwd_aod']) > 0):
            fwd_vals = np.array(reff_scatter_data[inst][case]['fwd_aod'])
            diff_vals = np.array(reff_scatter_data[inst][case]['diff_reff'])
            all_diff_reff.extend(diff_vals)
            max_fwd_aod_reff = max(max_fwd_aod_reff, np.max(fwd_vals))

# Set global axis limits
x_max_reff = max_fwd_aod_reff * 1.1
if len(all_diff_reff) > 0:
    y_max_reff = max(abs(np.min(all_diff_reff)), abs(np.max(all_diff_reff))) * 1.1
else:
    y_max_reff = 1.0

for col, instrument in enumerate(instrument_pairs):
    for row, case_type in enumerate(['ocean', 'land']):
        ax = axes[row, col]
        
        # Check if data exists for this combination
        if (instrument in reff_scatter_data and 
            case_type in reff_scatter_data[instrument] and 
            len(reff_scatter_data[instrument][case_type]['fwd_aod']) > 0):
            
            fwd_aod = np.array(reff_scatter_data[instrument][case_type]['fwd_aod'])
            diff_reff = np.array(reff_scatter_data[instrument][case_type]['diff_reff'])
            
            # Calculate point density for coloring
            if len(fwd_aod) > 1:
                try:
                    # Create a 2D density estimate
                    xy = np.vstack([fwd_aod, diff_reff])
                    kde = gaussian_kde(xy)
                    density = kde(xy)
                except:
                    # Fallback if KDE fails
                    density = np.ones(len(fwd_aod))
            else:
                density = np.ones(len(fwd_aod))
            
            # Create scatter plot with density-based coloring
            scatter = ax.scatter(fwd_aod, diff_reff, c=density, s=10, alpha=0.7, 
                               cmap='viridis', edgecolors='none')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Calculate statistics
            n_points = len(fwd_aod)
            rmse = np.sqrt(np.mean(diff_reff**2))
            bias = np.mean(diff_reff)
            
            # Display statistics
            ax.text(0.05, 0.95, f'N: {n_points}\nRMSE: {rmse:.4f}\nBias: {bias:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # No data available
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
        
        # Set consistent axis limits for all subplots
        ax.set_xlim(0, x_max_reff)
        ax.set_ylim(-y_max_reff, y_max_reff)
        
        # Set labels and titles
        if row == 1:  # Bottom row
            ax.set_xlabel('True AOD (Forward)')
        if col == 0:  # Left column
            ax.set_ylabel('rEff Difference (Retrieved - True) [μm]')
        
        # Title
        surface_name = case_type.capitalize()
        ax.set_title(f'{instrument} - {surface_name}')
        
        # Grid
        ax.grid(True, alpha=0.3)

# Add main title
fig.suptitle(f'rEff Retrieval Performance at {bckLambda:.3f} μm', fontsize=16, y=0.98)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save figure
aod_str = f"_aod{aodThresh}" if aodThresh > 0 else ""
case_version = "V3A" if "V3A" in filePatterns[0] else "V3"
output_dir = f"./{case_version}_results/"
reff_scatter_filename = f'{output_dir}rEff_scatter_comparison_wv{waveInd}{aod_str}.png'
plt.savefig(reff_scatter_filename, dpi=150, bbox_inches='tight')
print(f"Saved figure: {reff_scatter_filename}")
plt.close(fig)

print("\nComparison script finished.")

