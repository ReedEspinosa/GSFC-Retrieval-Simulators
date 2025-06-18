import numpy as np
import matplotlib.pyplot as plt
from simulateRetrieval import simulation
import os
import glob

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
waveInd = 3  # Wavelength index to analyze (e.g., 2 for 550 nm)
aodThresh = 0.0  # Set to 0 to disable AOD filtering
basePath = '/Users/wrespino/Synced/AOS/Pre-Phase-A/Polarimeter_Simulations/V1/'
filePatterns = [
    'V1megaharp1_marineVariable+dustVariableOcean_tFctrandLogNrm0.*_n*_nAng0.pkl',
    'V1megaharp1_pollutionVariable+dustVariableLand_tFctrandLogNrm0.*_n*_nAng0.pkl',
    'V1option2_marineVariable+dustVariableOcean_tFctrandLogNrm0.*_n*_nAng0.pkl',
    'V1option2_pollutionVariable+dustVariableLand_tFctrandLogNrm0.*_n*_nAng0.pkl',
    'V1megaharp4_marineVariable+dustVariableOcean_tFctrandLogNrm0.*_n*_nAng0.pkl',
    'V1megaharp4_pollutionVariable+dustVariableLand_tFctrandLogNrm0.*_n*_nAng0.pkl'
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
for pattern in filePatterns:
    full_path_pattern = os.path.join(basePath, pattern)
    sim = simulation(picklePath=full_path_pattern)
    sim.conerganceFilter(χthresh=1.5, forceχ2Calc=recalcChi, σ=σx, verbose=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
    # sim._addReffMode(modeCut=0.5, Force=True)

    if not sim.rsltFwd:
        print(f"Warning: No data loaded for pattern: {pattern}")
        processed_data.append(None)
        continue

    # Filter pixels by AOD threshold
    if aodThresh > 0:
        initial_pixel_count = len(sim.rsltFwd)
        inds2keep = [i for i, rslt in enumerate(sim.rsltFwd) if rslt['aod'][waveInd] >= aodThresh]
        sim.rsltFwd = [sim.rsltFwd[i] for i in inds2keep]
        sim.rsltBck = [sim.rsltBck[i] for i in inds2keep]
        print(f"  Filtered {initial_pixel_count - len(sim.rsltFwd)} pixels from {pattern.split('_')[0]} (AOD < {aodThresh}). Kept {len(sim.rsltFwd)}.")

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
    results[key] = []

for data in processed_data:
    for key in results:
        if data is None:
            results[key].append(np.nan)
            continue
        
        if key == 'rEffMode (fine)':
            val = data.get('rEffMode', [np.nan, np.nan])[0]
            results[key].append(val)
        elif key == 'rEffMode (coarse)':
            val = data.get('rEffMode', [np.nan, np.nan])[1]
            results[key].append(val)
        else:
            val = data.get(key, np.nan)
            results[key].append(np.mean(val) if not key=='PM2.5' else val)

if not results:
    print("Error: No data was successfully processed. Halting.")
    exit()
    
# --- Plotting ---
print("Generating and saving figures...")

# Need to load one file to get a sample lambda
simBase = simulation(picklePath=os.path.join(basePath, filePatterns[0]))
bckLambda = simBase.rsltBck[0]['lambda'][waveInd]

for var, values in results.items():
    fig, ax = plt.subplots(figsize=(6, 5)) # Make plot narrower
    
    n_pairs = len(instrument_pairs)
    bar_width = 0.35
    index = np.arange(n_pairs)
    
    if var == 'PM2.5':
        # Only plot land bars for PM2.5
        land_rmses = [values[i+1] for i in range(0, len(values), 2)]
        ax.bar(index, land_rmses, bar_width, label='Land', color=colors[1])
        ax.set_xticks(index)
        ax.set_xticklabels(instrument_pairs)
        # Remove the legend for PM2.5
        # ax.legend().remove()
    else:
        ocean_rmses = [values[i] for i in range(0, len(values), 2)]
        land_rmses = [values[i+1] for i in range(0, len(values), 2)]
        ax.bar(index - bar_width/2, ocean_rmses, bar_width, label='Ocean', color=colors[0])
        ax.bar(index + bar_width/2, land_rmses, bar_width, label='Land', color=colors[1])
        ax.set_xticks(index)
        ax.set_xticklabels(instrument_pairs)
        ax.legend()
    
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
    save_filename = f'./RMSE_{var.replace(" ", "_")}_comparison_wv{waveInd}{aod_str}.png'
    plt.savefig(save_filename)
    print(f"Saved figure: {save_filename}")
    plt.close(fig)

print("\nComparison script finished.")

