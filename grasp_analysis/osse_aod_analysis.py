import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# Try importing cartopy for mapping, with a fallback if not installed
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy is not installed. Map will be plotted as a simple scatter plot without coastlines.")

def get_matched_files(base_dir):
    """
    Scans the base directory for YYYY/MM/DD subdirectories and pairs 
    RETRIEVED and TRUTH files for each hour based on the exact naming convention.
    """
    base_path = Path(base_dir)
    retrieved_files = list(base_path.rglob("*RETRIEVED.nc4"))
    
    matched_pairs = []
    for ret_file in retrieved_files:
        truth_filename = ret_file.name.replace("RETRIEVED", "TRUTH")
        truth_file = ret_file.parent / truth_filename
        
        if truth_file.exists():
            parts = ret_file.relative_to(base_path).parts
            if len(parts) >= 3:
                month = parts[1]
                matched_pairs.append({
                    'month': month,
                    'retrieved': ret_file,
                    'truth': truth_file
                })
            
    return matched_pairs

def extract_aod_data(matched_pairs, aod_var_name="aod"):
    """
    Extracts AOD data, grouped by month for plotting.
    Identifies points where Retrieved AOD > 4 and extracts metadata for CSV.
    """
    monthly_data = {}
    high_aod_records = []
    
    for pair in matched_pairs:
        month = pair['month']
        ret_filename = pair['retrieved'].name
        
        try:
            with xr.open_dataset(pair['retrieved']) as ds_ret, \
                 xr.open_dataset(pair['truth']) as ds_tru:
                
                if aod_var_name not in ds_ret or aod_var_name not in ds_tru:
                    print(f"Warning: '{aod_var_name}' not found in {ret_filename}. Skipping.")
                    continue

                # Get DataArrays
                ret_aod_da = ds_ret[aod_var_name]
                tru_aod_da = ds_tru[aod_var_name]

                # Flatten AOD arrays
                ret_aod_flat = ret_aod_da.values.flatten()
                tru_aod_flat = tru_aod_da.values.flatten()
                
                if ret_aod_flat.shape != tru_aod_flat.shape:
                    print(f"Shape mismatch in {ret_filename}. Skipping.")
                    continue
                
                # Safely extract and broadcast latitude/longitude
                try:
                    lat_da = ds_ret['latitude']
                    lon_da = ds_ret['longitude']
                    
                    # Broadcast coordinates to match the exact shape of the AOD array
                    # This safely duplicates lat/lon across extra dimensions (like wavelength)
                    _, lat_b, lon_b = xr.broadcast(ret_aod_da, lat_da, lon_da)
                    
                    lat_flat = lat_b.values.flatten()
                    lon_flat = lon_b.values.flatten()
                except KeyError:
                    print(f"Warning: 'latitude' or 'longitude' missing in {ret_filename}. Filling with NaN.")
                    lat_flat = np.full(ret_aod_flat.shape, np.nan)
                    lon_flat = np.full(ret_aod_flat.shape, np.nan)

                # Valid points mask (no NaNs)
                valid_mask = ~np.isnan(ret_aod_flat) & ~np.isnan(tru_aod_flat)
                
                # Find outliers (Retrieved AOD > 4)
                outlier_mask = valid_mask & (ret_aod_flat > 4.0)
                outlier_indices = np.where(outlier_mask)[0]
                
                for idx in outlier_indices:
                    high_aod_records.append({
                        'filename': ret_filename,
                        'pixel_number': idx,  # Flattened index
                        'retrieved_aod': ret_aod_flat[idx],
                        'truth_aod': tru_aod_flat[idx],
                        'latitude': lat_flat[idx],
                        'longitude': lon_flat[idx]
                    })
                
                # Keep valid data for plotting
                ret_aod_valid = ret_aod_flat[valid_mask]
                tru_aod_valid = tru_aod_flat[valid_mask]
                
                if month not in monthly_data:
                    monthly_data[month] = {'truth': [], 'retrieved': []}
                    
                monthly_data[month]['truth'].extend(tru_aod_valid)
                monthly_data[month]['retrieved'].extend(ret_aod_valid)
                
        except Exception as e:
            print(f"Error reading files for month {month} ({ret_filename}): {e}")
            
    for month in monthly_data:
        monthly_data[month]['truth'] = np.array(monthly_data[month]['truth'])
        monthly_data[month]['retrieved'] = np.array(monthly_data[month]['retrieved'])
        
    return monthly_data, high_aod_records

def save_high_aod_to_csv(high_aod_records, out_filename="high_aod_outliers.csv"):
    if not high_aod_records:
        print("No retrieved AOD values > 4 found. Skipping CSV creation.")
        return []
        
    df = pd.DataFrame(high_aod_records)
    df.to_csv(out_filename, index=False)
    print(f"Saved {len(high_aod_records)} outliers to {out_filename}")
    return df

def plot_high_aod_map(df_records, out_filename="high_aod_map.png"):
    """
    Plots the geographical locations of the high AOD pixels on a map.
    """
    if df_records is None or len(df_records) == 0:
        print("No high AOD records to map.")
        return
        
    # Drop rows where lat/lon might be NaN just to be safe
    df_valid = df_records.dropna(subset=['latitude', 'longitude'])
    
    if len(df_valid) == 0:
        print("No valid latitude/longitude coordinates found for mapping.")
        return

    plt.figure(figsize=(12, 6))

    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        
        scatter = ax.scatter(df_valid['longitude'], df_valid['latitude'], 
                             c=df_valid['retrieved_aod'], cmap='Reds', 
                             s=30, edgecolor='k', linewidth=0.5,
                             transform=ccrs.PlateCarree(),
                             zorder=5)
    else:
        # Fallback to standard scatter plot if cartopy isn't installed
        plt.grid(True, linestyle='--', alpha=0.5)
        scatter = plt.scatter(df_valid['longitude'], df_valid['latitude'], 
                              c=df_valid['retrieved_aod'], cmap='Reds', 
                              s=30, edgecolor='k', linewidth=0.5)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    cbar = plt.colorbar(scatter, label='Retrieved AOD')
    plt.title(f'Locations of High Retrieved AOD (> 4.0)\nTotal points: {len(df_valid)}')
    
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    print(f"Saved map of high AOD locations to {out_filename}")
    plt.close()

def plot_monthly_4panel(monthly_data, out_filename="aod_monthly_4panel.png"):
    months = sorted(list(monthly_data.keys()))[:4]
    if not months: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, month in enumerate(months):
        ax = axes[i]
        x = monthly_data[month]['truth']
        y = monthly_data[month]['retrieved']
        ax.scatter(x, y, alpha=0.3, s=5, c='blue')
        max_val = max(np.max(x) if len(x) > 0 else 1, np.max(y) if len(y) > 0 else 1)
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='1:1 Line')
        ax.set_title(f"Month: {month} (N={len(x)})")
        ax.set_xlabel("Truth AOD")
        ax.set_ylabel("Retrieved AOD")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    for j in range(len(months), 4):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    print(f"Saved monthly 4-panel plot to {out_filename}")
    plt.close()

def plot_all_months(monthly_data, out_filename="aod_all_months.png"):
    all_truth, all_retrieved = [], []
    for month in monthly_data:
        all_truth.extend(monthly_data[month]['truth'])
        all_retrieved.extend(monthly_data[month]['retrieved'])
    if not all_truth: return
    all_truth = np.array(all_truth)
    all_retrieved = np.array(all_retrieved)
    plt.figure(figsize=(8, 7))
    plt.scatter(all_truth, all_retrieved, alpha=0.3, s=5, c='purple')
    max_val = max(np.max(all_truth), np.max(all_retrieved))
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='1:1 Line')
    plt.title(f"All Months: Retrieved vs Truth AOD (N={len(all_truth)})")
    plt.xlabel("Truth AOD")
    plt.ylabel("Retrieved AOD")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    print(f"Saved all-months plot to {out_filename}")
    plt.close()

if __name__ == "__main__":
    BASE_DIR = "/home/dgiles/nobackup/AIST/software/OSSE_Test_Run"
    AOD_VARIABLE_NAME = "aod" 
    
    print(f"Scanning directory: {BASE_DIR}")
    matched_pairs = get_matched_files(BASE_DIR)
    print(f"Found {len(matched_pairs)} RETRIEVED/TRUTH file pairs.")
    
    if matched_pairs:
        print("Extracting data... (This may take a moment)")
        monthly_data, high_aod_records = extract_aod_data(matched_pairs, aod_var_name=AOD_VARIABLE_NAME)
        
        # Save Outliers to CSV and store DataFrame
        df_outliers = save_high_aod_to_csv(high_aod_records, out_filename="high_aod_outliers.csv")
        
        # Generate Plots
        print("Generating plots...")
        if df_outliers is not None and len(df_outliers) > 0:
            plot_high_aod_map(df_outliers, out_filename="high_aod_map.png")
            
        plot_monthly_4panel(monthly_data)
        plot_all_months(monthly_data)
        print("Analysis complete.")
    else:
        print("No matching file pairs were found. Please verify the directory structure.")
