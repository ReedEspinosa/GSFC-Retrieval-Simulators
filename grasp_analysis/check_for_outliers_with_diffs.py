import os
import re
import tarfile
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

def extract_datetime_from_filename(filename):
    """
    Extracts date and time from the filename (e.g., matching '20060301_2300z').
    """
    # Regex to find YYYYMMDD_HHMM(z or Z)
    match = re.search(r'(\d{8})_(\d{4}[zZ])', filename)
    if match:
        date_str, time_str = match.groups()
        return date_str, time_str
    return "UnknownDate", "UnknownTime"

def find_corresponding_zip(filepath, date_str, time_str):
    """
    Looks for a zip file in the same directory matching the date and time.
    """
    # Search the same directory for a .zip containing the date and time strings
    potential_zips = list(filepath.parent.glob(f"*{date_str}_{time_str}*.zip"))
    
    # Fallback to just .zip files if exact date/time matching fails, though this is riskier
    if not potential_zips:
        potential_zips = list(filepath.parent.glob("*.zip"))
        
    return potential_zips[0] if potential_zips else None

def extract_aod_data(matched_pairs, aod_var_name="aod"):
    """
    Extracts AOD data, grouped by month for plotting.
    Identifies high AOD > 4 and absolute differences > 0.5.
    """
    monthly_data = {}
    high_aod_records = []
    diff_records = []
    zips_to_tar = set()
    
    for pair in matched_pairs:
        month = pair['month']
        ret_filepath = pair['retrieved']
        ret_filename = ret_filepath.name
        
        try:
            with xr.open_dataset(ret_filepath) as ds_ret, \
                 xr.open_dataset(pair['truth']) as ds_tru:
                
                if aod_var_name not in ds_ret or aod_var_name not in ds_tru:
                    continue

                ret_aod_da = ds_ret[aod_var_name]
                tru_aod_da = ds_tru[aod_var_name]

                ret_aod_flat = ret_aod_da.values.flatten()
                tru_aod_flat = tru_aod_da.values.flatten()
                
                if ret_aod_flat.shape != tru_aod_flat.shape:
                    continue
                
                # Safely extract and broadcast latitude/longitude
                try:
                    lat_da = ds_ret['latitude']
                    lon_da = ds_ret['longitude']
                    _, lat_b, lon_b = xr.broadcast(ret_aod_da, lat_da, lon_da)
                    lat_flat = lat_b.values.flatten()
                    lon_flat = lon_b.values.flatten()
                except KeyError:
                    lat_flat = np.full(ret_aod_flat.shape, np.nan)
                    lon_flat = np.full(ret_aod_flat.shape, np.nan)

                # Valid points mask (no NaNs)
                valid_mask = ~np.isnan(ret_aod_flat) & ~np.isnan(tru_aod_flat)
                
                # Metrics
                abs_diff = np.abs(ret_aod_flat - tru_aod_flat)
                # Avoid division by zero for relative difference by masking
                safe_tru = np.where(tru_aod_flat == 0, np.nan, tru_aod_flat)
                rel_diff = (ret_aod_flat - tru_aod_flat) / safe_tru
                
                # --- Requirement: High AOD > 4 ---
                outlier_mask = valid_mask & (ret_aod_flat > 4.0)
                for idx in np.where(outlier_mask)[0]:
                    high_aod_records.append({
                        'filename': ret_filename,
                        'pixel_number': idx,
                        'retrieved_aod': ret_aod_flat[idx],
                        'truth_aod': tru_aod_flat[idx],
                        'latitude': lat_flat[idx],
                        'longitude': lon_flat[idx]
                    })
                
                # --- Requirement: Absolute Difference > 0.5 ---
                diff_mask = valid_mask & (abs_diff > 0.5)
                bad_diff_indices = np.where(diff_mask)[0]
                
                if len(bad_diff_indices) > 0:
                    date_str, time_str = extract_datetime_from_filename(ret_filename)
                    zip_file = find_corresponding_zip(ret_filepath, date_str, time_str)
                    zip_name = zip_file.name if zip_file else "None Found"
                    
                    if zip_file:
                        zips_to_tar.add(zip_file)
                    
                    for idx in bad_diff_indices:
                        diff_records.append({
                            'Date': date_str,
                            'Time': time_str,
                            'Retrieved_AOD': ret_aod_flat[idx],
                            'Truth_AOD': tru_aod_flat[idx],
                            'Relative_Difference': rel_diff[idx],
                            'Absolute_Difference': abs_diff[idx],
                            'Zip_File': zip_name
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
        
    return monthly_data, high_aod_records, diff_records, zips_to_tar

def save_differences_and_tar(diff_records, zips_to_tar, csv_out="aod_diff_outliers.csv", tar_out="packaged_outliers.tar"):
    """
    Saves the absolute difference > 0.5 records to CSV and packages the corresponding zips.
    """
    if diff_records:
        df = pd.DataFrame(diff_records)
        df.to_csv(csv_out, index=False)
        print(f"Saved {len(diff_records)} absolute difference outliers to {csv_out}")
    else:
        print("No absolute difference > 0.5 found.")
        
    if zips_to_tar:
        print(f"Packaging {len(zips_to_tar)} corresponding zip files into {tar_out}...")
        with tarfile.open(tar_out, "w") as tar:
            for zip_path in zips_to_tar:
                tar.add(zip_path, arcname=zip_path.name)
        print(f"Successfully created {tar_out}.")
    else:
        print("No associated zip files found to package.")

def save_high_aod_to_csv(high_aod_records, out_filename="high_aod_outliers.csv"):
    if not high_aod_records:
        return []
    df = pd.DataFrame(high_aod_records)
    df.to_csv(out_filename, index=False)
    print(f"Saved {len(high_aod_records)} high AOD outliers to {out_filename}")
    return df

def plot_high_aod_map(df_records, out_filename="high_aod_map.png"):
    if df_records is None or len(df_records) == 0:
        return
    df_valid = df_records.dropna(subset=['latitude', 'longitude'])
    if len(df_valid) == 0:
        return
    plt.figure(figsize=(12, 6))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        scatter = ax.scatter(df_valid['longitude'], df_valid['latitude'], 
                             c=df_valid['retrieved_aod'], cmap='Reds', s=30,
                             transform=ccrs.PlateCarree(), zorder=5)
    else:
        scatter = plt.scatter(df_valid['longitude'], df_valid['latitude'], 
                              c=df_valid['retrieved_aod'], cmap='Reds', s=30)
    plt.colorbar(scatter, label='Retrieved AOD')
    plt.title('Locations of High Retrieved AOD (> 4.0)')
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
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
        ax.legend()
    for j in range(len(months), 4):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
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
    plt.title("All Months: Retrieved vs Truth AOD")
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    plt.close()

def plot_aod_difference(monthly_data, out_filename="aod_difference_vs_truth.png"):
    all_truth, all_retrieved = [], []
    for month in monthly_data:
        all_truth.extend(monthly_data[month]['truth'])
        all_retrieved.extend(monthly_data[month]['retrieved'])
    if not all_truth: return
    all_truth = np.array(all_truth)
    difference = np.array(all_retrieved) - all_truth
    plt.figure(figsize=(9, 6))
    plt.scatter(all_truth, difference, alpha=0.3, s=5, c='teal')
    plt.axhline(0, color='k', linestyle='--', linewidth=1.5)
    plt.title("AOD Difference (Retrieved - Truth) vs Truth AOD")
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    BASE_DIR = "/home/dgiles/nobackup/AIST/software/OSSE_Test_Run"
    AOD_VARIABLE_NAME = "aod" 
    
    print(f"Scanning directory: {BASE_DIR}")
    matched_pairs = get_matched_files(BASE_DIR)
    
    if matched_pairs:
        print("Extracting data... (This may take a moment)")
        monthly_data, high_aod_records, diff_records, zips_to_tar = extract_aod_data(matched_pairs, AOD_VARIABLE_NAME)
        
        # Save Outliers & Package Zips
        df_outliers = save_high_aod_to_csv(high_aod_records, "high_aod_outliers.csv")
        save_differences_and_tar(diff_records, zips_to_tar, "aod_diff_outliers.csv", "packaged_outliers.tar")
        
        # Generate Plots
        print("Generating plots...")
        if df_outliers is not None and len(df_outliers) > 0:
            plot_high_aod_map(df_outliers, "high_aod_map.png")
            
        plot_monthly_4panel(monthly_data)
        plot_all_months(monthly_data)
        plot_aod_difference(monthly_data)
        
        print("Analysis complete.")
    else:
        print("No matching file pairs were found.")
