#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot total lidar depolarization ratio for aerosols (unified version)

This script reads aerosol optical properties from netCDF files and calculates 
the total (aerosol + Rayleigh) lidar depolarization ratio at multiple wavelengths 
for polydisperse aerosols with lognormal size distributions.

Supports both hexahedral and spheroidal particle types automatically.

Author: Created for GSFC Retrieval Simulators
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys
import argparse

# Set matplotlib parameters for higher quality rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['image.interpolation'] = 'bicubic'
plt.rcParams['image.cmap'] = 'viridis'

# Add GSFC-GRASP-Python-Interface to path for accessing utilities
sys.path.append(os.path.join("..", "GSFC-GRASP-Python-Interface"))
try:
    from MADCAP_functions import loadVARSnetCDF
except ImportError:
    # Fallback if MADCAP_functions not available
    print("Warning: MADCAP_functions not available, using basic netCDF reading")

def detect_particle_type(file_path, data_shape):
    """
    Auto-detect particle type from file path and data characteristics
    
    Returns:
    - particle_type: 'hexahedral' or 'spheroidal'
    - ratio_index: 0 for hexahedral, 1 for spheroidal
    - needs_interpolation: True if spheroidal data needs mr interpolation
    """
    # Check file path first
    if 'hexahedral' in file_path.lower() or 'saito' in file_path.lower():
        return 'hexahedral', 0, False
    elif 'spheroidal' in file_path.lower() or 'grasp-v' in file_path.lower():
        return 'spheroidal', 1, True
    
    # Check data characteristics
    ratio_dim, mr_dim, mi_dim, x_dim = data_shape[:4]
    
    # Spheroidal typically has 2 ratios, many mr values, fewer x values
    # Hexahedral typically has 1 ratio, fewer mr values, many x values
    if ratio_dim == 2 and mr_dim > 15:
        return 'spheroidal', 1, True
    elif ratio_dim == 1 and x_dim > 200:
        return 'hexahedral', 0, False
    else:
        # Default assumption with warning
        print(f"Warning: Could not auto-detect particle type from path '{file_path}' and shape {data_shape}")
        print("Assuming hexahedral particles. Use --particle-type to override.")
        return 'hexahedral', 0, False

def read_netcdf_data(file_path):
    """Read aerosol optical properties from netCDF file"""
    print(f"Reading data from: {file_path}")
    
    with Dataset(file_path, 'r') as nc:
        # Read key variables
        data = {}
        data['mr'] = nc.variables['mr'][:]  # Real refractive index
        data['mi'] = nc.variables['mi'][:]  # Imaginary refractive index  
        data['x'] = nc.variables['x'][:]    # Size parameter (2π*r/λ)
        data['angle'] = nc.variables['angle'][:]  # Scattering angles
        data['scama'] = nc.variables['scama'][:]  # Scattering matrix elements
        
        # Convert masked arrays to regular arrays if needed
        for key in data:
            if hasattr(data[key], 'mask'):
                data[key] = np.ma.filled(data[key], fill_value=0.0)
        
        # Reference wavelength from variable description (340 nm)
        data['lambda_ref'] = 0.340  # μm
        
        print(f"Data dimensions:")
        print(f"  Real RI (mr): {len(data['mr'])} values")
        print(f"  Imaginary RI (mi): {len(data['mi'])} values") 
        print(f"  Size parameters (x): {len(data['x'])} values")
        print(f"  Scattering angles: {len(data['angle'])} values")
        print(f"  Scattering matrix shape: {data['scama'].shape}")
        print(f"  Reference wavelength: {data['lambda_ref']} μm")
        
    return data

def calculate_rayleigh_properties(altitude_km, wavelength_um):
    """
    Calculate Rayleigh scattering properties at given altitude and wavelength
    """
    # Standard atmosphere parameters
    pressure_sea_level = 1013.25  # hPa
    scale_height = 8.0  # km
    
    # Pressure at altitude using barometric formula
    pressure = pressure_sea_level * np.exp(-altitude_km / scale_height)
    pressure_ratio = pressure / pressure_sea_level
    
    # Rayleigh extinction coefficient (scaled from empirical sea level value)
    # At sea level, 532 nm: ~11.6 Mm^-1
    rayleigh_ext_sea_level_532nm = 11.6  # Mm^-1
    
    # Scale for pressure and wavelength (λ^-4 dependence)
    rayleigh_ext = (rayleigh_ext_sea_level_532nm * pressure_ratio * 
                   (0.532 / wavelength_um)**4)
    
    # Rayleigh depolarization ratio (wavelength independent)
    rayleigh_depol = 0.0295
    
    return rayleigh_ext, rayleigh_depol

def interpolate_to_reference_mr(data, reference_mr, ratio_index):
    """
    Interpolate data to reference mr values (for spheroidal data)
    """
    from scipy.interpolate import interp1d
    
    print(f"Interpolating scattering matrix from mr shape {data['scama'].shape} to {len(reference_mr)} mr values...")
    
    original_mr = data['mr']
    original_scama = data['scama']
    
    # Create new scattering matrix with reference mr dimensions
    new_scama = np.zeros((original_scama.shape[0], len(reference_mr), 
                         original_scama.shape[2], original_scama.shape[3],
                         original_scama.shape[4], original_scama.shape[5]))
    
    # Interpolate for each combination of other dimensions
    for mi_idx in range(original_scama.shape[2]):
        for x_idx in range(original_scama.shape[3]):
            for element_idx in range(original_scama.shape[4]):
                for angle_idx in range(original_scama.shape[5]):
                    # Extract data for this combination
                    values = original_scama[ratio_index, :, mi_idx, x_idx, element_idx, angle_idx]
                    
                    # Handle masked arrays
                    if hasattr(values, 'mask'):
                        values = np.ma.filled(values, fill_value=0.0)
                    
                    # Convert mr array if it's masked too
                    mr_for_interp = original_mr
                    if hasattr(mr_for_interp, 'mask'):
                        mr_for_interp = np.ma.filled(mr_for_interp, fill_value=np.nan)
                    
                    # Create interpolation function
                    interp_func = interp1d(mr_for_interp, values, bounds_error=False, fill_value='extrapolate')
                    
                    # Interpolate to reference mr values
                    new_scama[ratio_index, :, mi_idx, x_idx, element_idx, angle_idx] = interp_func(reference_mr)
    
    # Update data dictionary
    interpolated_data = data.copy()
    interpolated_data['mr'] = reference_mr
    interpolated_data['scama'] = new_scama
    
    print(f"Interpolation complete. New scama shape: {new_scama.shape}")
    return interpolated_data

def lognormal_volume_distribution(r, r_g, sigma_g):
    """
    Calculate lognormal volume size distribution
    
    Parameters:
    - r: radius array (μm)
    - r_g: geometric mean radius (μm) 
    - sigma_g: geometric standard deviation
    
    Returns:
    - dV/dlnr: volume distribution (normalized to integrate to 1)
    """
    ln_r = np.log(r)
    ln_rg = np.log(r_g)
    ln_sigma = np.log(sigma_g)
    
    # Volume distribution: dV/dlnr ∝ r³ * n(r)
    # For lognormal: n(r) ∝ (1/r) * exp(-(ln(r) - ln(r_g))²/(2*ln²(σ_g)))
    # So: dV/dlnr ∝ r² * exp(-(ln(r) - ln(r_g))²/(2*ln²(σ_g)))
    
    dV_dlnr = r**2 * np.exp(-(ln_r - ln_rg)**2 / (2 * ln_sigma**2))
    
    # Normalize so integral over ln(r) equals 1
    dV_dlnr = dV_dlnr / np.trapz(dV_dlnr, ln_r)
    
    return dV_dlnr

def calculate_polydisperse_depolarization(r_eff, mr_idx, mi_idx, data, target_wavelength, 
                                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index):
    """
    Calculate volume-averaged depolarization for lognormal size distribution
    
    Parameters:
    - r_eff: effective radius (μm)
    - mr_idx, mi_idx: indices for refractive index
    - data: netCDF data dictionary
    - target_wavelength: wavelength (μm)
    - aerosol_ext, rayleigh_ext, rayleigh_depol: extinction and depolarization parameters
    - ratio_index: 0 for hexahedral, 1 for spheroidal
    
    Returns:
    - total_depol: volume-averaged total depolarization ratio
    """
    # Lognormal distribution parameters
    ln_sigma = 0.6  # Given constraint
    sigma_g = np.exp(ln_sigma)  # Geometric standard deviation
    
    # Convert effective radius to geometric mean radius
    # r_eff = r_g * exp(2.5 * ln²(σ_g))
    r_g = r_eff / np.exp(2.5 * ln_sigma**2)
    
    # Set up radius grid for integration (25 points for speed)
    r_min = r_g / (sigma_g**3)  # 3 standard deviations below
    r_max = r_g * (sigma_g**3)  # 3 standard deviations above
    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), 25)
    
    # Calculate volume distribution
    dV_dlnr = lognormal_volume_distribution(r_grid, r_g, sigma_g)
    
    # Convert radii to size parameters
    x_grid = 2 * np.pi * r_grid / target_wavelength
    
    # Extract scattering matrix data at 180° (backscatter)
    x_data = data['x']
    angle_180_idx = -1  # Last angle should be 180°
    p11_180 = data['scama'][ratio_index, mr_idx, mi_idx, :, 0, angle_180_idx]  # P11 at 180°
    p22_180 = data['scama'][ratio_index, mr_idx, mi_idx, :, 1, angle_180_idx]  # P22 at 180°
    
    # Vectorized interpolation using scipy for speed
    from scipy.interpolate import interp1d
    
    # Create interpolation functions
    p11_interp = interp1d(x_data, p11_180, bounds_error=False, fill_value='extrapolate')
    p22_interp = interp1d(x_data, p22_180, bounds_error=False, fill_value='extrapolate')
    
    # Calculate P11 and P22 for all size parameters at once
    p11_grid = p11_interp(x_grid)
    p22_grid = p22_interp(x_grid)
    
    # Vectorized aerosol depolarization calculation
    aerosol_depol_grid = (p11_grid - p22_grid) / (p11_grid + p22_grid)
    
    # Calculate total depolarization for each size (vectorized)
    total_ext = aerosol_ext + rayleigh_ext
    total_depol_grid = ((aerosol_depol_grid * aerosol_ext + rayleigh_depol * rayleigh_ext) / total_ext)
    
    # Volume-weighted average
    ln_r_grid = np.log(r_grid)
    polydisperse_depol = np.trapz(total_depol_grid * dV_dlnr, ln_r_grid)
    
    return polydisperse_depol

def get_reference_mr_values():
    """Get reference mr values from hexahedral data for consistent comparison"""
    hexahedral_file = "/Users/wrespino/Synced/STG_AerosolModelExchange/GRASP-LUT-Export/GRASP-Kernels_netCDF-Versions/kernel-Saito-Hexahedra_psi0.7_1degAngRes_V4.nc"
    
    try:
        with Dataset(hexahedral_file, 'r') as nc:
            reference_mr = nc.variables['mr'][:]
            # Convert masked array to regular array if needed
            if hasattr(reference_mr, 'mask'):
                reference_mr = np.ma.filled(reference_mr, fill_value=np.nan)
            else:
                reference_mr = np.array(reference_mr)
        return reference_mr
    except:
        # Default mr values if hexahedral file not available
        print("Warning: Could not read reference mr values, using defaults")
        return np.array([1.3701, 1.413433, 1.456767, 1.5001, 1.543433, 1.586767, 1.6301, 1.673433])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot lidar depolarization ratios for aerosols')
    parser.add_argument('file_path', help='Path to netCDF file containing aerosol optical properties')
    parser.add_argument('--particle-type', choices=['hexahedral', 'spheroidal'], 
                       help='Force particle type (auto-detected if not specified)')
    parser.add_argument('--aerosol-ext', type=float, default=640, 
                       help='Aerosol extinction coefficient (Mm^-1, default: 640)')
    parser.add_argument('--altitude', type=float, default=2.2, 
                       help='Altitude in km (default: 2.2)')
    parser.add_argument('--wavelengths', nargs='+', type=float, default=[0.355, 0.532, 1.064],
                       help='Wavelengths in μm (default: 0.355 0.532 1.064)')
    
    args = parser.parse_args()
    
    # Configuration from arguments
    file_path = args.file_path
    aerosol_ext = args.aerosol_ext
    altitude_km = args.altitude
    wavelengths = args.wavelengths
    
    # Size distribution parameters
    ln_sigma = 0.6  # ln(σ_g) where σ_g is geometric standard deviation
    
    # Read the data
    data = read_netcdf_data(file_path)
    
    # Auto-detect or use specified particle type
    if args.particle_type:
        particle_type = args.particle_type
        ratio_index = 1 if particle_type == 'spheroidal' else 0
        needs_interpolation = particle_type == 'spheroidal'
        print(f"\nUsing specified particle type: {particle_type}")
    else:
        particle_type, ratio_index, needs_interpolation = detect_particle_type(file_path, data['scama'].shape)
        print(f"\nAuto-detected particle type: {particle_type}")
    
    print(f"Using ratio index: {ratio_index}")
    print(f"Interpolation needed: {needs_interpolation}")
    
    # Handle spheroidal data interpolation
    if needs_interpolation:
        reference_mr = get_reference_mr_values()
        print(f"\nInterpolating to reference mr values: {reference_mr}")
        data = interpolate_to_reference_mr(data, reference_mr, ratio_index)
    
    # Particle size range (effective radius in μm)
    r_eff_min = 0.300  # 300 nm
    r_eff_max = 6.000  # 6000 nm
    r_eff_values = np.linspace(r_eff_min, r_eff_max, 20)
    
    # Mi filtering criteria
    mi_max = 0.01
    
    # Process each wavelength
    for target_wavelength in wavelengths:
        print(f"\n{'='*60}")
        print(f"Processing wavelength: {target_wavelength*1000:.0f} nm")
        print(f"{'='*60}")
        
        # Calculate Rayleigh scattering properties
        rayleigh_ext, rayleigh_depol = calculate_rayleigh_properties(altitude_km, target_wavelength)
        
        print(f"\nRayleigh scattering at {altitude_km} km altitude:")
        print(f"  Extinction coefficient: {rayleigh_ext:.4f} Mm^-1")
        print(f"  Depolarization ratio: {rayleigh_depol:.4f}")
        
        # Calculate contributions
        total_ext = aerosol_ext + rayleigh_ext
        aerosol_contribution = aerosol_ext / total_ext * 100
        rayleigh_contribution = rayleigh_ext / total_ext * 100
        
        print(f"  Aerosol contribution: {aerosol_contribution:.1f}%")
        print(f"  Rayleigh contribution: {rayleigh_contribution:.1f}%")
        
        # Filter mi values
        mi_mask = np.abs(data['mi']) <= mi_max
        
        print(f"\nPolydisperse lognormal distributions:")
        print(f"  ln(σ) = {ln_sigma}")
        print(f"  σ_g = {np.exp(ln_sigma):.3f}")
        print(f"  Effective radius range: {r_eff_min*1000:.0f} - {r_eff_max*1000:.0f} nm")
        print(f"  Mi range: |mi| <= {mi_max}")
        print(f"  Mi points after filtering: {np.sum(mi_mask)}")
        
        # Calculate depolarization values for all combinations first to get global range
        print("Calculating polydisperse depolarization values...")
        all_depol_values = []
        
        for mr_idx in range(len(data['mr'])):
            for mi_idx in np.where(mi_mask)[0]:
                for r_eff in r_eff_values:
                    total_depol = calculate_polydisperse_depolarization(
                        r_eff, mr_idx, mi_idx, data, target_wavelength,
                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index)
                    all_depol_values.append(total_depol)
        
        # Get global min/max for consistent scaling
        global_min = np.min(all_depol_values)
        global_max = np.max(all_depol_values)
        
        # Handle case where range is very small
        if global_max - global_min < 1e-6:
            print(f"Warning: Very small depolarization range detected")
            # Add small artificial range for plotting
            center = (global_min + global_max) / 2
            half_range = max(1e-4, abs(center) * 0.1)  # 10% of center value or minimum 1e-4
            global_min = center - half_range
            global_max = center + half_range
            
        print(f"Global depolarization range: {global_min:.4f} to {global_max:.4f}")
        
        # Create figure with subplots (2x4 for 8 mr values)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        print(f"Creating {len(data['mr'])} subplots for mr values")
        
        # Plot for each mr value
        for mr_idx, mr_val in enumerate(data['mr']):
            ax = axes[mr_idx]
            
            # Initialize arrays for contour data
            X_reff = []
            Y_mi = []
            Z_depol = []
            
            # Loop through filtered mi and effective radius values
            for mi_idx in np.where(mi_mask)[0]:
                mi_val = data['mi'][mi_idx]
                
                for r_eff in r_eff_values:
                    # Calculate polydisperse depolarization
                    total_depol = calculate_polydisperse_depolarization(
                        r_eff, mr_idx, mi_idx, data, target_wavelength,
                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index)
                    
                    # Store for plotting
                    X_reff.append(r_eff * 1000)  # Convert to nm
                    Y_mi.append(abs(mi_val))  # Use absolute value
                    Z_depol.append(total_depol)
            
            # Create regular grid for contour plotting
            if len(X_reff) > 0:
                # Create meshgrid with higher resolution for smoother plots
                reff_grid = np.linspace(r_eff_min*1000, r_eff_max*1000, 80)
                
                # Create mi grid with denser sampling in low mi region
                mi_low = np.linspace(1e-4, 1e-3, 20)  # Dense sampling for very low mi
                mi_high = np.linspace(1e-3, mi_max, 35)  # Regular sampling for higher mi  
                mi_grid = np.concatenate([mi_low, mi_high[1:]])  # Remove duplicate at 1e-3
                
                X_grid, Y_grid = np.meshgrid(reff_grid, mi_grid)
                
                # Interpolate data onto grid preserving variation
                from scipy.interpolate import griddata
                
                # Use cubic interpolation first, then fill gaps with linear
                Z_grid = griddata((X_reff, Y_mi), Z_depol, (X_grid, Y_grid), method='cubic')
                
                # Fill NaN values with linear interpolation 
                nan_mask = np.isnan(Z_grid)
                if np.any(nan_mask):
                    Z_grid_linear = griddata((X_reff, Y_mi), Z_depol, (X_grid, Y_grid), method='linear')
                    Z_grid[nan_mask] = Z_grid_linear[nan_mask]
                    
                    # Final fallback to nearest neighbor for any remaining NaNs
                    nan_mask = np.isnan(Z_grid)
                    if np.any(nan_mask):
                        Z_grid_nearest = griddata((X_reff, Y_mi), Z_depol, (X_grid, Y_grid), method='nearest')
                        Z_grid[nan_mask] = Z_grid_nearest[nan_mask]
                
                # Create smooth contour plot with optimal color levels for discrimination
                # Use global min/max for consistent scaling across all subplots
                levels = np.linspace(global_min, global_max, 50)  # 50 levels for good smoothness with visible discrimination
                contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=levels, cmap='viridis', extend='both', antialiased=True)
                
                # Format plot
                ax.set_xlabel('Effective Radius (nm)')
                ax.set_ylabel('Imaginary RI (|mi|)')
                ax.set_title(f'mr = {mr_val:.3f}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar to last subplot in each row
                if mr_idx == 3 or mr_idx == 7:
                    cbar = plt.colorbar(contour, ax=ax)
                    cbar.set_label('Total Depolarization Ratio')
                    # Set colorbar limits to global range for consistency with more ticks for smoother appearance
                    cbar.set_ticks(np.linspace(global_min, global_max, 8))
                    cbar.set_ticklabels([f'{val:.3f}' for val in np.linspace(global_min, global_max, 8)])
            else:
                ax.text(0.5, 0.5, 'No valid data\nfor this mr value', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_xlabel('Effective Radius (nm)')
                ax.set_ylabel('Imaginary RI (|mi|)')
                ax.set_title(f'mr = {mr_val:.3f}')
                ax.grid(True, alpha=0.3)
        
        # Overall figure formatting
        plt.suptitle(f'Total Lidar Depolarization Ratio at {target_wavelength*1000:.0f} nm\n'
                    f'Polydisperse {particle_type} aerosols (ln(σ) = {ln_sigma})\n'
                    f'Aerosol extinction: {aerosol_ext} Mm$^{{-1}}$, Altitude: {altitude_km} km', fontsize=14)
        
        plt.tight_layout()
        
        # Save the figure with higher DPI for smoother appearance
        output_file = f'lidar_depolarization_{particle_type}_polydisperse_{target_wavelength*1000:.0f}nm.png'
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        print(f"\nFigure saved as: {output_file}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    print(f"\nAll wavelengths processed successfully!")

if __name__ == "__main__":
    main() 