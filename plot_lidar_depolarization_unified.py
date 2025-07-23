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
plt.rcParams['image.cmap'] = 'plasma'

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
        data['qext'] = nc.variables['qext'][:]    # Extinction efficiency (same shape as qsca)
        # Add qsca reading
        data['qsca'] = nc.variables['qsca'][:]    # Scattering efficiency (same shape as x)
        
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
    Calculate bulk depolarization for lognormal size distribution using cross-section weighting.
    Scales aerosol contribution so that total extinction matches user-supplied aerosol_ext (Mm^-1).
    P11 and P22 are scattering properties and are scaled by scattering, not extinction.
    
    Parameters:
    - r_eff: effective radius (μm)
    - mr_idx, mi_idx: indices for refractive index
    - data: netCDF data dictionary
    - target_wavelength: wavelength (μm)
    - aerosol_ext: user-supplied aerosol extinction (Mm^-1)
    - rayleigh_ext: Rayleigh extinction coefficient (Mm^-1) - must match units of aerosol_ext
    - rayleigh_depol: Rayleigh depolarization ratio
    - ratio_index: 0 for hexahedral, 1 for spheroidal
    
    Returns:
    - total_depol: bulk depolarization ratio (cross-section weighted, scaled)
    """
    ln_sigma = 0.547  # Given constraint
    sigma_g = np.exp(ln_sigma)  # Geometric std dev
    r_g = r_eff / np.exp(2.5 * ln_sigma**2)
    r_min = r_g / (sigma_g**3)
    r_max = r_g * (sigma_g**3)
    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), 25)
    dV_dlnr = lognormal_volume_distribution(r_grid, r_g, sigma_g)
    x_grid = 2 * np.pi * r_grid / target_wavelength
    x_data = data['x']
    angle_180_idx = -1  # Last angle is 180°
    p11_180 = data['scama'][ratio_index, mr_idx, mi_idx, :, 0, angle_180_idx].squeeze()
    p22_180 = data['scama'][ratio_index, mr_idx, mi_idx, :, 1, angle_180_idx].squeeze()
    if data['qsca'].ndim == 4:
        qsca_data = data['qsca'][ratio_index, mr_idx, mi_idx, :].squeeze()
        qext_data = data['qext'][ratio_index, mr_idx, mi_idx, :].squeeze()
    elif data['qsca'].ndim == 3:
        qsca_data = data['qsca'][mr_idx, mi_idx, :].squeeze()
        qext_data = data['qext'][mr_idx, mi_idx, :].squeeze()
    else:
        raise ValueError(f"Unexpected qsca/qext ndim: {data['qsca'].ndim}")
    from scipy.interpolate import interp1d
    p11_interp = interp1d(x_data, p11_180, bounds_error=False, fill_value='extrapolate')
    p22_interp = interp1d(x_data, p22_180, bounds_error=False, fill_value='extrapolate')
    qsca_interp = interp1d(x_data, qsca_data, bounds_error=False, fill_value='extrapolate')
    qext_interp = interp1d(x_data, qext_data, bounds_error=False, fill_value='extrapolate')
    p11_grid = p11_interp(x_grid)
    p22_grid = p22_interp(x_grid)
    qsca_grid = qsca_interp(x_grid)
    qext_grid = qext_interp(x_grid)
    # Scattering and extinction cross section: Csca = qsca * pi * r^2, Cext = qext * pi * r^2
    csca_grid = qsca_grid * np.pi * r_grid**2
    cext_grid = qext_grid * np.pi * r_grid**2
    v_grid = (4/3) * np.pi * r_grid**3
    # Number-weighted distribution: dN/dlnr = dV/dlnr / v(r)
    number_weight = dV_dlnr / v_grid
    # Weight for cross-section integration: number of particles * cross section
    weight = csca_grid * number_weight
    ln_r_grid = np.log(r_grid)
    # Integrate cross-section-weighted P11 and P22 for aerosol (scattering only)
    sum_p11_aero = np.trapz(p11_grid * weight, ln_r_grid)
    sum_p22_aero = np.trapz(p22_grid * weight, ln_r_grid)
    # Integrate extinction cross section for scaling
    weight_ext = cext_grid * number_weight
    total_aerosol_cext = np.trapz(weight_ext, ln_r_grid)  # [μm^2]
    # Scale by extinction: scale = aerosol_ext / total_aerosol_cext
    # Units: (Mm^-1) / (μm^2) - but since we're using relative contributions,
    # the units cancel in the final depolarization ratio
    if total_aerosol_cext > 0:
        scale = aerosol_ext / total_aerosol_cext
    else:
        scale = 0.0
    sum_p11_aero *= scale # WRE - At this point, this is scaled to match the user-supplied aerosol_ext with units of Sr^-1*Mm^-1
    sum_p22_aero *= scale # WRE - At this point, this is scaled to match the user-supplied aerosol_ext with units of Sr^-1*Mm^-1
    # Rayleigh: get cross section, P11, P22 at 180°
    rayleigh_csca = rayleigh_ext  # Use extinction as cross section for relative weighting
    rayleigh_p11 = 1.0
    rayleigh_p22 = 1.0 - 2 * rayleigh_depol / (1 + rayleigh_depol)
    # Add Rayleigh to sums
    sum_p11 = sum_p11_aero + rayleigh_p11 * rayleigh_csca
    sum_p22 = sum_p22_aero + rayleigh_p22 * rayleigh_csca
    # Bulk depolarization ratio
    total_depol = (sum_p11 - sum_p22) / (sum_p11 + sum_p22)
    if np.isscalar(total_depol):
        return float(total_depol)
    elif hasattr(total_depol, 'shape') and total_depol.shape == ():
        return float(total_depol)
    else:
        raise ValueError(f"total_depol is not scalar, shape={getattr(total_depol, 'shape', None)}, value={total_depol}")

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
    parser.add_argument('hexahedral_file', help='Path to hexahedral netCDF file')
    parser.add_argument('spheroidal_file', help='Path to spheroidal netCDF file')
    parser.add_argument('--aerosol-ext', type=float, default=640, 
                       help='Aerosol extinction coefficient (Mm^-1, default: 640)')
    parser.add_argument('--altitude', type=float, default=2.2, 
                       help='Altitude in km (default: 2.2)')
    parser.add_argument('--wavelengths', nargs='+', type=float, default=[0.355, 0.532, 1.064],
                       help='Wavelengths in μm (default: 0.355 0.532 1.064)')
    parser.add_argument('--plot-method', choices=['contourf', 'pcolormesh'], default='pcolormesh',
                       help='Plotting method: contourf (smooth contours) or pcolormesh (discrete cells, reduces aliasing)')
    
    args = parser.parse_args()
    
    # Configuration from arguments
    hexahedral_file = args.hexahedral_file
    spheroidal_file = args.spheroidal_file
    aerosol_ext = args.aerosol_ext
    altitude_km = args.altitude
    wavelengths = args.wavelengths
    plot_method = args.plot_method
    
    # Size distribution parameters
    ln_sigma = 0.547  # ln(σ_g) where σ_g is geometric standard deviation
    
    # Specific mr values for comparison
    target_mr_values = np.array([1.37, 1.47, 1.57, 1.67])
    
    # Read both datasets
    print("Reading hexahedral data...")
    hex_data = read_netcdf_data(hexahedral_file)
    print("\nReading spheroidal data...")
    sph_data = read_netcdf_data(spheroidal_file)
    
    # Handle spheroidal data interpolation to target mr values
    print(f"\nInterpolating spheroidal data to target mr values: {target_mr_values}")
    sph_data = interpolate_to_reference_mr(sph_data, target_mr_values, ratio_index=1)
    
    # Interpolate hexahedral data to same mr values for consistency
    print(f"Interpolating hexahedral data to target mr values: {target_mr_values}")
    hex_data = interpolate_to_reference_mr(hex_data, target_mr_values, ratio_index=0)
    
    # Particle size range (effective radius in μm) - log spacing for better physics representation
    # Adjusted to give volume median radius range 99-8000 nm (so 100 nm tick shows)
    r_eff_min = 0.1337  # 133.7 nm effective → 99 nm volume median
    r_eff_max = 10.8    # 10,800 nm effective → 8000 nm volume median
    r_eff_values = np.logspace(np.log10(r_eff_min), np.log10(r_eff_max), 20)
    
    # Convert to volume median radius for display consistency with other plots
    # For lognormal distribution: r_v = r_eff * exp(-1.0 * ln²(σ_g))
    ln_sigma = 0.547  # Given constraint
    r_v_conversion_factor = np.exp(-1.0 * ln_sigma**2)  # = exp(-0.299) ≈ 0.7410
    r_v_min = r_eff_min * r_v_conversion_factor  # Volume median radius range
    r_v_max = r_eff_max * r_v_conversion_factor
    
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
        
        # Filter mi values for both datasets
        hex_mi_mask = np.abs(hex_data['mi']) <= mi_max
        sph_mi_mask = np.abs(sph_data['mi']) <= mi_max
        
        print(f"\nPolydisperse lognormal distributions:")
        print(f"  ln(σ) = {ln_sigma}")
        print(f"  σ_g = {np.exp(ln_sigma):.3f}")
        print(f"  Volume median radius range: {r_v_min*1000:.0f} - {r_v_max*1000:.0f} nm (log spacing)")
        print(f"  Effective radius range: {r_eff_min*1000:.0f} - {r_eff_max*1000:.0f} nm (internal calculation)")
        print(f"  Mi range: 1e-4 <= |mi| <= {mi_max} (log spacing)")
        print(f"  Hexahedral mi points after filtering: {np.sum(hex_mi_mask)}")
        print(f"  Spheroidal mi points after filtering: {np.sum(sph_mi_mask)}")
        
        # Calculate depolarization values for both datasets to get global range
        print("Calculating polydisperse depolarization values for both particle types...")
        all_depol_values = []
        
        # Calculate for hexahedral particles
        for mr_idx in range(len(target_mr_values)):
            for mi_idx in np.where(hex_mi_mask)[0]:
                for r_eff in r_eff_values:
                    total_depol = calculate_polydisperse_depolarization(
                        r_eff, mr_idx, mi_idx, hex_data, target_wavelength,
                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index=0)
                    all_depol_values.append(total_depol)
        
        # Calculate for spheroidal particles  
        for mr_idx in range(len(target_mr_values)):
            for mi_idx in np.where(sph_mi_mask)[0]:
                for r_eff in r_eff_values:
                    total_depol = calculate_polydisperse_depolarization(
                        r_eff, mr_idx, mi_idx, sph_data, target_wavelength,
                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index=1)
                    all_depol_values.append(total_depol)
        
        # Get global min/max for consistent scaling across both particle types
        # Force minimum to 0 for consistent colorbar scaling
        global_min = 0.0
        global_max = np.max(all_depol_values)
        
        # Handle case where range is very small
        if global_max - global_min < 1e-6:
            print(f"Warning: Very small depolarization range detected")
            global_max = max(1e-4, global_min + 1e-4)
            
        print(f"Global depolarization range (both particle types): {global_min:.4f} to {global_max:.4f}")
        
        # Determine contour line spacing
        data_range = global_max - global_min
        if data_range > 0.4:
            contour_spacing = 0.1
        elif data_range > 0.1:
            contour_spacing = 0.05
        else:
            contour_spacing = 0.02
        print(f"Using contour line spacing: {contour_spacing:.3f}")
        
        # Create unified figure with 2x4 subplots: spheroids on top, hexahedra on bottom
        # Use constrained layout to prevent colorbar width issues
        fig, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
        
        print(f"Creating unified 8-panel plot: 4 spheroidal (top) + 4 hexahedral (bottom)")
        
        # Plot spheroidal particles in top row
        for mr_idx, mr_val in enumerate(target_mr_values):
            ax = axes[0, mr_idx]  # Top row
            
            # Initialize arrays for contour data
            X_rv = []  # Volume median radius data
            Y_mi = []
            Z_depol = []
            
            # Loop through filtered mi and effective radius values for spheroidal particles
            for mi_idx in np.where(sph_mi_mask)[0]:
                mi_val = sph_data['mi'][mi_idx]
                
                for r_eff in r_eff_values:
                    # Calculate polydisperse depolarization
                    total_depol = calculate_polydisperse_depolarization(
                        r_eff, mr_idx, mi_idx, sph_data, target_wavelength,
                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index=1)
                    
                    # Store for plotting (convert effective radius to volume median radius)
                    r_v = r_eff * r_v_conversion_factor  # Convert to volume median radius
                    X_rv.append(r_v * 1000)  # Convert to nm
                    Y_mi.append(abs(mi_val))  # Use absolute value
                    Z_depol.append(total_depol)
            
            # Create regular grid for contour plotting
            if len(X_rv) > 0:
                # Create high-resolution meshgrid with log spacing to reduce aliasing effects
                rv_grid = np.logspace(np.log10(r_v_min*1000), np.log10(r_v_max*1000), 200)  # Increased resolution
                
                # Create mi grid with log spacing for better physics representation
                mi_min = 1e-4  # Minimum mi value for plotting
                mi_grid = np.logspace(np.log10(mi_min), np.log10(mi_max), 150)  # Increased resolution
                
                X_grid, Y_grid = np.meshgrid(rv_grid, mi_grid)
                
                # Interpolate data onto grid preserving variation
                from scipy.interpolate import griddata
                
                # Use cubic interpolation first, then fill gaps with linear
                Z_grid = griddata((X_rv, Y_mi), Z_depol, (X_grid, Y_grid), method='cubic')
                
                # Fill NaN values with linear interpolation 
                nan_mask = np.isnan(Z_grid)
                if np.any(nan_mask):
                    Z_grid_linear = griddata((X_rv, Y_mi), Z_depol, (X_grid, Y_grid), method='linear')
                    Z_grid[nan_mask] = Z_grid_linear[nan_mask]
                    
                    # Final fallback to nearest neighbor for any remaining NaNs
                    nan_mask = np.isnan(Z_grid)
                    if np.any(nan_mask):
                        Z_grid_nearest = griddata((X_rv, Y_mi), Z_depol, (X_grid, Y_grid), method='nearest')
                        Z_grid[nan_mask] = Z_grid_nearest[nan_mask]
                
                # Create plot using specified method for optimal quality
                # Use global min/max for consistent scaling across all subplots
                if plot_method == 'contourf':
                    # Traditional contour plot with smooth interpolation
                    color_levels = np.linspace(global_min, global_max, 255)  # 255 levels for continuous color bar
                    contour_filled = ax.contourf(X_grid, Y_grid, Z_grid, levels=color_levels, cmap='plasma', extend='both', antialiased=True)
                else:  # pcolormesh
                    # Discrete cell-based plot to reduce aliasing in sharp gradient regions
                    contour_filled = ax.pcolormesh(X_grid, Y_grid, Z_grid, cmap='plasma', 
                                                 vmin=global_min, vmax=global_max, shading='auto', antialiased=True)
                
                # Clean pcolormesh plots without contour lines for better visual clarity
                
                # Format plot with log x-axis and log y-axis
                # Only show x-axis label on bottom row, y-axis label on leftmost column
                if mr_idx == 0:  # Leftmost column
                    ax.set_ylabel('Imaginary RI (|k|)')
                ax.set_title(f'Spheroidal mr = {mr_val:.2f}')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(r_v_min*1000, r_v_max*1000)
                ax.set_ylim(1e-4, mi_max)
                ax.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines for both axes
                
            else:
                ax.text(0.5, 0.5, 'No valid data\nfor this mr value', 
                       transform=ax.transAxes, ha='center', va='center')
                # Only show y-axis label on leftmost column
                if mr_idx == 0:  # Leftmost column
                    ax.set_ylabel('Imaginary RI (|k|)')
                ax.set_title(f'Spheroidal mr = {mr_val:.2f}')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(r_v_min*1000, r_v_max*1000)
                ax.set_ylim(1e-4, mi_max)
                ax.grid(True, alpha=0.3, which='both')
            

        
        # Plot hexahedral particles in bottom row
        for mr_idx, mr_val in enumerate(target_mr_values):
            ax = axes[1, mr_idx]  # Bottom row
            
            # Initialize arrays for contour data
            X_rv = []  # Volume median radius data
            Y_mi = []
            Z_depol = []
            
            # Loop through filtered mi and effective radius values for hexahedral particles
            for mi_idx in np.where(hex_mi_mask)[0]:
                mi_val = hex_data['mi'][mi_idx]
                
                for r_eff in r_eff_values:
                    # Calculate polydisperse depolarization
                    total_depol = calculate_polydisperse_depolarization(
                        r_eff, mr_idx, mi_idx, hex_data, target_wavelength,
                        aerosol_ext, rayleigh_ext, rayleigh_depol, ratio_index=0)
                    
                    # Store for plotting (convert effective radius to volume median radius)
                    r_v = r_eff * r_v_conversion_factor  # Convert to volume median radius
                    X_rv.append(r_v * 1000)  # Convert to nm
                    Y_mi.append(abs(mi_val))  # Use absolute value
                    Z_depol.append(total_depol)
            
            # Create regular grid for contour plotting
            if len(X_rv) > 0:
                # Create high-resolution meshgrid with log spacing to reduce aliasing effects
                rv_grid = np.logspace(np.log10(r_v_min*1000), np.log10(r_v_max*1000), 200)  # Increased resolution
                
                # Create mi grid with log spacing for better physics representation
                mi_min = 1e-4  # Minimum mi value for plotting
                mi_grid = np.logspace(np.log10(mi_min), np.log10(mi_max), 150)  # Increased resolution
                
                X_grid, Y_grid = np.meshgrid(rv_grid, mi_grid)
                
                # Interpolate data onto grid preserving variation
                # Use cubic interpolation first, then fill gaps with linear
                Z_grid = griddata((X_rv, Y_mi), Z_depol, (X_grid, Y_grid), method='cubic')
                
                # Fill NaN values with linear interpolation 
                nan_mask = np.isnan(Z_grid)
                if np.any(nan_mask):
                    Z_grid_linear = griddata((X_rv, Y_mi), Z_depol, (X_grid, Y_grid), method='linear')
                    Z_grid[nan_mask] = Z_grid_linear[nan_mask]
                    
                    # Final fallback to nearest neighbor for any remaining NaNs
                    nan_mask = np.isnan(Z_grid)
                    if np.any(nan_mask):
                        Z_grid_nearest = griddata((X_rv, Y_mi), Z_depol, (X_grid, Y_grid), method='nearest')
                        Z_grid[nan_mask] = Z_grid_nearest[nan_mask]
                
                # Create plot using specified method for optimal quality
                # Use global min/max for consistent scaling across all subplots
                if plot_method == 'contourf':
                    # Traditional contour plot with smooth interpolation
                    color_levels = np.linspace(global_min, global_max, 255)  # 255 levels for continuous color bar
                    contour_filled = ax.contourf(X_grid, Y_grid, Z_grid, levels=color_levels, cmap='plasma', extend='both', antialiased=True)
                else:  # pcolormesh
                    # Discrete cell-based plot to reduce aliasing in sharp gradient regions
                    contour_filled = ax.pcolormesh(X_grid, Y_grid, Z_grid, cmap='plasma', 
                                                 vmin=global_min, vmax=global_max, shading='auto', antialiased=True)
                
                # Clean pcolormesh plots without contour lines for better visual clarity
                
                # Format plot with log x-axis and log y-axis
                # Only show x-axis label on bottom row, y-axis label on leftmost column
                ax.set_xlabel('Volume Median Radius (nm)')  # Bottom row gets x-axis labels
                if mr_idx == 0:  # Leftmost column
                    ax.set_ylabel('Imaginary RI (|k|)')
                ax.set_title(f'Hexahedral mr = {mr_val:.2f}')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(r_v_min*1000, r_v_max*1000)
                ax.set_ylim(1e-4, mi_max)
                ax.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines for both axes
                
            else:
                ax.text(0.5, 0.5, 'No valid data\nfor this mr value', 
                       transform=ax.transAxes, ha='center', va='center')
                # Only show x-axis label on bottom row, y-axis label on leftmost column
                ax.set_xlabel('Volume Median Radius (nm)')  # Bottom row gets x-axis labels
                if mr_idx == 0:  # Leftmost column
                    ax.set_ylabel('Imaginary RI (|k|)')
                ax.set_title(f'Hexahedral mr = {mr_val:.2f}')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(r_v_min*1000, r_v_max*1000)
                ax.set_ylim(1e-4, mi_max)
                ax.grid(True, alpha=0.3, which='both')
                
        # Add shared colorbar on the right side
        # Create colorbar that spans both rows
        cbar = plt.colorbar(contour_filled, ax=axes, shrink=0.8, aspect=30)
        cbar.set_label('Total Depolarization Ratio', fontsize=14)
        cbar.set_ticks(np.linspace(global_min, global_max, 6))
        cbar.set_ticklabels([f'{val:.3f}' for val in np.linspace(global_min, global_max, 6)])
        
        # Overall figure formatting
        plt.suptitle(f'Total Lidar Depolarization Ratio at {target_wavelength*1000:.0f} nm\n'
                    f'Spheroidal (top) vs Hexahedral (bottom) - ln(σ) = {ln_sigma}\n'
                    f'Aerosol extinction: {aerosol_ext} Mm$^{{-1}}$, Altitude: {altitude_km} km, mi ≤ {mi_max}', fontsize=14)
        
        # Save the figure with higher DPI for smoother appearance
        output_file = f'lidar_depolarization_unified_comparison_{target_wavelength*1000:.0f}nm.png'
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        print(f"\nFigure saved as: {output_file}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    print(f"\nAll wavelengths processed successfully!")

if __name__ == "__main__":
    main() 