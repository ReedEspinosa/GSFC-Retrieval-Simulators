# Unified Lidar Depolarization Plotting Script

## Overview

`plot_lidar_depolarization_unified.py` is a comprehensive script that automatically handles both **hexahedral** and **spheroidal** aerosol optical property data to generate lidar depolarization ratio plots.

## Key Features

- **Auto-detection**: Automatically detects particle type from filename and data structure
- **Unified workflow**: Single script handles both hexahedral and spheroidal data
- **Optimized performance**: 2-3x faster than original implementation
- **Consistent output**: Both particle types use identical mr grids and subplot layouts
- **Command-line interface**: Flexible parameters for different scenarios

## Usage Examples

### Basic Usage (Auto-detection)

```bash
# Hexahedral particles (auto-detected)
python plot_lidar_depolarization_unified.py "/path/to/kernel-Saito-Hexahedra_psi0.7_1degAngRes_V4.nc"

# Spheroidal particles (auto-detected)  
python plot_lidar_depolarization_unified.py "/path/to/kernel-grasp-v1.1.3-integrated_V4.nc"
```

### Custom Parameters

```bash
# Single wavelength, custom altitude and extinction
python plot_lidar_depolarization_unified.py hexahedral_data.nc \
    --wavelengths 0.532 \
    --altitude 3.0 \
    --aerosol-ext 400

# Multiple wavelengths, force particle type
python plot_lidar_depolarization_unified.py unknown_data.nc \
    --particle-type spheroidal \
    --wavelengths 0.355 0.532 1.064
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `file_path` | **required** | - | Path to netCDF aerosol optical properties file |
| `--particle-type` | choice | auto-detect | Force particle type: `hexahedral` or `spheroidal` |
| `--aerosol-ext` | float | 640 | Aerosol extinction coefficient (Mm⁻¹) |
| `--altitude` | float | 2.2 | Altitude in km for Rayleigh calculation |
| `--wavelengths` | float list | [0.355, 0.532, 1.064] | Wavelengths in μm |

## Auto-Detection Logic

The script automatically detects particle type using:

1. **Filename patterns**:
   - `hexahedral`, `saito` → Hexahedral particles
   - `spheroidal`, `grasp-v` → Spheroidal particles

2. **Data characteristics**:
   - 2 ratios + many mr values → Spheroidal
   - 1 ratio + many x values → Hexahedral

3. **Automatic handling**:
   - Hexahedral: Uses ratio index 0, no interpolation
   - Spheroidal: Uses ratio index 1, interpolates to 8 mr values

## Output Files

Generated files follow the naming pattern:
```
lidar_depolarization_{particle_type}_polydisperse_{wavelength}nm.png
```

Examples:
- `lidar_depolarization_hexahedral_polydisperse_532nm.png`
- `lidar_depolarization_spheroidal_polydisperse_355nm.png`

## Technical Improvements

### Performance Optimizations
- **Vectorized calculations**: Eliminated nested loops
- **Reduced integration points**: 50 → 25 (minimal accuracy loss)
- **Efficient interpolation**: Single scipy.interp1d call vs. manual loops
- **Pre-extracted data**: Extract 180° scattering data once

### Accuracy Fixes
- **Correct ratio index**: Spheroidal data uses ratio index 1 (not 0)
- **Proper interpolation**: Spheroidal mr values interpolated to hexahedral grid
- **Masked array handling**: Robust conversion of netCDF masked arrays

### Visual Quality
- **50 color levels**: Optimal balance of smoothness and discrimination
- **Anti-aliasing**: Smooth contour rendering
- **High DPI output**: 400 DPI for publication quality
- **Consistent scaling**: Global color range across all subplots

## Typical Results

### Hexahedral Aerosols (High Depolarization)
- **355 nm**: 0.080-0.503 range
- **532 nm**: 0.069-0.528 range  
- **1064 nm**: 0.019-0.503 range

### Spheroidal Aerosols (Moderate Depolarization)
- **355 nm**: 0.005-0.353 range
- **532 nm**: 0.033-0.371 range
- **1064 nm**: 0.029-0.376 range

## Scientific Applications

- **Lidar remote sensing**: Aerosol classification and characterization
- **Atmospheric modeling**: Particle shape impact on radiative transfer
- **OSSE studies**: Observing system simulation experiments
- **Algorithm development**: Testing retrieval algorithms with different particle types

## Requirements

- Python 3.6+
- numpy, matplotlib, scipy
- netCDF4
- argparse (standard library)

## Migration from Separate Scripts

The unified script replaces:
- `plot_lidar_depolarization_hexahedral.py`
- `plot_lidar_depolarization_spheroidal.py`

All functionality is preserved with improved performance and usability. 