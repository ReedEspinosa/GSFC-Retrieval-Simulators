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

# Anti-aliased plotting for sharp gradients (default)
python plot_lidar_depolarization_unified.py data.nc \
    --plot-method pcolormesh

# Traditional smooth contours with contour lines
python plot_lidar_depolarization_unified.py data.nc \
    --plot-method contourf
```

### Volume Median Radius Output
The script automatically converts effective radius to volume median radius for display:
```
Volume median radius range: 70 - 3488 nm (log spacing)
Effective radius range: 100 - 5000 nm (internal calculation)
```
- **Conversion formula**: r_v = r_eff × exp(-ln²(σ_g)) ≈ r_eff × 0.698 for ln(σ) = 0.6
- **Display**: X-axis shows volume median radius for publication consistency
- **Calculations**: Internal polydisperse calculations use effective radius for accuracy

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `file_path` | **required** | - | Path to netCDF aerosol optical properties file |
| `--particle-type` | choice | auto-detect | Force particle type: `hexahedral` or `spheroidal` |
| `--aerosol-ext` | float | 640 | Aerosol extinction coefficient (Mm⁻¹) |
| `--altitude` | float | 2.2 | Altitude in km for Rayleigh calculation |
| `--wavelengths` | float list | [0.355, 0.532, 1.064] | Wavelengths in μm |
| `--plot-method` | choice | pcolormesh | Plotting method: `pcolormesh` (anti-aliased) or `contourf` (smooth) |

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

### Publication-Quality Output Characteristics
- **File sizes**: 697K-838K (optimized with anti-aliased pcolormesh rendering)
- **Resolution**: 400 DPI for crisp publication figures with 200×150 grid resolution
- **Anti-aliasing**: pcolormesh method eliminates artifacts in sharp gradient regions
- **Axis scaling**: Log-log scaling with major/minor grid lines for optimal physics representation
- **Size parameter**: Volume median radius (70-3488 nm) for atmospheric science consistency
- **Size conversion**: Internal calculations use effective radius, display shows volume median radius
- **mi coverage**: Enhanced 1e-4 to 0.01 range with superior low-absorption resolution
- **Plotting options**: pcolormesh (default, anti-aliased) or contourf (smooth contours)
- **Format**: High-quality PNG with transparency support

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
- **Volume median radius**: Accurate conversion from effective radius using lognormal relationships

### Visual Quality (Publication-Ready)
- **Anti-aliasing**: pcolormesh plotting reduces artifacts in sharp gradient regions
- **High resolution**: 200×150 grid points for smooth, detailed representation
- **Log-log scaling**: Both axes use logarithmic scaling for optimal physics representation
  - **X-axis**: Log-scale volume median radius for consistency with atmospheric science papers
  - **Y-axis**: Log-scale imaginary RI for enhanced low-mi detail
- **Size range**: 70-3488 nm volume median radius (equivalent to 100-5000 nm effective radius)
- **Enhanced mi coverage**: 1e-4 to 0.01 imaginary RI with log spacing
- **Flexible plotting**: Choose between pcolormesh (anti-aliased) or contourf (smooth contours)
  - Large range (>0.4): 0.1 spacing (hexahedral particles)
  - Medium range (0.1-0.4): 0.05 spacing (typical spheroidal)
  - Small range (<0.1): 0.02 spacing (low-depolarization cases)
- **Enhanced grid**: Both major and minor grid lines for both log axes
- **Anti-aliasing**: Smooth contour rendering with sub-pixel accuracy
- **High DPI output**: 400 DPI for crisp publication-quality figures
- **Consistent scaling**: Global color range across all subplots for easy comparison

## Typical Results

### Hexahedral Aerosols (High Depolarization) - Anti-Aliased pcolormesh
- **355 nm**: 0.019-0.502 range, 770K file size
- **532 nm**: 0.009-0.525 range, 738K file size  
- **1064 nm**: 0.002-0.487 range, 697K file size

### Spheroidal Aerosols (Moderate Depolarization) - Anti-Aliased pcolormesh
- **355 nm**: 0.018-0.346 range, 838K file size
- **532 nm**: 0.008-0.364 range, 792K file size
- **1064 nm**: 0.004-0.369 range, 725K file size

**Key Anti-Aliasing Benefits:**
- **Sharp gradient handling**: pcolormesh eliminates interpolation artifacts in high-gradient regions
- **File efficiency**: 5-10× smaller files (697K-838K vs 3.9-7.2 MB) with better quality
- **Computational efficiency**: Higher resolution (200×150) grid with optimized rendering
- **Publication ready**: Discrete cell method preserves fine details in complex patterns

**Key Volume Median Radius Benefits:**
- **Consistency**: Matches standard atmospheric science parameter conventions
- **Natural scaling**: Volume median radius naturally represents lognormal distributions
- **Paper compatibility**: Consistent with other atmospheric modeling plots
- **Physics representation**: Better alignment with size distribution moments

## Scientific Applications

- **Lidar remote sensing**: Aerosol classification and characterization with optimal parameter space resolution
- **Atmospheric modeling**: Particle shape impact on radiative transfer across full physical parameter range
- **OSSE studies**: Observing system simulation experiments with realistic size and absorption distributions
- **Algorithm development**: Testing retrieval algorithms with comprehensive parameter space coverage
- **Publication consistency**: Volume median radius matches standard atmospheric science conventions
- **Multi-modal analysis**: Superior discrimination between fine mode (~70-500 nm) and coarse mode (~1-3.5 μm)
- **Size distribution studies**: Natural representation of lognormal aerosol size distributions
- **Cross-study comparisons**: Compatible with other atmospheric modeling and measurement papers
- **Absorption characterization**: Log mi-scale reveals critical differences in weakly vs. moderately absorbing particles
- **Physics-based visualization**: Log-log scaling with volume median radius naturally represents aerosol parameter spaces

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