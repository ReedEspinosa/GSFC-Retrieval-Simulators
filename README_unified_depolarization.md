# Unified Lidar Depolarization Comparison Tool

## Overview

`plot_lidar_depolarization_unified.py` creates publication-quality **8-panel comparison plots** showing hexahedral vs spheroidal aerosol depolarization ratios side-by-side with identical parameters and consistent scaling.

## Key Features

- **Direct comparison**: Spheroidal (top row) vs Hexahedral (bottom row) in single plot
- **Unified scaling**: Global color range spans both particle types for accurate comparison
- **Specific mr values**: Fixed comparison at mr = [1.37, 1.47, 1.57, 1.67]
- **Extended mi range**: Up to 0.025 for comprehensive absorption coverage
- **Optimized parameters**: ln(σ) = 0.547 for realistic atmospheric distributions
- **Anti-aliased rendering**: High-resolution pcolormesh eliminates artifacts

## Usage Examples

### Basic Comparison (All Wavelengths)

```bash
# Generate 8-panel comparison plots for all three wavelengths
python plot_lidar_depolarization_unified.py \
    "/path/to/kernel-Saito-Hexahedra_psi0.7_1degAngRes_V4.nc" \
    "/path/to/kernel-grasp-v1.1.3-integrated_V4.nc"
```

### Custom Parameters

```bash
# Single wavelength with custom conditions
python plot_lidar_depolarization_unified.py \
    hexahedral_data.nc \
    spheroidal_data.nc \
    --wavelengths 0.532 \
    --altitude 3.0 \
    --aerosol-ext 400

# Anti-aliased plotting (default, recommended)
python plot_lidar_depolarization_unified.py \
    hexahedral_data.nc \
    spheroidal_data.nc \
    --plot-method pcolormesh

# Traditional smooth contours
python plot_lidar_depolarization_unified.py \
    hexahedral_data.nc \
    spheroidal_data.nc \
    --plot-method contourf
```

### Updated Parameters & Output
The script uses optimized atmospheric parameters:
```
ln(σ) = 0.547
σ_g = 1.728  
Volume median radius range: 99 - 8007 nm (log spacing, optimized for tick display)
Effective radius range: 134 - 10800 nm (internal calculation)
Imaginary RI range: 1e-4 <= |k| <= 0.01 (standard absorption coverage)
mr values: [1.37, 1.47, 1.57, 1.67] (fixed comparison points)
Colorbar range: 0.0 - max(data) (zero-anchored for consistent scaling)
```
- **Conversion formula**: r_v = r_eff × exp(-ln²(σ_g)) ≈ r_eff × 0.741 for ln(σ) = 0.547
- **Display**: X-axis shows volume median radius for publication consistency
- **Layout improvements**: Cleaner axis labeling, shared colorbar, optimal spacing

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `hexahedral_file` | **required** | - | Path to hexahedral netCDF file |
| `spheroidal_file` | **required** | - | Path to spheroidal netCDF file |
| `--aerosol-ext` | float | 640 | Aerosol extinction coefficient (Mm⁻¹) |
| `--altitude` | float | 2.2 | Altitude in km for Rayleigh calculation |
| `--wavelengths` | float list | [0.355, 0.532, 1.064] | Wavelengths in μm |
| `--plot-method` | choice | pcolormesh | Plotting method: `pcolormesh` (anti-aliased) or `contourf` (smooth) |

## 8-Panel Layout

The unified comparison plot structure:

1. **Top row (Spheroidal)**: 4 panels showing spheroidal depolarization
   - Uses ratio index 1 from scattering matrix
   - Interpolated to target mr values for consistency

2. **Bottom row (Hexahedral)**: 4 panels showing hexahedral depolarization  
   - Uses ratio index 0 from scattering matrix
   - Interpolated to same target mr values

3. **Consistent scaling**:
   - Global min/max calculated across both particle types
   - Single color range ensures accurate visual comparison
   - Separate colorbars for each particle type with identical scaling

## Output Files

Generated unified comparison files follow the naming pattern:
```
lidar_depolarization_unified_comparison_{wavelength}nm.png
```

Examples:
- `lidar_depolarization_unified_comparison_355nm.png`
- `lidar_depolarization_unified_comparison_532nm.png`  
- `lidar_depolarization_unified_comparison_1064nm.png`

### Publication-Quality Output Characteristics
- **File sizes**: 705K-793K (optimized clean layout without visual distractions)
- **Figure dimensions**: 18×10 inches (8-panel comparison layout)
- **Resolution**: 400 DPI for crisp publication figures with 200×150 grid resolution
- **Layout optimization**: Constrained layout prevents colorbar width distortion
- **Axis labeling**: Clean layout with labels only on outer axes (left column, bottom row)
- **Y-axis notation**: Imaginary RI (|k|) following standard atmospheric science convention
- **Colorbar**: Single shared colorbar spanning both particle types, zero-anchored scaling
- **Clean visualization**: Smooth pcolormesh plots without contour line distractions
- **Anti-aliasing**: pcolormesh method eliminates artifacts in sharp gradient regions
- **Colormap**: Plasma colormap for enhanced visual contrast and accessibility
- **Axis scaling**: Log-log scaling with major/minor grid lines for optimal physics representation
- **Size parameter**: Volume median radius (99-8007 nm) optimized for tick display
- **Size conversion**: Internal calculations use effective radius, display shows volume median radius
- **Imaginary RI coverage**: Standard 1e-4 to 0.01 range for atmospheric aerosol absorption
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

### Unified Comparison Results (ln(σ) = 0.547, mi ≤ 0.01, Clean Layout)
Direct side-by-side comparison with zero-anchored scaling and optimized clean layout:

- **355 nm**: 0.000-0.505 global range, 793K file size
- **532 nm**: 0.000-0.531 global range, 758K file size
- **1064 nm**: 0.000-0.536 global range, 705K file size

### Key Comparison Insights
- **Hexahedral particles**: Consistently higher depolarization across all mr values
- **Spheroidal particles**: Lower depolarization but significant variation with mr
- **Wavelength dependence**: Both particle types show similar spectral patterns
- **Extended size range**: Volume median radius (99-8007 nm) covers super-coarse mode particles
- **Clean visualization**: Smooth plasma colormap without contour line distractions
- **Direct comparison**: Clear side-by-side assessment with optimal visual clarity

**Key Unified Comparison Benefits:**
- **Direct comparison**: Side-by-side visualization eliminates scaling bias between particle types
- **Zero-anchored scaling**: Colorbar minimum fixed at 0.0 for consistent interpretation
- **Clean visualization**: Smooth plasma colormap provides clear discrimination without distractions
- **Optimal layout**: Axis labels only on outer edges, shared colorbar prevents width distortion
- **Publication efficiency**: Single comprehensive figure replaces multiple separate plots
- **Scientific insight**: Clear visualization of particle shape effects across parameter space

**Key Technical Benefits:**
- **Layout optimization**: Constrained layout with proper colorbar positioning
- **Axis efficiency**: Reduced label redundancy improves visual clarity
- **Standard notation**: |k| symbol for imaginary refractive index following atmospheric conventions
- **Optimal tick display**: 99-8007 nm range ensures 100 nm tick is visible
- **Anti-aliasing**: pcolormesh eliminates interpolation artifacts in high-gradient regions
- **Plasma colormap**: Enhanced visual contrast for better discrimination and accessibility
- **Volume median radius**: Consistent with atmospheric science publication standards
- **Efficient files**: 705K-793K optimized for web and print publication

## Scientific Applications

- **Particle shape comparison**: Direct quantitative assessment of hexahedral vs spheroidal depolarization differences
- **Lidar algorithm development**: Unified reference for testing shape-sensitive retrieval algorithms
- **OSSE studies**: Comprehensive particle shape impact assessment across realistic parameter ranges
- **Atmospheric modeling**: Quantify particle shape effects on lidar observations with consistent scaling
- **Publication-ready figures**: Single comprehensive plots suitable for manuscripts and presentations
- **Extended absorption studies**: mi ≤ 0.025 covers dust, biomass burning, and urban aerosols
- **Size distribution optimization**: ln(σ) = 0.547 represents realistic atmospheric polydispersity
- **Multi-wavelength analysis**: Consistent wavelength dependence comparison across particle shapes
- **Parameter space exploration**: Fixed mr = [1.37, 1.47, 1.57, 1.67] enables systematic comparison
- **Cross-validation studies**: Unified scaling enables accurate relative performance assessment

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