import numpy as np

def calculate_lidar_depolarization(
    r_grid, qsca, p11_180, p22_180, vol_median_radius, ln_sigma
):
    """
    Calculates the bulk lidar depolarization ratio for a lognormal particle
    volume distribution.

    The function integrates single-particle optical properties over a lognormal
    area distribution derived from the provided volume distribution parameters.

    Args:
        r_grid (np.ndarray): 1D array of particle radii for which the optical
                             properties are defined [micrometers].
        qsca (np.ndarray): 1D array of scattering efficiencies corresponding
                           to r_grid.
        p11_180 (np.ndarray): 1D array of the P11 scattering matrix element at
                              180 degrees (backscatter) for each radius in r_grid.
        p22_180 (np.ndarray): 1D array of the P22 scattering matrix element at
                              180 degrees for each radius in r_grid.
        vol_median_radius (float): The volume median radius of the lognormal
                                   distribution [micrometers].
        ln_sigma (float): The natural logarithm of the geometric standard
                          deviation (sigma_g) of the size distribution.

    Returns:
        float: The calculated bulk lidar depolarization ratio, a dimensionless
               quantity.
    """
    # The integration for bulk optical properties is weighted by the particle
    # surface area (since C_sca = q_sca * pi * r^2). We can perform the
    # integration over the surface area distribution directly.

    # We convert the volume median radius to an area median radius.
    # The relationship is: ln(r_volume_median) = ln(r_area_median) + (ln(sigma_g))^2
    log_r_area_median = np.log(vol_median_radius) - ln_sigma**2
    r_area_median = np.exp(log_r_area_median)

    # Define the lognormal area distribution a(r). The total area
    # concentration cancels out in the final ratio, so we can omit it.
    # a(r) propto (1/r) * exp(-[ln(r) - ln(ra)]^2 / (2 * ln_sigma^2))
    log_term = (np.log(r_grid) - np.log(r_area_median))**2
    exp_term = np.exp(-log_term / (2 * ln_sigma**2))
    # The full a(r) has other constants, but they cancel in the ratio.
    area_dist = (1 / r_grid) * exp_term

    # The integrands are weighted by the scattering efficiency and area distribution.
    # Integrand propto q_sca * PhaseFunction * a(r)

    # Perpendicular backscatter intensity is proportional to (P11 - P22)
    integrand_perp = qsca * (p11_180 - p22_180) * area_dist

    # Parallel backscatter intensity is proportional to (P11 + P22)
    integrand_parallel = qsca * (p11_180 + p22_180) * area_dist

    # Numerically integrate using the trapezoidal rule.
    bulk_backscatter_perp = np.trapz(integrand_perp, r_grid)
    bulk_backscatter_parallel = np.trapz(integrand_parallel, r_grid)

    # Avoid division by zero if there is no parallel backscatter.
    if bulk_backscatter_parallel == 0:
        return 0.0

    # The bulk depolarization ratio is the ratio of these two integrated quantities.
    depol_ratio = bulk_backscatter_perp / bulk_backscatter_parallel

    return depol_ratio

# --- Example Usage ---

if __name__ == '__main__':
    # 1. Define the size grid for the optical properties.
    # This grid should be fine enough to resolve features in the optical properties.
    r_grid = np.linspace(0.01, 20, 1000)  # Radii from 0.01 to 20 micrometers

    # 2. Define the single-particle optical properties.
    # NOTE: These are synthetic properties for demonstration. In a real scenario,
    # you would obtain these from a scattering model (e.g., Mie, T-matrix).
    # For perfect spheres, P11 = P22, and depolarization is 0.
    # For non-spherical particles, P11 > P22, leading to non-zero depolarization.
    qsca = 2.5 * (1 - np.exp(-0.8 * r_grid))  # A plausible scattering efficiency curve
    p11_180 = 1.5 * np.ones_like(r_grid)      # P11 is often normalized in some way
    # Let's model P22 to be less than P11, simulating non-sphericity.
    # This function creates a dip, making certain sizes more depolarizing.
    p22_180 = p11_180 * (1 - 0.4 * np.exp(-((r_grid - 3.0)**2) / 4.0))

    # 3. Define the parameters of the bulk aerosol lognormal volume distribution.
    vol_median_radius_dist = 2.5  # micrometers
    ln_sigma_dist = 0.6           # A moderately broad distribution

    # 4. Calculate the bulk lidar depolarization ratio.
    bulk_depolarization = calculate_lidar_depolarization(
        r_grid, qsca, p11_180, p22_180, vol_median_radius_dist, ln_sigma_dist
    )

    # 5. Print the result.
    print("--- Lidar Depolarization Calculation ---")
    print(f"Volume Median Radius: {vol_median_radius_dist} µm")
    print(f"ln(sigma_g):          {ln_sigma_dist}")
    print("-" * 38)
    print(f"Calculated Bulk Depolarization Ratio: {bulk_depolarization:.4f}")

