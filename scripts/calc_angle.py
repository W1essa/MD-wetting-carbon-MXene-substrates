"""
Contact Angle Analysis from MD Density Profile
Calculates contact angle using linear fitting of droplet interface
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = '/home/vbarv/projects/lammps/data/density_final.dat'
OUTPUT_IMAGE = '/home/vbarv/projects/lammps/data/contact_angle_result.png'

# Analysis parameters
LIQUID_THRESHOLD = 0.5  # Fraction of max density to define liquid phase
Z_OFFSET = 2.0          # Skip bottom layer (Angstrom)
FIT_HEIGHT = 15.0       # Height range for linear fit (Angstrom)


# ============================================================================
# FORMULA VERIFICATION
# ============================================================================
# Contact angle θ measured from horizontal substrate:
#
# For a droplet on a surface, the interface slope is:
#   slope = dx/dz = tan(α)  where α is angle from vertical
#
# Contact angle from horizontal:
#   α = arctan(|slope|)  where slope = dx/dz
#   θ = 90° - α  (general case)
#   Handle obtuse angles by checking slope direction
# ============================================================================


def load_density_data(filename):
    """Load density profile data from LAMMPS output."""
    print(f"Reading {filename}...")
    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    data.append([float(p) for p in parts])
                except ValueError:
                    continue
    
    data = np.array(data)
    return data[:, 1], data[:, 2], data[:, 4]  # x, z, density


def extract_liquid_phase(x, z, density, threshold_fraction=0.5):
    """Extract only liquid phase based on density threshold."""
    max_density = np.max(density)
    threshold = max_density * threshold_fraction
    
    mask = density > threshold
    print(f"Max density: {max_density:.3f}")
    print(f"Liquid threshold: {threshold:.3f}")
    print(f"Liquid points: {np.sum(mask)}/{len(mask)}")
    
    return x[mask], z[mask]


def extract_interface_points(x_liq, z_liq, z_min, z_max):
    """Extract left and right interface points within specified z-range."""
    x_center = np.mean(x_liq)
    
    x_left, z_left = [], []
    x_right, z_right = [], []
    
    unique_z = np.unique(z_liq)
    for z in unique_z:
        if z_min < z < z_max:
            x_at_z = x_liq[z_liq == z]
            if len(x_at_z) > 0:
                x_min = np.min(x_at_z)
                x_max = np.max(x_at_z)
                
                if x_min < x_center:
                    x_left.append(x_min)
                    z_left.append(z)
                
                if x_max > x_center:
                    x_right.append(x_max)
                    z_right.append(z)
    
    return np.array(x_left), np.array(z_left), np.array(x_right), np.array(z_right)


def calculate_contact_angle(z_points, x_points, side='left'):
    """
    Calculate contact angle from interface points using linear regression.
    
    Args:
        z_points: vertical coordinates (height)
        x_points: lateral coordinates
        side: 'left' or 'right' interface
    
    Returns:
        contact_angle: angle in degrees from horizontal substrate
        coeffs: [slope, intercept] of linear fit
    """
    if len(z_points) < 3:
        return 0.0, [0.0, 0.0]
    
    # Linear regression: x = slope * z + intercept
    coeffs = np.polyfit(z_points, x_points, 1)
    slope = coeffs[0]  # dx/dz
    
    # slope = tan(alpha) where alpha is angle from vertical
    alpha = np.degrees(np.arctan(np.abs(slope)))
    
    # Contact angle from horizontal substrate
    contact_angle = 90 - alpha
    
    # Handle obtuse angles (>90 deg)
    if side == 'left' and slope < 0:
        contact_angle = 180 - contact_angle
    elif side == 'right' and slope > 0:
        contact_angle = 180 - contact_angle
    
    return contact_angle, coeffs


def plot_results(x_liq, z_liq, x_l, z_l, x_r, z_r, 
                 fit_l, fit_r, angle_avg, z_bottom, fit_height, output_file):
    """Generate visualization of contact angle analysis."""
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Linear Method (Bottom {fit_height:.1f} A): {angle_avg:.1f}°')
    
    # Full droplet shape
    plt.scatter(x_liq, z_liq, c='lightgray', alpha=0.3)
    
    # Interface points
    plt.scatter(x_l, z_l, c='blue', s=15, label='Left Points')
    plt.scatter(x_r, z_r, c='red', s=15, label='Right Points')
    
    # Fitted lines
    z_plot = np.linspace(z_bottom, z_bottom + fit_height + 5, 50)
    x_plot_l = fit_l[0] * z_plot + fit_l[1]
    x_plot_r = fit_r[0] * z_plot + fit_r[1]
    
    plt.plot(x_plot_l, z_plot, 'b-', lw=3, label='Left Slope')
    plt.plot(x_plot_r, z_plot, 'r-', lw=3, label='Right Slope')
    
    # Substrate line
    plt.axhline(z_bottom, c='black', ls='--')
    
    # Formatting
    plt.ylim(z_bottom - 5, z_bottom + 40)
    plt.axis('equal')
    plt.legend()
    
    plt.savefig(output_file)
    print(f"Saved: {output_file}")


def main():
    """Main analysis workflow."""
    
    # Load data
    x, z, density = load_density_data(DATA_FILE)
    
    # Extract liquid phase
    x_liq, z_liq = extract_liquid_phase(x, z, density, LIQUID_THRESHOLD)
    z_bottom = np.min(z_liq)
    print(f"Water Bottom Z: {z_bottom:.2f}")
    
    # Define fitting region
    z_min = z_bottom + Z_OFFSET
    z_max = z_bottom + FIT_HEIGHT
    
    # Extract interface points
    x_l, z_l, x_r, z_r = extract_interface_points(x_liq, z_liq, z_min, z_max)
    
    # Calculate contact angles
    angle_left, fit_left = calculate_contact_angle(z_l, x_l, side='left')
    angle_right, fit_right = calculate_contact_angle(z_r, x_r, side='right')
    angle_avg = (angle_left + angle_right) / 2
    
    # Results
    print("="*40)
    print(f"LEFT Angle:  {angle_left:.2f}°")
    print(f"RIGHT Angle: {angle_right:.2f}°")
    print(f"AVERAGE ANGLE: {angle_avg:.2f}°")
    print("="*40)
    
    # Visualization
    plot_results(x_liq, z_liq, x_l, z_l, x_r, z_r,
                 fit_left, fit_right, angle_avg, z_bottom, FIT_HEIGHT, OUTPUT_IMAGE)


if __name__ == '__main__':
    main()