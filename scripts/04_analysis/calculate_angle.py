"""
Contact Angle Analysis of a Water Droplet
=========================================
This script merges robust local tangent fitting with a geometric check,
analyzing 2-D density fields produced by LAMMPS chunk/ave.

PRIMARY METHOD: Local tangent fitting with iterative sigma-clipping.
SECONDARY METHOD: Geometric angle check (theta = 2 * arctan(h/a)).

Design:
- Adheres to SOLID and Clean Code principles.
- Easy extension for new simulation systems.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# 1. CONFIGURATION & DATA STRUCTURES
# =============================================================================

@dataclass
class SystemConfig:
    """Stores paths and specific behavior flags for a simulation system."""
    name: str
    density_file: str
    data_file: str
    out_dir: str
    is_flat: bool

CONFIGS: Dict[str, SystemConfig] = {
    "graphene": SystemConfig(
        name="Graphene",
        density_file="../data/density_graphene.dat",
        data_file="../data/equilibrated_graphene.data",
        out_dir="../output",
        is_flat=True
    ),
    "graphite": SystemConfig(
        name="Graphite",
        density_file="../data/density_graphite.dat",
        data_file="../data/equilibrated_graphite.data",
        out_dir="../output",
        is_flat=True
    ),
    "structured_graphite": SystemConfig(
        name="Structured Graphite",
        density_file="../data/density_structured_graphite.dat",
        data_file="../data/equilibrated_structured_graphite.data",
        out_dir="../output",
        is_flat=False # Acts as "pillars" in the original script
    )
}

# --- ACTIVE SYSTEM SELECTION ---
# Change this variable to analyze a different system (e.g., "graphene", "structured_graphite")
SYSTEM_NAME = "graphite"
CFG = CONFIGS[SYSTEM_NAME]

@dataclass
class AlgorithmSettings:
    """
    Centralized hyperparameters for physics and math algorithms.
    MODIFY THESE VALUES to tune the contact angle fitting behavior.
    """
    substrate_type: int = 3          # LAMMPS atom type for the substrate (Carbon)
    top_layer_tol: float = 0.5       # Tolerance (Å) to find the top Z-coordinate of the substrate
    liquid_cutoff_frac: float = 0.10 # Fraction of max density to consider as 'liquid phase'
    interface_frac: float = 0.50     # Fraction of bulk density defining the Gibbs dividing surface
    smooth_sigma: float = 1.5        # Gaussian blur sigma for 2D density smoothing
    
    # --- FIT WINDOW PARAMETERS (CRITICAL FOR TANGENT PLACEMENT) ---
    # These define the vertical slice (Z-axis in Å) above the contact point
    # where the script will draw the tangent line. 
    # Adjust these if the tangent line catches the hydration layer (set 'lo' higher) 
    # or the droplet apex (set 'hi' lower).
    z_win_lo_flat: float = 4.6       
    z_win_hi_flat: float = 10.8      
    z_win_lo_pillars: float = 6.0    
    z_win_hi_pillars: float = 18.0   
    
    z_bin: float = 0.5               # Bin size for vertical profile compression
    min_pts: int = 5                 # Minimum data points required for a valid linear fit
    
    # --- ROBUST FITTING PARAMETERS ---
    min_r2_target: float = 0.92      # Target R-squared for a "good" tangent fit
    sigma_clip: float = 1.3          # Multiplier for outlier removal (sigma-clipping)
    max_clip_iter: int = 5           # Maximum iterations for removing noise/outliers
    window_step: float = 1.5         # How much to shift the fit window up if the current fit is poor
    max_win_shifts: int = 6          # Maximum number of vertical window shifts allowed

PARAMS = AlgorithmSettings()

# Color Palette for consistent visualization
PALETTE = {
    "left_tangent": "#C0392B",   # Deep Red
    "right_tangent": "#2471A3",  # Steel Blue
    "contour": "#1C2833",        # Near-black
    "liquid": "#AED6F1",         # Pale Blue
    "liquid_hist": "#2471A3",    # Steel blue for histogram
    "bulk_ref": "#922B21",       # Dark red
    "gibbs_ref": "#1A5276",      # Dark blue
    "substrate": "#7F8C8D"       # Gray
}


# =============================================================================
# 2. FILE PARSERS
# =============================================================================

class DataParser:
    """Handles data extraction from LAMMPS output files."""
    
    @staticmethod
    def get_surface_z(data_file_path: str, substrate_type: int, tol: float) -> float:
        """Finds the maximum Z-coordinate of the substrate layer."""
        z_coords = []
        with open(data_file_path, 'r') as f:
            in_atoms = False
            for line in f:
                line = line.strip()
                if line.startswith("Atoms"):
                    in_atoms = True
                    continue
                if not in_atoms or not line or line.startswith("#"):
                    continue
                if any(line.startswith(s) for s in ("Velocities", "Bonds", "Angles")):
                    break
                parts = line.split()
                if len(parts) >= 7 and int(parts[2]) == substrate_type:
                    z_coords.append(float(parts[6]))
        
        if not z_coords:
            raise ValueError(f"No substrate atoms (type {substrate_type}) found in {data_file_path}")
        
        z_coords = np.array(z_coords)
        return float(np.median(z_coords[z_coords >= z_coords.max() - tol]))

    @staticmethod
    def load_density_dat(dat_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads 2D chunk/ave density data generated by LAMMPS."""
        rows = []
        with open(dat_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 5:
                    try:
                        rows.append([float(v) for v in parts])
                    except ValueError:
                        continue
        if not rows:
            raise ValueError(f"No valid data rows found in {dat_file_path}")
        arr = np.array(rows)
        return arr[:, 1], arr[:, 2], arr[:, 4]  # x, z, density


# =============================================================================
# 3. DENSITY FIELD PROCESSING
# =============================================================================

class DensityProcessor:
    """Converts 1D chunk data to 2D meshes and calculates droplet properties."""

    @staticmethod
    def estimate_bulk_density(density: np.ndarray, z: np.ndarray, surface_z: float) -> float:
        """Estimates the liquid bulk density ignoring vapor and substrate layers."""
        above = (z > surface_z) & (density > 0.0)
        rho_above = density[above]
        if rho_above.size == 0:
            raise ValueError("No density found above the substrate.")
        
        cutoff = PARAMS.liquid_cutoff_frac * rho_above.max()
        liquid_bins = rho_above[rho_above > cutoff]
        if liquid_bins.size == 0:
            raise ValueError("No liquid bins found above cutoff.")
            
        rho_bulk = float(np.median(liquid_bins))
        return rho_bulk, cutoff

    @staticmethod
    def create_grid(x: np.ndarray, z: np.ndarray, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates a 2D mesh grid for contouring and plotting."""
        xu, zu = np.unique(x), np.unique(z)
        xi = {v: i for i, v in enumerate(xu)}
        zi = {v: i for i, v in enumerate(zu)}
        
        rho_grid = np.zeros((zu.size, xu.size), dtype=float)
        for xv, zv, rv in zip(x, z, density):
            rho_grid[zi[zv], xi[xv]] = rv
        return xu, zu, rho_grid

    @staticmethod
    def extract_contour(x_grid: np.ndarray, z_grid: np.ndarray, rho_grid: np.ndarray, 
                        level: float, surface_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the Gibbs dividing surface contour from the density field."""
        # Zero below surface to prevent contouring algorithms from snapping to substrate
        cut_idx = np.searchsorted(z_grid, surface_z, side="right")
        safe_grid = rho_grid.copy()
        safe_grid[:cut_idx, :] = 0.0
        
        smooth = gaussian_filter(safe_grid, sigma=PARAMS.smooth_sigma)
        contours = find_contours(smooth, level=level)
        if not contours:
            raise ValueError(f"No contour found at density level {level:.4f}")
            
        # Select the longest bounding-box contour (the main droplet interface)
        best_contour = None
        best_score = -np.inf
        for c in contours:
            zp = np.interp(c[:, 0], np.arange(z_grid.size), z_grid)
            xp = np.interp(c[:, 1], np.arange(x_grid.size), x_grid)
            score = (xp.max() - xp.min()) * (zp.max() - zp.min())
            if score > best_score:
                best_score = score
                best_contour = (xp, zp)
                
        return best_contour


# =============================================================================
# 4. MATH LOGIC: TANGENTS & CHECKS
# =============================================================================

class TangentFitter:
    """Primary method: Local tangent calculation with robust sigma-clipping."""

    @staticmethod
    def _compress_side(xs: np.ndarray, zs: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Bins data vertically and extracts extreme X points to avoid inner contour noise."""
        z0 = zs.min()
        bidx = np.floor((zs - z0) / PARAMS.z_bin).astype(int)
        xs_out, zs_out = [], []
        for b in np.unique(bidx):
            m = bidx == b
            xs_out.append(xs[m].min() if side == "left" else xs[m].max())
            zs_out.append(zs[m].mean())
        return np.array(xs_out), np.array(zs_out)

    @staticmethod
    def _polyfit_r2(xs: np.ndarray, zs: np.ndarray) -> Tuple[float, float, float]:
        """Standard 1D linear regression with R-squared evaluation."""
        m, b = np.polyfit(zs, xs, 1)
        res = xs - (m * zs + b)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((xs - xs.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return m, b, r2

    @staticmethod
    def _sigma_clip_fit(xs: np.ndarray, zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Iteratively removes outliers to find the true macroscopic tangent."""
        m, b, r2 = TangentFitter._polyfit_r2(xs, zs)
        mask = np.ones(len(xs), dtype=bool)

        for _ in range(PARAMS.max_clip_iter):
            residuals = xs - (m * zs + b)
            sigma = residuals.std()
            if sigma < 1e-9:
                break
            new_mask = np.abs(residuals) < PARAMS.sigma_clip * sigma
            if new_mask.sum() < PARAMS.min_pts:
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            m, b, r2 = TangentFitter._polyfit_r2(xs[mask], zs[mask])
            
        return xs[mask], zs[mask], m, b, r2

    @staticmethod
    def fit_side(x_cnt: np.ndarray, z_cnt: np.ndarray, z_contact: float, side: str, is_flat: bool) -> Optional[dict]:
        """Evaluates the contact angle for one side of the droplet."""
        x_mid = np.median(x_cnt)
        z_win_lo = PARAMS.z_win_lo_flat if is_flat else PARAMS.z_win_lo_pillars
        z_win_hi = PARAMS.z_win_hi_flat if is_flat else PARAMS.z_win_hi_pillars
        
        z_lo0 = z_contact + z_win_lo
        z_hi0 = z_contact + z_win_hi

        best_result = None
        n_shifts = PARAMS.max_win_shifts if is_flat else 1

        for shift in range(n_shifts):
            dz = shift * PARAMS.window_step
            z_lo, z_hi = z_lo0 + dz, z_hi0 + dz

            # Isolate the relevant boundary segment
            mask = (z_cnt >= z_lo) & (z_cnt <= z_hi)
            mask &= (x_cnt < x_mid) if side == "left" else (x_cnt > x_mid)
            xs_raw, zs_raw = x_cnt[mask], z_cnt[mask]
            
            if xs_raw.size < PARAMS.min_pts:
                continue

            xs_c, zs_c = TangentFitter._compress_side(xs_raw, zs_raw, side)
            if xs_c.size < PARAMS.min_pts:
                continue

            if is_flat:
                xs_f, zs_f, m, b, r2 = TangentFitter._sigma_clip_fit(xs_c, zs_c)
            else:
                m, b, r2 = TangentFitter._polyfit_r2(xs_c, zs_c)
                xs_f, zs_f = xs_c, zs_c

            # Convert slope to degrees
            angle = float(np.degrees(np.arctan2(1.0, abs(m))))
            if side == "left" and m < 0: angle = 180.0 - angle
            if side == "right" and m > 0: angle = 180.0 - angle

            x_contact = float(m * z_contact + b)

            result = {
                "xs": xs_f, "zs": zs_f, "m": float(m), "b": float(b), "r2": float(r2),
                "angle": angle, "x_contact": x_contact, "side": side, "z_win": (z_lo, z_hi)
            }

            # Return immediately for non-flat substrates, otherwise look for best R2
            if not is_flat: return result
            
            if best_result is None or r2 > best_result["r2"]:
                best_result = result
            if r2 >= PARAMS.min_r2_target:
                break
                
        return best_result


class SecondaryMath:
    """Alternative measurement heuristics (Geometric evaluation)."""
    
    @staticmethod
    def geometric_angle(lin_left: dict, lin_right: dict, z_contact: float, z_apex: float) -> Optional[float]:
        """Calculates theta = 2 * arctan(h/a) using tangent contact points as base width."""
        if not lin_left or not lin_right: return None
        a = 0.5 * (lin_right["x_contact"] - lin_left["x_contact"])
        h = z_apex - z_contact
        if a <= 0 or h <= 0: return None
        return float(np.degrees(2.0 * np.arctan(h / a)))


# =============================================================================
# 5. VISUALIZATION (Publication Ready Plots)
# =============================================================================

class Plotter:
    @staticmethod
    def _setup_style():
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": ":",
            "legend.frameon": True,
            "figure.dpi": 300
        })

    @staticmethod
    def plot_histogram(density: np.ndarray, z: np.ndarray, surface_z: float, 
                       rho_bulk: float, cutoff: float, rho_thr: float, out_path: Path):
        """Generates a density histogram to visualize the bulk and interface thresholds."""
        Plotter._setup_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        
        above = (z > surface_z) & (density > 0.0)
        rho_a = density[above]
        
        ax.hist(rho_a[rho_a > cutoff], bins=200, color=PALETTE["liquid_hist"], alpha=0.85, linewidth=0,
                label=f"Liquid phase ($\\rho > {cutoff:.2f}$)")
        
        ax.axvline(rho_bulk, color=PALETTE["bulk_ref"], lw=1.6, zorder=5, label=f"$\\rho_{{\\rm bulk}} = {rho_bulk:.2f}$")
        ax.axvline(rho_thr, color=PALETTE["gibbs_ref"], lw=1.6, ls="--", zorder=5, label="Gibbs Threshold")
        ax.axvspan(0, cutoff, color="#D5E8F7", alpha=0.45, zorder=0, label="Vapor / Interface")
        
        ax.set_xlim(0, min(rho_a.max() * 1.05, 1.85))
        ax.set_xlabel("Density, $\\rho$ (g cm$^{-3}$)", fontsize=12)
        ax.set_ylabel("Bin count", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_main(x_grid: np.ndarray, z_grid: np.ndarray, rho_grid: np.ndarray, 
                  x_cnt: np.ndarray, z_cnt: np.ndarray,
                  lin_left: dict, lin_right: dict, theta_geom: float,
                  z_contact: float, z_apex: float, avg_tangent: float, angle_err: float,
                  surface_z: float, out_path: Path, sys_label: str):
        """Generates the main 2D density heatmap overlayed with contours and tangent lines."""
        Plotter._setup_style()
        fig, ax = plt.subplots(figsize=(8.0, 6.0)) 
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")

        # 1. Set plot proportions
        ax.set_aspect("equal")

        # 2. Background Density Heatmap
        masked_density = np.ma.masked_where(rho_grid < 0.05, rho_grid)
        levels = np.linspace(0.05, np.max(rho_grid), 15)
        cf = ax.contourf(x_grid, z_grid, masked_density, levels=levels, cmap="Blues", extend="max", alpha=0.85)
        
        # Align colorbar height perfectly with the main plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label("Density (g/cm$^3$)", rotation=270, labelpad=15)

        # 3. Gibbs dividing surface & Substrate plane
        ax.plot(x_cnt, z_cnt, "-", color=PALETTE["contour"], lw=1.8, label="Gibbs dividing surface", zorder=3)
        ax.axhline(surface_z, color=PALETTE["substrate"], ls=":", lw=1.5, zorder=2, label="Substrate plane")

        # 4. Tangent Lines and Arcs
        z_line = np.array([surface_z - 0.5, z_contact + PARAMS.z_win_hi_pillars + 3])
        for fit, color, label_text in [(lin_left, PALETTE["left_tangent"], "Left tangent"), 
                                       (lin_right, PALETTE["right_tangent"], "Right tangent")]:
            if not fit: continue
            x_full = fit["m"] * z_line + fit["b"]
            ax.plot(x_full, z_line, "--", color=color, lw=1.8, zorder=5, solid_capstyle="round", label=label_text)
            ax.plot(fit["x_contact"], surface_z, "o", ms=6, mfc="white", mec=color, mew=1.6, zorder=8)
            
            t1, t2 = (90.0, 90.0 + (180.0 - fit["angle"])) if fit["side"] == "left" else (90.0 - (180.0 - fit["angle"]), 90.0)
            ax.add_patch(mpatches.Arc((fit["x_contact"], surface_z), width=12, height=12, 
                                      theta1=t1, theta2=t2, color=color, lw=1.4, zorder=9))

        # 5. Styling and Layout
        ax.set_xlabel("$x$ (Å)", fontsize=12)
        ax.set_ylabel("$z$ (Å)", fontsize=12)
        ax.set_xlim(x_cnt.min() - 15, x_cnt.max() + 15)
        
        # Add vertical space above the droplet
        ax.set_ylim(surface_z - 5, z_apex + 30)

        # Angle annotation
        angle_text = rf"$\theta = {avg_tangent:.1f}^\circ \pm {angle_err:.1f}^\circ$"
        ax.text(0.50, 0.73, angle_text, transform=ax.transAxes, fontsize=14, 
                va='top', ha='center', color='black')

        # Position legend in the upper right corner
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, 
                  loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    print(f"\n[{CFG.name}] Starting Contact Angle Analysis...")
    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse Data
    surface_z = DataParser.get_surface_z(CFG.data_file, PARAMS.substrate_type, PARAMS.top_layer_tol)
    x, z, density = DataParser.load_density_dat(CFG.density_file)
    print(f"Substrate Z: {surface_z:.2f} Å")

    # 2. Physics & Grid calculations
    rho_bulk, cutoff = DensityProcessor.estimate_bulk_density(density, z, surface_z)
    rho_thr = rho_bulk * PARAMS.interface_frac
    print(f"Bulk density: {rho_bulk:.3f} g/cm³, Gibbs threshold: {rho_thr:.3f} g/cm³")
    
    x_grid, z_grid, rho_grid = DensityProcessor.create_grid(x, z, density)
    x_cnt, z_cnt = DensityProcessor.extract_contour(x_grid, z_grid, rho_grid, rho_thr, surface_z)
    z_contact, z_apex = float(z_cnt.min()), float(z_cnt.max())

    # 3. Fit Angles
    print("\n--- Model Fitting ---")
    lin_left = TangentFitter.fit_side(x_cnt, z_cnt, z_contact, "left", CFG.is_flat)
    lin_right = TangentFitter.fit_side(x_cnt, z_cnt, z_contact, "right", CFG.is_flat)
    
    if not lin_left or not lin_right:
        print("ERROR: Tangent fitting failed. Ensure fit window parameters are correct.")
        return

    # Calculate average angle and error margin (droplet asymmetry)
    avg_tangent = 0.5 * (lin_left["angle"] + lin_right["angle"])
    angle_err = abs(lin_left["angle"] - lin_right["angle"]) / 2.0
    
    print(f"PRIMARY (Tangents) -> Left: {lin_left['angle']:.1f}° | Right: {lin_right['angle']:.1f}°")
    print(f"MAIN RESULT: {avg_tangent:.2f}° ± {angle_err:.2f}°")

    # Secondary geometric check
    theta_geom = SecondaryMath.geometric_angle(lin_left, lin_right, z_contact, z_apex)
    if theta_geom: print(f"CHECK (Geometric): {theta_geom:.2f}°")

    # 4. Generate final plots
    print("\n--- Generating Plots ---")
    hist_out = out_dir / f"histogram_{SYSTEM_NAME}.png"
    main_out = out_dir / f"contact_angle_{SYSTEM_NAME}.png"

    Plotter.plot_histogram(density, z, surface_z, rho_bulk, cutoff, rho_thr, hist_out)
    Plotter.plot_main(x_grid, z_grid, rho_grid, x_cnt, z_cnt, lin_left, lin_right, theta_geom, 
                      z_contact, z_apex, avg_tangent, angle_err, surface_z, main_out, CFG.name)
    
    print(f"Files saved to: {out_dir}")

if __name__ == "__main__":
    main()