"""
Contact Angle Analysis of a Water Droplet
=========================================
This script merges robust local tangent fitting with circle-fitting heuristics,
analyzing 2-D density fields produced by LAMMPS chunk/ave.

PRIMARY METHOD: Local tangent fitting with iterative sigma-clipping.
SECONDARY METHOD 1: Geometric angle check (theta = 2 * arctan(h/a)).
SECONDARY METHOD 2: Least-squares circle fit to the contour surface.

Design:
- Adheres to SOLID and Clean Code principles.
- Easy extension for new simulation systems.
- High-quality publication-ready visualization (Density Heatmap).
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from skimage.measure import find_contours

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

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
        density_file="../../data/density_graphene.dat",
        data_file="../../data/equilibrated_graphene.data",
        out_dir="../../output",
        is_flat=True
    ),
    "graphite": SystemConfig(
        name="Graphite",
        density_file="../../data/density_graphite.dat",
        data_file="../../data/equilibrated_graphite.data",
        out_dir="../../output",
        is_flat=True
    ),
    "structured_graphite": SystemConfig(
        name="Structured Graphite",
        density_file="../../data/density_structured_graphite.dat",
        data_file="../../data/equilibrated_structured_graphite.data",
        out_dir="../../output",
        is_flat=False # Acts as "pillars" in the original script
    )
}

# --- ACTIVE SYSTEM SELECTION ---
SYSTEM_NAME = "graphene"
CFG = CONFIGS[SYSTEM_NAME]

@dataclass
class AlgorithmSettings:
    """Centralized hyperparameters for physics and math algorithms."""
    substrate_type: int = 3
    top_layer_tol: float = 0.5
    liquid_cutoff_frac: float = 0.10
    interface_frac: float = 0.50
    smooth_sigma: float = 1.5
    
    # Fit window parameters
    z_win_lo_flat: float = 4.6
    z_win_hi_flat: float = 10.8
    z_win_lo_pillars: float = 6.0
    z_win_hi_pillars: float = 18.0
    z_bin: float = 0.5
    min_pts: int = 5
    
    # Robust fitting
    min_r2_target: float = 0.92
    sigma_clip: float = 1.3
    max_clip_iter: int = 5
    window_step: float = 1.5
    max_win_shifts: int = 6

PARAMS = AlgorithmSettings()

# Color Palette for consistent visualization
PALETTE = {
    "left_tangent": "#C0392B",   # Deep Red
    "right_tangent": "#2471A3",  # Steel Blue
    "circle_fit": "#8E44AD",     # Purple (For the circle cross-check)
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
    """Converts 1D data to 2D meshes and calculates bulk properties."""

    @staticmethod
    def estimate_bulk_density(density: np.ndarray, z: np.ndarray, surface_z: float) -> float:
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
        # Zero below surface to prevent contouring algorithms from snapping to substrate
        cut_idx = np.searchsorted(z_grid, surface_z, side="right")
        safe_grid = rho_grid.copy()
        safe_grid[:cut_idx, :] = 0.0
        
        smooth = gaussian_filter(safe_grid, sigma=PARAMS.smooth_sigma)
        contours = find_contours(smooth, level=level)
        if not contours:
            raise ValueError(f"No contour found at density level {level:.4f}")
            
        # Get longest bounding-box contour
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
# 4. MATH LOGIC: TANGENTS & CIRCLES
# =============================================================================

class TangentFitter:
    """Original method: Local tangent calculation with robust sigma-clipping."""

    @staticmethod
    def _compress_side(xs: np.ndarray, zs: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
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
        m, b = np.polyfit(zs, xs, 1)
        res = xs - (m * zs + b)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((xs - xs.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return m, b, r2

    @staticmethod
    def _sigma_clip_fit(xs: np.ndarray, zs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
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

            angle = float(np.degrees(np.arctan2(1.0, abs(m))))
            if side == "left" and m < 0: angle = 180.0 - angle
            if side == "right" and m > 0: angle = 180.0 - angle

            x_contact = float(m * z_contact + b)

            result = {
                "xs": xs_f, "zs": zs_f, "m": float(m), "b": float(b), "r2": float(r2),
                "angle": angle, "x_contact": x_contact, "side": side, "z_win": (z_lo, z_hi)
            }

            if not is_flat: return result
            
            if best_result is None or r2 > best_result["r2"]:
                best_result = result
            if r2 >= PARAMS.min_r2_target:
                break
                
        return best_result


class SecondaryMath:
    """Alternative measurement heuristics (Geometric and Circle Fitting)."""
    
    @staticmethod
    def geometric_angle(lin_left: dict, lin_right: dict, z_contact: float, z_apex: float) -> Optional[float]:
        """theta = 2 * arctan(h/a) using tangent contact points as base width."""
        if not lin_left or not lin_right: return None
        a = 0.5 * (lin_right["x_contact"] - lin_left["x_contact"])
        h = z_apex - z_contact
        if a <= 0 or h <= 0: return None
        return float(np.degrees(2.0 * np.arctan(h / a)))

    @staticmethod
    def circle_fit_angle(x_cnt: np.ndarray, z_cnt: np.ndarray, surface_z: float) -> Optional[dict]:
        """Least squares fit of a circle to the contour, evaluating angle geometrically."""
        z_median = np.median(z_cnt)
        lower_mask = (z_cnt < z_median) & (z_cnt > surface_z + 0.5)
        x_fit, z_fit = x_cnt[lower_mask], z_cnt[lower_mask]

        if len(x_fit) < 5: return None

        def residuals(params, x, z):
            xc, zc, r = params
            return np.sqrt((x - xc)**2 + (z - zc)**2) - r

        xc_guess, zc_guess = np.mean(x_fit), np.max(z_fit)
        r_guess = (np.max(x_fit) - np.min(x_fit)) / 2.0
        
        res = least_squares(residuals, [xc_guess, zc_guess, r_guess], args=(x_fit, z_fit), 
                            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, np.inf]))
        
        xc, zc, R = res.x
        h_diff = max(-1.0, min(1.0, (zc - surface_z) / R))
        theta = 90.0 + math.degrees(math.asin(h_diff))
        
        return {"theta": theta, "xc": xc, "zc": zc, "R": R}


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
                  lin_left: dict, lin_right: dict, circle_data: dict,
                  z_contact: float, z_apex: float, avg_tangent: float, 
                  surface_z: float, out_path: Path, sys_label: str):
        Plotter._setup_style()
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")

        # 1. Background Density Heatmap
        masked_density = np.ma.masked_where(rho_grid < 0.05, rho_grid)
        levels = np.linspace(0.05, np.max(rho_grid), 15)
        cf = ax.contourf(x_grid, z_grid, masked_density, levels=levels, cmap="Blues", extend="max", alpha=0.85)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Density (g/cm$^3$)", rotation=270, labelpad=15)

        # 2. Gibbs dividing surface & Substrate
        ax.plot(x_cnt, z_cnt, "-", color=PALETTE["contour"], lw=1.8, label="Gibbs dividing surface", zorder=3)
        ax.axhline(surface_z, color=PALETTE["substrate"], ls=":", lw=1.5, zorder=2, label="Substrate plane")

        # 3. Tangent Lines (Primary Method)
        z_line = np.array([surface_z - 0.5, z_contact + PARAMS.z_win_hi_pillars + 3])
        for fit, color, label_text in [(lin_left, PALETTE["left_tangent"], "Left tangent"), 
                                       (lin_right, PALETTE["right_tangent"], "Right tangent")]:
            if not fit: continue
            x_full = fit["m"] * z_line + fit["b"]
            ax.plot(x_full, z_line, "--", color=color, lw=1.8, zorder=5, solid_capstyle="round", label=label_text)
            ax.plot(fit["x_contact"], surface_z, "o", ms=6, mfc="white", mec=color, mew=1.6, zorder=8)
            
            # Tangent Arcs
            t1, t2 = (90.0, 90.0 + (180.0 - fit["angle"])) if fit["side"] == "left" else (90.0 - (180.0 - fit["angle"]), 90.0)
            ax.add_patch(mpatches.Arc((fit["x_contact"], surface_z), width=12, height=12, 
                                      theta1=t1, theta2=t2, color=color, lw=1.4, zorder=9))

        # 4. Circle Fit (Secondary Check overlay)
        if circle_data:
            xc, zc, R = circle_data["xc"], circle_data["zc"], circle_data["R"]
            circ = plt.Circle((xc, zc), R, color=PALETTE["circle_fit"], fill=False, ls="-.", lw=1.2, alpha=0.8, label="Circle Fit check")
            ax.add_patch(circ)
            ax.plot(xc, zc, '+', color=PALETTE["circle_fit"], ms=6)

        # 5. Styling and Info Box
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ (Å)", fontsize=12)
        ax.set_ylabel("$z$ (Å)", fontsize=12)
        ax.set_xlim(x_cnt.min() - 15, x_cnt.max() + 15)
        ax.set_ylim(surface_z - 5, z_apex + 10)

        # Combine text summary
        box_text = f"System: {sys_label}\n"
        box_text += f"Tangents Angle = {avg_tangent:.1f}°\n"
        if circle_data: box_text += f"Circle Fit Angle = {circle_data['theta']:.1f}°"
        
        ax.text(0.04, 0.96, box_text, transform=ax.transAxes, fontsize=10, 
                fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9))

        # Prevent legend duplication
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="upper right")

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

    avg_tangent = 0.5 * (lin_left["angle"] + lin_right["angle"])
    print(f"PRIMARY (Tangents) -> Left: {lin_left['angle']:.1f}° | Right: {lin_right['angle']:.1f}°")
    print(f"MAIN RESULT (Tangent Avg): {avg_tangent:.2f}°")

    # Secondary checks
    theta_geom = SecondaryMath.geometric_angle(lin_left, lin_right, z_contact, z_apex)
    circle_data = SecondaryMath.circle_fit_angle(x_cnt, z_cnt, surface_z)
    if theta_geom: print(f"CHECK 1 (Geometric):       {theta_geom:.2f}°")
    if circle_data: print(f"CHECK 2 (Circle Fit):      {circle_data['theta']:.2f}°")

    # 4. Generate Graphics
    print("\n--- Generating Plots ---")
    hist_out = out_dir / f"histogram_{SYSTEM_NAME}.png"
    main_out = out_dir / f"contact_angle_{SYSTEM_NAME}.png"

    Plotter.plot_histogram(density, z, surface_z, rho_bulk, cutoff, rho_thr, hist_out)
    Plotter.plot_main(x_grid, z_grid, rho_grid, x_cnt, z_cnt, lin_left, lin_right, circle_data, 
                      z_contact, z_apex, avg_tangent, surface_z, main_out, CFG.name)
    
    print(f"Files saved to: {out_dir}")
    print("Success!\n")


if __name__ == "__main__":
    main()