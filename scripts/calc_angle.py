"""
contact_angle.py
================
Contact angle measurement for a cylindrical water droplet on a flat substrate,
computed from 2-D density fields produced by LAMMPS chunk/ave.

PRIMARY METHOD  — local tangent at the contact line
    The contour edge in a narrow z-window just above the surface is binned by
    height, compressed to one representative x per bin, and fitted with a
    straight line  x = m·z + b.  The contact angle follows from the slope:

        θ  =  arctan(Δz / Δx)  =  arctan(1 / |m|)

    corrected to the obtuse quadrant when the slope sign indicates an
    overhanging interface.

SECONDARY METHOD — geometric cross-check from droplet dimensions
    θ_geom = 2 · arctan(h / a)
    where h = apex height above the contact line, a = contact half-width.
    Used only as a sanity check; the tangent result is reported as the answer.

ROBUST FITTING (flat substrate only)
    For systems where the Gibbs surface curves near the three-phase contact
    line (wetting foot, pinning artefacts), a plain least-squares fit gives
    low R².  Two additional steps are applied:
        1. Iterative sigma-clipping — points that deviate more than SIGMA_CLIP
           standard deviations from the current fit line are removed; the line
           is then recomputed.  This converges in a few iterations.
        2. Automatic window elevation — if R² after clipping is still below
           MIN_R2_TARGET, the fit window is shifted upward by WINDOW_STEP
           increments (up to MAX_WIN_SHIFTS times) until a straighter portion
           of the interface is found.
    The pillars system is unaffected; it uses a single-pass plain fit.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours


# =============================================================================
# PUBLICATION STYLE — applied globally via rcParams
# =============================================================================
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":    "stix",
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.30,
    "grid.linestyle":      ":",
    "grid.linewidth":      0.5,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    "xtick.minor.size":    2,
    "ytick.minor.size":    2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon":      True,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "0.75",
    "legend.fontsize":     8,
    "figure.dpi":          150,
})


# =============================================================================
# SYSTEM SELECTION
# =============================================================================
SYSTEM = "flat"          # "flat"  →  flat graphene substrate
                         # "pillars" →  graphite with passivated pillars


# =============================================================================
# FILE PATHS  (adjust to match your directory layout)
# =============================================================================
if SYSTEM == "flat":
    DATA_FILE    = Path("../data/density_flat.dat")
    SYSTEM_FILE  = Path("../data/equilibrated_flat_system_v2.data")
    OUTPUT_FILE  = Path("../output/contact_angle_flat.png")
    SYSTEM_LABEL = "Flat graphene"
else:
    DATA_FILE    = Path("../data/density_pillars.dat")
    SYSTEM_FILE  = Path("../data/equilibrated_pillars_system_v2.data")
    OUTPUT_FILE  = Path("../output/contact_angle_pillars.png")
    SYSTEM_LABEL = "Graphite with pillars (passivated)"


# =============================================================================
# SUBSTRATE DETECTION
# =============================================================================
SUBSTRATE_TYPE     = 3      # atom-type index of substrate atoms in LAMMPS data
TOP_LAYER_TOL      = 0.5    # Å tolerance for identifying the topmost layer
SURFACE_Z_OVERRIDE = None   # set to a float (Å) to bypass auto-detection


# =============================================================================
# DENSITY / CONTOUR PARAMETERS
# =============================================================================
LIQUID_CUTOFF_FRACTION = 0.10   # fraction of peak density below which bins are
                                  # treated as vapour when estimating bulk density
INTERFACE_FRACTION     = 0.50   # Gibbs dividing surface: ½ ρ_bulk
SMOOTH_SIGMA           = 1.5    # Gaussian smoothing σ applied before contouring


# =============================================================================
# FIT WINDOW — z-range relative to the contact-line z
# =============================================================================
if SYSTEM == "flat":
    Z_WIN_LO = 4.6     # Å above z_contact where the fit window starts
    Z_WIN_HI = 10.8    # Å above z_contact where the fit window ends
else:
    Z_WIN_LO = 6.0
    Z_WIN_HI = 18.0

Z_BIN   = 0.5   # height of each compression bin (Å)
MIN_PTS = 5     # minimum number of compressed points required for a valid fit


# =============================================================================
# ROBUST FITTING PARAMETERS  (flat substrate only)
# =============================================================================
MIN_R2_TARGET  = 0.92   # target R²; fitting stops once this is reached
SIGMA_CLIP     = 1.3    # outlier threshold in units of residual std
MAX_CLIP_ITER  = 5      # maximum sigma-clipping iterations per window
WINDOW_STEP    = 1.5    # Å to shift the fit window upward per attempt
MAX_WIN_SHIFTS = 6      # maximum number of upward window shifts


# =============================================================================
# COLOUR PALETTE — restrained, publication-friendly
# =============================================================================
C_LEFT    = "#C0392B"   # deep red   — left  tangent
C_RIGHT   = "#2471A3"   # steel blue — right tangent
C_CONTOUR = "#1C2833"   # near-black — Gibbs dividing surface
C_LIQUID  = "#AED6F1"   # pale blue  — liquid density bins (scatter)

C_ALL     = "#BFC9CA"   # light grey — all bins in the density histogram
C_LIQ_H   = "#2471A3"   # steel blue — liquid bins in the density histogram
C_BULK    = "#922B21"   # dark red   — bulk density reference line
C_GIBBS   = "#1A5276"   # dark blue  — Gibbs threshold line


# =============================================================================
# I/O HELPERS
# =============================================================================

def load_density(path: Path):
    """
    Read a LAMMPS chunk/ave density file.

    Expected format (whitespace-separated, lines with 5 columns):
        chunk_id  x_coord  z_coord  Ncount  density[g/cm3]

    Lines beginning with '#' or having != 5 columns are skipped.

    Returns
    -------
    x, z, density : np.ndarray (1-D, same length)
    """
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No valid data rows found in {path}")
    arr = np.array(rows, dtype=float)
    return arr[:, 1], arr[:, 2], arr[:, 4]


def load_surface_z(path: Path, stype: int, tol: float) -> float:
    """
    Determine the z-coordinate of the topmost substrate layer from a LAMMPS
    data file.

    Reads the 'Atoms' section and collects z-coordinates of all atoms whose
    type matches `stype`.  The surface z is defined as the median of all atoms
    within `tol` Å of the absolute maximum z, which makes the estimate
    insensitive to a few outlier atoms.

    Returns
    -------
    surface_z : float (Å)
    """
    in_atoms = False
    zvals = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("Atoms"):
                in_atoms = True
                continue
            if not in_atoms or not line or line.startswith("#"):
                continue
            if any(line.startswith(s) for s in
                   ("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers")):
                break
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                if int(parts[2]) == stype:
                    zvals.append(float(parts[6]))
            except ValueError:
                continue
    if not zvals:
        raise ValueError(f"No substrate atoms (type {stype}) found in {path}")
    zvals    = np.array(zvals, dtype=float)
    surface_z = float(np.median(zvals[zvals >= zvals.max() - tol]))
    print(f"[surface]  z_max = {zvals.max():.4f} Å   surface_z = {surface_z:.4f} Å")
    return surface_z


# =============================================================================
# BULK DENSITY ESTIMATION
# =============================================================================

def estimate_bulk_density(density, x, z, surface_z):
    """
    Estimate the bulk liquid density from the density field.

    All non-zero bins above the substrate are considered.  Bins below
    LIQUID_CUTOFF_FRACTION × max are classified as vapour/interface and
    excluded.  The bulk density is the median of the remaining 'liquid' bins,
    which is more robust than the mean against the density peak near the
    substrate.

    Raises a warning when the result lies outside the physically expected
    range 0.5–1.4 g/cm³ for water.

    Returns
    -------
    rho_bulk : float (g/cm³)
    """
    above       = (z > surface_z) & (density > 0.0)
    rho_above   = density[above]
    if rho_above.size == 0:
        raise ValueError("No non-zero density bins found above surface_z.")
    cutoff      = LIQUID_CUTOFF_FRACTION * rho_above.max()
    liquid_bins = rho_above[rho_above > cutoff]
    if liquid_bins.size == 0:
        raise ValueError(f"No bins above the liquid cutoff ({cutoff:.4f} g/cm³).")
    rho_bulk = float(np.median(liquid_bins))
    print(f"[density]  max={rho_above.max():.4f}  cutoff={cutoff:.4f}  "
          f"liquid bins={liquid_bins.size}  bulk={rho_bulk:.4f} g/cm³")
    if rho_bulk < 0.5 or rho_bulk > 1.4:
        print(f"[WARNING]  bulk density = {rho_bulk:.4f} g/cm³ is outside the "
              f"expected 0.5–1.4 g/cm³ range.")
    return rho_bulk


# =============================================================================
# DENSITY GRID AND GIBBS DIVIDING SURFACE
# =============================================================================

def make_grid(x, z, density):
    """
    Reshape the flat (x, z, density) arrays into a 2-D grid.

    Returns
    -------
    x_grid  : 1-D array of unique x-values (Å)
    z_grid  : 1-D array of unique z-values (Å)
    rho_grid: 2-D array, shape (z_grid.size, x_grid.size)
    """
    xu, zu = np.unique(x), np.unique(z)
    xi = {v: i for i, v in enumerate(xu)}
    zi = {v: i for i, v in enumerate(zu)}
    grid = np.zeros((zu.size, xu.size), dtype=float)
    for xv, zv, rv in zip(x, z, density):
        grid[zi[zv], xi[xv]] = rv
    return xu, zu, grid


def zero_below_surface(grid, z_grid, surface_z):
    """
    Zero out all grid rows at or below the substrate surface so that the
    contouring algorithm never locks onto the substrate itself.
    """
    cut = np.searchsorted(z_grid, surface_z, side="right")
    grid = grid.copy()
    grid[:cut, :] = 0.0
    return grid


def get_contour(x_grid, z_grid, rho_grid, level):
    """
    Extract the Gibbs dividing surface at density = `level`.

    The grid is first Gaussian-smoothed (σ = SMOOTH_SIGMA) to remove
    numerical noise before contouring.  If multiple closed contours are
    found (e.g. satellite droplets or vapour pockets), the one with the
    largest bounding-box area — x-span × z-span — is selected as the
    main droplet interface.

    Returns
    -------
    x_cnt, z_cnt : 1-D arrays of contour coordinates (Å)
    """
    smooth   = gaussian_filter(rho_grid, sigma=SMOOTH_SIGMA)
    contours = find_contours(smooth, level=level)
    if not contours:
        raise ValueError(f"No contour found at level {level:.4f} g/cm³.")
    best, best_score = None, -np.inf
    for c in contours:
        zp = np.interp(c[:, 0], np.arange(z_grid.size), z_grid)
        xp = np.interp(c[:, 1], np.arange(x_grid.size), x_grid)
        score = (xp.max() - xp.min()) * (zp.max() - zp.min())
        if score > best_score:
            best_score = score
            best = (xp, zp)
    return best


# =============================================================================
# CONTOUR COMPRESSION
# =============================================================================

def compress_side(xs, zs, side):
    """
    Reduce contour scatter to one representative point per height bin.

    Within each bin of height Z_BIN, the extreme x-value is kept:
        left  side → minimum x  (leftmost edge of the interface)
        right side → maximum x  (rightmost edge)
    The z-coordinate of each bin is the mean of all points in that bin.

    This compression removes the multiple contour points per height that arise
    from contouring a smoothed grid and makes the subsequent linear fit stable.

    Returns
    -------
    xs_out, zs_out : compressed arrays, one point per bin
    """
    z0   = zs.min()
    bidx = np.floor((zs - z0) / Z_BIN).astype(int)
    xs_out, zs_out = [], []
    for b in np.unique(bidx):
        m = bidx == b
        xs_out.append(xs[m].min() if side == "left" else xs[m].max())
        zs_out.append(zs[m].mean())
    return np.array(xs_out), np.array(zs_out)


# =============================================================================
# PLAIN LINEAR FIT HELPER
# =============================================================================

def _polyfit_r2(xs, zs):
    """
    Fit the model  x = m·z + b  by ordinary least squares.

    Note the axis convention: z is the independent variable because the
    interface is nearly vertical near the contact line (large Δz, small Δx).
    Fitting x(z) avoids division-by-zero in near-vertical segments.

    Returns
    -------
    m, b : float   slope and intercept
    r2   : float   coefficient of determination
    """
    m, b   = np.polyfit(zs, xs, 1)
    res    = xs - (m * zs + b)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((xs - xs.mean())**2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2


# =============================================================================
# ITERATIVE SIGMA-CLIPPING FIT  (flat substrate only)
# =============================================================================

def _sigma_clip_fit(xs, zs):
    """
    Fit a line to (zs, xs) with iterative sigma-clipping.

    Algorithm
    ---------
    1. Fit the full data set.
    2. Compute residuals; remove points with |residual| > SIGMA_CLIP · σ.
    3. Refit the surviving points.
    4. Repeat until no new points are removed or MAX_CLIP_ITER is reached.

    This is effective when a small fraction of contour points belong to the
    wetting foot (curved region near the substrate) and pull the regression
    line away from the true linear interface higher up.

    Returns
    -------
    xs_clean, zs_clean : points surviving all clipping rounds
    m, b, r2           : final fit parameters
    """
    m, b, r2 = _polyfit_r2(xs, zs)
    mask     = np.ones(len(xs), dtype=bool)

    for _ in range(MAX_CLIP_ITER):
        residuals = xs - (m * zs + b)
        sigma     = residuals.std()
        if sigma < 1e-9:
            break                                     # perfectly collinear
        new_mask = np.abs(residuals) < SIGMA_CLIP * sigma
        if new_mask.sum() < MIN_PTS:
            break                                     # would remove too many
        if np.array_equal(new_mask, mask):
            break                                     # converged
        mask    = new_mask
        m, b, r2 = _polyfit_r2(xs[mask], zs[mask])

    return xs[mask], zs[mask], m, b, r2


# =============================================================================
# MAIN TANGENT FITTING FUNCTION
# =============================================================================

def fit_tangent(x_cnt, z_cnt, z_contact, side):
    """
    Fit a tangent line to one side of the Gibbs dividing surface and return
    the contact angle at the substrate.

    The fitting strategy depends on the system:

    Pillars system
        Single-pass ordinary least squares in the fixed window
        [z_contact + Z_WIN_LO, z_contact + Z_WIN_HI].

    Flat system
        Step 1 — sigma-clipping fit in the base window.
        Step 2 — if R² < MIN_R2_TARGET, shift the window upward by
                  WINDOW_STEP Å and repeat, up to MAX_WIN_SHIFTS times.
        The window with the highest R² is returned; if MIN_R2_TARGET is
        reached early, iteration stops immediately.

    Contact angle convention
    ------------------------
    The slope m = dx/dz of the fit line gives the interface orientation:

        angle = arctan(1 / |m|)   →  angle ∈ (0°, 90°) for acute interfaces

    When the slope sign implies an overhanging interface the supplementary
    angle (180° − angle) is returned instead, covering the obtuse case
    (90° < θ < 180°) that occurs for hydrophobic substrates.

    Parameters
    ----------
    x_cnt, z_cnt : contour coordinate arrays (Å)
    z_contact    : z of the contact line (lowest contour point, Å)
    side         : "left" or "right"

    Returns
    -------
    dict with keys: xs, zs, m, b, r2, angle, x_contact, side, z_win, win_shift
    None if fewer than MIN_PTS compressed points are available.
    """
    x_mid = np.median(x_cnt)
    z_lo0 = z_contact + Z_WIN_LO
    z_hi0 = z_contact + Z_WIN_HI

    best_result = None
    n_shifts    = MAX_WIN_SHIFTS if SYSTEM == "flat" else 1

    for shift in range(n_shifts):
        dz   = shift * WINDOW_STEP
        z_lo = z_lo0 + dz
        z_hi = z_hi0 + dz

        # Select contour points on the correct side within the current window
        mask  = (z_cnt >= z_lo) & (z_cnt <= z_hi)
        mask &= (x_cnt < x_mid) if side == "left" else (x_cnt > x_mid)
        xs_raw, zs_raw = x_cnt[mask], z_cnt[mask]
        if xs_raw.size < MIN_PTS:
            continue

        # Compress to one representative x per height bin
        xs_c, zs_c = compress_side(xs_raw, zs_raw, side)
        if xs_c.size < MIN_PTS:
            continue

        # Choose fitting method based on system
        if SYSTEM == "flat":
            xs_f, zs_f, m, b, r2 = _sigma_clip_fit(xs_c, zs_c)
        else:
            m, b, r2 = _polyfit_r2(xs_c, zs_c)
            xs_f, zs_f = xs_c, zs_c

        # Derive the contact angle from the slope
        angle = float(np.degrees(np.arctan2(1.0, abs(m))))
        if side == "left"  and m < 0:
            angle = 180.0 - angle   # obtuse interface opening to the right
        if side == "right" and m > 0:
            angle = 180.0 - angle   # obtuse interface opening to the left

        # Extrapolate the fit line to the contact-line z
        x_contact = float(m * z_contact + b)

        result = {
            "xs":        xs_f,
            "zs":        zs_f,
            "m":         float(m),
            "b":         float(b),
            "r2":        float(r2),
            "angle":     angle,
            "x_contact": x_contact,
            "side":      side,
            "z_win":     (z_lo, z_hi),
            "win_shift": shift,
        }

        # Pillars: return immediately after the single attempt
        if SYSTEM != "flat":
            return result

        # Flat: track the best result so far
        if best_result is None or r2 > best_result["r2"]:
            best_result = result

        if r2 >= MIN_R2_TARGET:
            print(f"[tangent]  {side}: R²={r2:.4f} ≥ {MIN_R2_TARGET} "
                  f"(shift={shift}, window=[{z_lo:.1f}, {z_hi:.1f}] Å)")
            break
        else:
            print(f"[tangent]  {side}: R²={r2:.4f} < {MIN_R2_TARGET} "
                  f"(shift={shift}, window=[{z_lo:.1f}, {z_hi:.1f}] Å) — "
                  f"shifting window upward...")

    if best_result is not None and SYSTEM == "flat":
        r = best_result
        print(f"[tangent]  {side} FINAL: R²={r['r2']:.4f}  "
              f"θ={r['angle']:.2f}°  "
              f"window=[{r['z_win'][0]:.1f}, {r['z_win'][1]:.1f}] Å")

    return best_result


# =============================================================================
# GEOMETRIC CROSS-CHECK
# =============================================================================

def geometric_angle(lin_left, lin_right, z_contact, z_apex):
    """
    Estimate the contact angle from the overall droplet dimensions.

    For a circular-cap cross-section:
        θ = 2 · arctan(h / a)
    where h = apex height above the contact line (Å)
          a = half the contact-line width (Å)

    This is independent of the tangent-fitting procedure and serves as a
    sanity check.  A large discrepancy (> 10°) indicates that the droplet
    profile deviates significantly from a circular cap.

    Returns (theta_geom, h, a) or (None, None, None) on failure.
    """
    if lin_left is None or lin_right is None:
        return None, None, None
    a = 0.5 * (lin_right["x_contact"] - lin_left["x_contact"])
    h = z_apex - z_contact
    if a <= 0 or h <= 0:
        return None, None, None
    theta = float(np.degrees(2.0 * np.arctan(h / a)))
    return theta, h, a


# =============================================================================
# DENSITY HISTOGRAM — diagnostic figure
# =============================================================================

def save_histogram(density, x, z, surface_z, rho_bulk, rho_thr, output_dir: Path):
    """
    Save a publication-quality histogram of the density distribution.

    Three overlapping elements are drawn:
      · All non-zero bins above the substrate (grey) — full distribution
      · Liquid bins above the vapour cutoff (blue) — bins used for bulk estimate
      · Reference lines for ρ_bulk and the Gibbs threshold ½ρ_bulk
      · A shaded vapour region below the cutoff

    The figure is saved to output_dir/density_histogram_{SYSTEM}.png.
    """
    above  = (z > surface_z) & (density > 0.0)
    rho_a  = density[above]
    cutoff = LIQUID_CUTOFF_FRACTION * rho_a.max()

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # All bins — background grey
    ax.hist(rho_a, bins=200, color=C_ALL, alpha=0.9, linewidth=0,
            label="All bins above substrate", zorder=1)
    # Liquid bins — coloured overlay
    ax.hist(rho_a[rho_a > cutoff], bins=200, color=C_LIQ_H, alpha=0.75,
            linewidth=0,
            label=f"Liquid bins ($\\rho > {cutoff:.3f}$ g cm$^{{-3}}$)",
            zorder=2)

    # Vertical reference lines
    ax.axvline(rho_bulk, color=C_BULK, lw=1.6, zorder=5,
               label=f"$\\rho_{{\\rm bulk}} = {rho_bulk:.3f}$ g cm$^{{-3}}$")
    ax.axvline(rho_thr, color=C_GIBBS, lw=1.6, ls="--", zorder=5,
               label=(f"Gibbs threshold = {rho_thr:.3f} g cm$^{{-3}}$"
                      f"  ($\\frac{{1}}{{2}}\\rho_{{\\rm bulk}}$)"))

    # Vapour region shading
    ax.axvspan(0, cutoff, color="#D5E8F7", alpha=0.45, zorder=0,
               label=f"Vapour / interface ($\\rho < {cutoff:.3f}$ g cm$^{{-3}}$)")

    x_max = min(rho_a.max() * 1.05, 1.85)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Density, $\\rho$ (g cm$^{-3}$)", fontsize=10)
    ax.set_ylabel("Bin count",                        fontsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.set_title(f"Density distribution — {SYSTEM_LABEL}",
                 fontsize=10, fontweight="bold", pad=8)

    plt.tight_layout()
    out = output_dir / f"density_histogram_{SYSTEM}.png"
    plt.savefig(out, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"[saved]    {out}")


# =============================================================================
# MAIN CONTACT ANGLE FIGURE
# =============================================================================

def plot_main(x_liq, z_liq, x_cnt, z_cnt,
              lin_left, lin_right, z_contact, z_apex,
              avg_tangent, theta_geom,
              surface_z, output_file: Path):
    """
    Produce the main contact-angle figure with:
      · Liquid density bin scatter (rasterised for speed)
      · Gibbs dividing surface contour
      · Substrate and fit-window reference lines
      · Tangent lines, fit-point markers, angle arcs and annotations
      · Central summary box (mean angle + geometric check)
    """
    fig, ax = plt.subplots(figsize=(6.5, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Liquid density bins — rasterised scatter ──────────────────────────────
    step = max(1, x_liq.size // 30000)
    ax.scatter(x_liq[::step], z_liq[::step],
               s=3, alpha=0.22, color=C_LIQUID, linewidths=0,
               label="Liquid density bins", zorder=1, rasterized=True)

    # ── Gibbs dividing surface ────────────────────────────────────────────────
    ax.plot(x_cnt, z_cnt, "-", color=C_CONTOUR, lw=1.4,
            label="Gibbs dividing surface ($\\frac{1}{2}\\rho_{\\rm bulk}$)",
            zorder=3)

    # ── Substrate plane ───────────────────────────────────────────────────────
    ax.axhline(surface_z, color="#7F8C8D", ls=":", lw=1.2, zorder=2,
               label=f"Substrate surface  $z_s = {surface_z:.2f}$ Å")

    # ── Fit-window bounds — one horizontal pair per tangent ───────────────────
    # Left and right may have different windows after auto-shifting, so they
    # are drawn separately in matching colours at reduced opacity.
    win_labels_added = set()
    for fit, color in [(lin_left, C_LEFT), (lin_right, C_RIGHT)]:
        if fit is None:
            continue
        z_lo, z_hi = fit["z_win"]
        win_key = (round(z_lo, 2), round(z_hi, 2))
        for zv in [z_lo, z_hi]:
            ax.axhline(zv, color=color, ls="--", lw=0.65, alpha=0.55, zorder=2)
        if win_key not in win_labels_added:
            ax.plot([], [], color=color, ls="--", lw=0.8,
                    label=f"Fit window ({fit['side']}): [{z_lo:.1f}, {z_hi:.1f}] Å")
            win_labels_added.add(win_key)

    # ── Tangent lines, markers and angle annotations ──────────────────────────
    z_line = np.array([surface_z - 0.5, z_contact + Z_WIN_HI + 3])

    for fit, color in [(lin_left, C_LEFT), (lin_right, C_RIGHT)]:
        if fit is None:
            continue
        side = fit["side"]

        # Compressed fit points
        ax.scatter(fit["xs"], fit["zs"],
                   s=14, color=color, zorder=6, linewidths=0, alpha=0.85)

        # Tangent line extrapolated through the full z_line range
        x_full = fit["m"] * z_line + fit["b"]
        ax.plot(x_full, z_line, "-", color=color, lw=1.8,
                zorder=5, solid_capstyle="round")

        # Contact-point marker (open circle at the substrate)
        ax.plot(fit["x_contact"], surface_z,
                "o", ms=5, mfc="white", mec=color, mew=1.6, zorder=8)

        # Angle arc (radius 6 Å, centred on the contact point)
        arc_r = 6.0
        if side == "left":
            t1, t2 = 90.0, 90.0 + (180.0 - fit["angle"])
        else:
            t1, t2 = 90.0 - (180.0 - fit["angle"]), 90.0
        arc = mpatches.Arc(
            (fit["x_contact"], surface_z),
            width=arc_r * 2, height=arc_r * 2,
            theta1=t1, theta2=t2,
            color=color, lw=1.4, zorder=9
        )
        ax.add_patch(arc)

        # Annotation box: θ and R²
        z_ann = z_contact + 0.5 * (Z_WIN_LO + Z_WIN_HI)
        x_ann = fit["m"] * z_ann + fit["b"]
        ha    = "right" if side == "left" else "left"
        dx    = -4.5    if side == "left" else 4.5
        label = (f"$\\theta_{{\\rm {side}}} = {fit['angle']:.1f}^\\circ$\n"
                 f"$R^2 = {fit['r2']:.3f}$")
        ax.annotate(
            label,
            xy=(x_ann, z_ann), xytext=(x_ann + dx, z_ann),
            fontsize=8.5, color=color, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec=color, lw=1.0, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8)
        )

    # ── Central result summary box ────────────────────────────────────────────
    x_center = 0.5 * (x_cnt.min() + x_cnt.max())
    z_center = z_contact + 0.6 * (z_apex - z_contact)
    geom_str = (f"\nGeom. check: $\\theta = {theta_geom:.1f}^\\circ$"
                if theta_geom is not None else "")
    ax.text(x_center, z_center,
            f"$\\bar{{\\theta}} = {avg_tangent:.1f}^\\circ${geom_str}",
            fontsize=12, fontweight="bold",
            ha="center", va="center", color="#1A252F",
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="white", ec="#1A252F", lw=1.2, alpha=0.90),
            zorder=10)

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("$x$ (Å)", fontsize=10)
    ax.set_ylabel("$z$ (Å)", fontsize=10)
    ax.set_xlim(x_cnt.min() - 10, x_cnt.max() + 10)
    ax.set_ylim(surface_z - 4,    z_apex + 8)
    ax.set_aspect("equal")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#CCCCCC", linewidth=0.55,
            linestyle=":", zorder=0)
    ax.grid(True, which="minor", color="#E8E8E8", linewidth=0.30,
            linestyle=":", zorder=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Contact angle — {SYSTEM_LABEL}\n"
        f"Local tangent method,  $\\theta = {avg_tangent:.1f}^\\circ$",
        fontsize=10, fontweight="bold", pad=8
    )

    # ── Legend (compact, upper right) ─────────────────────────────────────────
    left_p  = mpatches.Patch(
        color=C_LEFT,
        label=f"Left tangent  $\\theta = {lin_left['angle']:.1f}^\\circ$"
    )
    right_p = mpatches.Patch(
        color=C_RIGHT,
        label=f"Right tangent  $\\theta = {lin_right['angle']:.1f}^\\circ$"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [left_p, right_p],
              fontsize=7.5, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_file, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"[saved]    {output_file}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    # ── Load raw density data ─────────────────────────────────────────────────
    x, z, density = load_density(DATA_FILE)
    print(f"[data]     pts={density.size}  "
          f"x=[{x.min():.1f}, {x.max():.1f}]  "
          f"z=[{z.min():.1f}, {z.max():.1f}]  "
          f"ρ=[{density.min():.4f}, {density.max():.4f}]")

    # ── Locate the substrate surface ──────────────────────────────────────────
    surface_z = (float(SURFACE_Z_OVERRIDE) if SURFACE_Z_OVERRIDE is not None
                 else load_surface_z(SYSTEM_FILE, SUBSTRATE_TYPE, TOP_LAYER_TOL))

    # ── Bulk density and Gibbs threshold ──────────────────────────────────────
    rho_bulk = estimate_bulk_density(density, x, z, surface_z)
    rho_thr  = INTERFACE_FRACTION * rho_bulk
    print(f"[density]  Gibbs threshold = {INTERFACE_FRACTION:.0%} × "
          f"{rho_bulk:.4f} = {rho_thr:.4f} g/cm³")

    # ── Diagnostic density histogram ──────────────────────────────────────────
    save_histogram(density, x, z, surface_z, rho_bulk, rho_thr,
                   OUTPUT_FILE.parent)

    # ── Build density grid and extract Gibbs surface ──────────────────────────
    x_grid, z_grid, rho_grid = make_grid(x, z, density)
    rho_grid = zero_below_surface(rho_grid, z_grid, surface_z)
    x_cnt, z_cnt = get_contour(x_grid, z_grid, rho_grid, rho_thr)
    print(f"[contour]  x=[{x_cnt.min():.1f}, {x_cnt.max():.1f}]  "
          f"z=[{z_cnt.min():.1f}, {z_cnt.max():.1f}]  pts={x_cnt.size}")

    z_span = z_cnt.max() - z_cnt.min()
    if z_span < 10.0:
        print(f"[WARNING]  Contour z-span = {z_span:.1f} Å — droplet is very flat.")

    # ── Contact-line geometry ─────────────────────────────────────────────────
    z_contact = float(z_cnt.min())
    z_apex    = float(z_cnt.max())
    print(f"[geometry] z_contact={z_contact:.2f} Å  z_apex={z_apex:.2f} Å  "
          f"height={z_apex - z_contact:.2f} Å")
    print(f"[window]   base fit window = "
          f"[{z_contact + Z_WIN_LO:.2f}, {z_contact + Z_WIN_HI:.2f}] Å")

    # ── Tangent fitting ───────────────────────────────────────────────────────
    lin_left  = fit_tangent(x_cnt, z_cnt, z_contact, "left")
    lin_right = fit_tangent(x_cnt, z_cnt, z_contact, "right")

    if lin_left is None or lin_right is None:
        print("[ERROR]  Tangent fit failed — too few contour points in window.\n"
              "         Try adjusting Z_WIN_LO / Z_WIN_HI.")
        return

    avg_tangent = 0.5 * (lin_left["angle"] + lin_right["angle"])
    theta_geom, h_geom, a_geom = geometric_angle(
        lin_left, lin_right, z_contact, z_apex
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"System              : {SYSTEM_LABEL}")
    print(f"surface_z           : {surface_z:.4f} Å")
    print(f"z_contact (contour) : {z_contact:.4f} Å")
    print(f"Tangent LEFT        : {lin_left['angle']:.2f}°  "
          f"R² = {lin_left['r2']:.4f}  "
          f"window = [{lin_left['z_win'][0]:.1f}, {lin_left['z_win'][1]:.1f}] Å")
    print(f"Tangent RIGHT       : {lin_right['angle']:.2f}°  "
          f"R² = {lin_right['r2']:.4f}  "
          f"window = [{lin_right['z_win'][0]:.1f}, {lin_right['z_win'][1]:.1f}] Å")
    print(f"Tangent average     : {avg_tangent:.2f}°   ← MAIN RESULT")
    if theta_geom is not None:
        print(f"Geometric check     : {theta_geom:.2f}°  "
              f"(h = {h_geom:.1f} Å,  a = {a_geom:.1f} Å)")
        diff = abs(avg_tangent - theta_geom)
        if diff > 10:
            print(f"[NOTE]  Tangent vs geometric differ by {diff:.1f}° — "
                  "droplet profile deviates from a circular cap.")
    print("=" * 60)

    if lin_left["r2"] < 0.85 or lin_right["r2"] < 0.85:
        print("[WARNING]  R² < 0.85 on at least one side.  "
              "Consider narrowing Z_WIN_HI or reducing SIGMA_CLIP.")

    # ── Main figure ───────────────────────────────────────────────────────────
    liq = density > rho_thr
    plot_main(x[liq], z[liq], x_cnt, z_cnt,
              lin_left, lin_right, z_contact, z_apex,
              avg_tangent, theta_geom,
              surface_z, OUTPUT_FILE)


if __name__ == "__main__":
    main()