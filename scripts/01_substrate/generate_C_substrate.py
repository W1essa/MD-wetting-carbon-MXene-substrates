#!/usr/bin/env python3
import os
import numpy as np
from scipy.spatial import cKDTree
from ase import Atoms
from ase.build import graphene

# =====================================================================
# --- CONFIGURATION PARAMETERS ---
# =====================================================================
# Choose what structure to generate:
#   'graphene'            — single monolayer graphene sheet (flat finite flake)
#   'graphite'            — multi-layer graphite slab, no carving
#   'structured_graphite' — graphite slab carved into cylindrical pillars
STRUCTURE_TYPE = 'graphene'

# Set to True to hydrogen-passivate open edges for each structure type.
# Passivation is recommended whenever atoms are free to move (MD relaxation
# or finite-temperature runs), even if the structure is later frozen —
# open edges produce force artifacts in AIREBO/ReaxFF.
PASSIVATE_GRAPHENE            = True
PASSIVATE_GRAPHITE            = True
PASSIVATE_STRUCTURED_GRAPHITE = True

TARGET_L        = 400.0  # A  — target lateral (X, Y) size of the sheet
PILLAR_PERIOD   = 10.0   # A  — centre-to-centre distance between pillars
PILLAR_RADIUS   = 4.0    # A  — radius of each pillar cylinder
PILLAR_HEIGHT   = 10.0   # A  — slab height used for structured_graphite
GRAPHITE_HEIGHT = 10.0   # A  — slab height used for plain graphite
C_C_BOND        = 1.42   # A  — C-C bond length in graphene/graphite
INTERLAYER_DIST = 3.35   # A  — interlayer spacing in graphite (experimental)

# Passivation geometry constants
C_C_MIN = 1.30  # A  — lower bound for a valid C-C bond
C_C_MAX = 1.60  # A  — upper bound for a valid C-C bond
C_H     = 1.09  # A  — C-H bond length
MIN_SEP = 0.85  # A  — minimum allowed distance after placing a hydrogen

# =====================================================================
# --- GENERATION FUNCTIONS ---
# =====================================================================

def create_graphene_monolayer():
    """
    Build a rectangular monolayer graphene flake of size ~TARGET_L x TARGET_L.

    ASE's graphene() builder produces a periodic sheet; we over-tile it
    (nx x ny unit cells) and then trim stray atoms on the left/right edges
    that fall outside the rectangular bounding box, leaving a clean flake.
    """
    a_graphene = C_C_BOND * np.sqrt(3)          # lattice constant ~2.46 A
    nx = int(TARGET_L / a_graphene) + 5          # +5: safety margin before trim
    ny = int(TARGET_L / a_graphene) + 5

    monolayer = graphene(size=(nx, ny, 1), vacuum=50.0)
    monolayer.center(axis=(0, 1), vacuum=0.0)    # remove lateral vacuum

    positions = monolayer.get_positions()
    x, y, z   = positions.T
    tol        = 1e-3

    # Determine the rectangular bounding box from the bottom and top rows
    x_min = x[np.abs(y - y.min()) < tol].min()
    x_max = x[np.abs(y - y.max()) < tol].max()

    # Drop atoms that stick out beyond the box on the left or right
    valid_mask = ~((x < x_min - tol) | (x > x_max + tol))
    return monolayer[valid_mask]


def stack_graphite_layers(monolayer, slab_height, interlayer_dist, bond_length):
    """
    Stack copies of a graphene monolayer into an AB-stacked graphite slab.

    Layer A sits at z = 0 (the input monolayer).
    Layer B (odd indices) is shifted by (a/3, 0) in XY — the standard
    crystallographic AB shift that places one B-sublattice atom above the
    centre of each A hexagon.  The shift equals bond_length / sqrt(3) = a/3.
    Even-indexed layers (next A) get no XY shift, recreating the ABA... sequence.
    """
    a_lattice    = bond_length * np.sqrt(3)      # graphene lattice constant
    ab_shift_x   = a_lattice / 3.0               # correct AB shift ~0.82 A
    # ab_shift_y stays 0 — shift is purely along the armchair (x) direction

    graphite_slab = monolayer.copy()
    layers_needed = int(np.ceil(slab_height / interlayer_dist)) + 1

    for layer_num in range(1, layers_needed):
        new_layer = monolayer.copy()
        if layer_num % 2 == 1:                   # odd layer → B layer
            new_layer.positions[:, 0] += ab_shift_x
        new_layer.positions[:, 2] += layer_num * interlayer_dist
        graphite_slab.extend(new_layer)

    return graphite_slab


def generate_pillar_centers(x_range, y_range, radius, period):
    """
    Return a list of (x, y) pillar centre coordinates distributed on a
    regular grid with the given period, centred inside the slab footprint,
    and with at least one radius of clearance from each edge.
    """
    def centered_coords(lo, hi, p, r):
        # Available span after keeping one radius margin on each side
        span = hi - lo - 2.0 * r
        if span < 0:
            return np.zeros(0)
        n      = int(span / p) + 1
        center = (lo + hi) / 2.0
        return center + (np.arange(n) - (n - 1) / 2.0) * p

    x_centers = centered_coords(*x_range, period, radius)
    y_centers  = centered_coords(*y_range, period, radius)
    return [(x, y) for x in x_centers for y in y_centers]


def carve_pillars_from_graphite(graphite_slab, radius, period, base_tol=0.1):
    """
    Carve a graphite slab into a pillared surface:
      - the bottom layer (z ≈ z_min) is kept entirely as the base plate;
      - from all upper layers, only atoms inside pillar cylinders are kept.

    Cylinder test is done in XY only (infinite cylinders along Z),
    using vectorised squared-distance comparisons for efficiency.
    """
    positions = graphite_slab.get_positions()
    z_min     = positions[:, 2].min()

    # Identify the flat base layer
    base_mask      = np.isclose(positions[:, 2], z_min, atol=base_tol)
    base_positions = positions[base_mask]

    x_range = (base_positions[:, 0].min(), base_positions[:, 0].max())
    y_range = (base_positions[:, 1].min(), base_positions[:, 1].max())

    pillar_centers = generate_pillar_centers(x_range, y_range, radius, period)
    if not pillar_centers:
        return graphite_slab[base_mask]

    # Vectorised XY distance: shape (n_atoms, n_pillars)
    centers_arr = np.array(pillar_centers)
    xy          = positions[:, np.newaxis, :2]
    dist_sq     = np.sum((xy - centers_arr[np.newaxis, :, :])**2, axis=2)

    # An atom survives if it is inside at least one pillar cylinder
    pillar_mask = np.any(dist_sq <= radius**2, axis=1)
    return graphite_slab[base_mask | pillar_mask]


# =====================================================================
# --- PASSIVATION & CLEANUP FUNCTIONS ---
# =====================================================================

def clean_dangling_carbons(atoms, rmin=C_C_MIN, rmax=C_C_MAX):
    """
    Iteratively remove carbon atoms with fewer than 2 C-C bonds.

    After carving, some carbons may be left with only one neighbour or none
    (dangling atoms / isolated fragments).  Such atoms cause unphysical
    forces in bond-order potentials.  We repeat the pruning loop until no
    more atoms are removed.
    """
    clean = atoms.copy()
    while True:
        pos  = clean.get_positions()
        tree = cKDTree(pos)
        keep = [
            i for i, p in enumerate(pos)
            if sum(
                1 for j in tree.query_ball_point(p, rmax)
                if j != i and rmin <= np.linalg.norm(p - pos[j]) <= rmax
            ) >= 2
        ]
        if len(keep) == len(pos):
            break
        clean = clean[keep]
    return clean


def make_geo(atoms):
    """Return (positions, symbols, cKDTree) — a lightweight geometry bundle."""
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    return pos, sym, cKDTree(pos)


def classify_carbon_sites(geo, rmin=C_C_MIN, rmax=C_C_MAX):
    """
    Classify under-coordinated (edge) carbon atoms by bond count.

    Returns:
        edges2 — list of (i, j, k): atom i has exactly 2 C neighbours j, k.
                 These are standard zigzag/armchair edge sites; need 1 H each.
        edges1 — list of (i, j): atom i has exactly 1 C neighbour j.
                 These are corner atoms; need 2 H each to satisfy sp2.

    Bulk atoms (3 C neighbours) and H atoms are ignored.
    """
    pos, sym, tree = geo
    edges2, edges1 = [], []

    for i, symbol in enumerate(sym):
        if symbol != "C":
            continue
        neighbors = [
            j for j in tree.query_ball_point(pos[i], rmax)
            if j != i and sym[j] == "C" and rmin <= np.linalg.norm(pos[i] - pos[j]) <= rmax
        ]
        if   len(neighbors) == 2: edges2.append((i, neighbors[0], neighbors[1]))
        elif len(neighbors) == 1: edges1.append((i, neighbors[0]))

    return edges2, edges1


def passivate_from_classes(atoms, geo, edges2, edges1,
                           ch=C_H, min_sep=MIN_SEP, inplane=True, alpha_deg=90.0):
    """
    Place hydrogen atoms on classified edge sites to satisfy sp2 valence.

    edges2 sites (2 neighbours): one H is placed along the bisector pointing
    away from the two neighbours — i.e. in the direction (atom - midpoint).

    edges1 sites (1 neighbour, corner atoms): two H atoms are placed
    symmetrically at ±alpha_deg from the extension of the remaining C-C bond,
    giving the correct ~120° sp2 geometry (alpha_deg=60° means 60° away from
    the bond axis on each side, so the H-C-H angle is 2*60° = 120°).

    inplane=True projects all directions onto the XY plane before normalising,
    which is correct for flat or pillar-top layers but should be set to False
    if passivating truly 3-D edge geometries.

    A live cKDTree (dynamic_pos) is rebuilt before each placement to prevent
    hydrogen atoms from being placed too close to atoms placed earlier in the
    same passivation pass.
    """
    pos, sym, _  = geo
    h_positions  = []
    z_axis       = np.array([0.0, 0.0, 1.0])
    dynamic_pos  = list(pos)   # grows as we add H atoms

    def unit_vector(v):
        if inplane:
            v = np.array([v[0], v[1], 0.0])   # project onto XY plane
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else None

    def place_hydrogen(pC, direction):
        """Place one H at pC + ch*direction; nudge outward if too close."""
        pH = pC + ch * direction
        dmin, _ = cKDTree(dynamic_pos).query(pH, k=1)
        if (dmin if np.isscalar(dmin) else np.min(dmin)) < min_sep:
            pH = pC + (ch + 0.3) * direction   # small outward nudge
        dynamic_pos.append(pH)
        return pH

    # --- edges2: one H along the outward bisector ---
    for i, j, k in edges2:
        midpoint  = 0.5 * (pos[j] + pos[k])
        direction = unit_vector(pos[i] - midpoint)
        if direction is not None:
            h_positions.append(place_hydrogen(pos[i], direction))

    # --- edges1: two H atoms at ±alpha_deg from the bond extension ---
    ca, sa = np.cos(np.deg2rad(alpha_deg)), np.sin(np.deg2rad(alpha_deg))
    for i, j in edges1:
        v1 = unit_vector(pos[i] - pos[j])   # direction away from neighbour
        if v1 is None:
            continue
        # u is perpendicular to v1 in-plane
        u = np.cross(z_axis, v1)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(np.array([1.0, 0.0, 0.0]), v1)
        u /= np.linalg.norm(u)
        for d in (ca * v1 + sa * u, ca * v1 - sa * u):
            direction = unit_vector(d)
            if direction is not None:
                h_positions.append(place_hydrogen(pos[i], direction))

    if not h_positions:
        return atoms

    final = atoms.copy()
    final.extend(Atoms('H' * len(h_positions), positions=np.array(h_positions)))
    return final


def passivate_edges(atoms, ch=C_H, min_sep=MIN_SEP, inplane=True, alpha_deg=60.0):
    """Full passivation pipeline: clean dangling carbons, classify, then place H."""
    cleaned        = clean_dangling_carbons(atoms)
    geo            = make_geo(cleaned)
    edges2, edges1 = classify_carbon_sites(geo)
    return passivate_from_classes(
        cleaned, geo, edges2, edges1,
        ch=ch, min_sep=min_sep, inplane=inplane, alpha_deg=alpha_deg
    )


def check_overlaps(geo, cutoff=0.9):
    """Return a list of (i, j, distance) pairs that are closer than cutoff A."""
    pos, _, tree = geo
    pairs = tree.query_pairs(cutoff)
    return [(i, j, float(np.linalg.norm(pos[i] - pos[j]))) for (i, j) in sorted(pairs)]


# =====================================================================
# --- FILE WRITING ---
# =====================================================================

def write_lammps_full(structure: Atoms, filename: str, padding: float = 20.0):
    """
    Write the structure to a LAMMPS data file in 'atom_style full' format.

    Atom types: 1 = C (12.011 amu), 2 = H (1.008 amu).
    All atoms are assigned to molecule 1 and carry zero partial charge
    (charge is handled by the force field, not hard-coded here).
    A padding of 20 A is added on all sides of the simulation box.
    """
    pos        = structure.get_positions()
    sym        = structure.get_chemical_symbols()
    type_map   = {"C": 1, "H": 2}
    atom_types = [type_map.get(s, 1) for s in sym]
    num_types  = 2 if 2 in atom_types else 1

    mins = pos.min(axis=0) - padding
    maxs = pos.max(axis=0) + padding

    with open(filename, "w") as f:
        f.write("LAMMPS data: graphene/graphite (full)\n\n")
        f.write(f"{len(pos)} atoms\n\n")
        f.write(f"{num_types} atom types\n\n")
        f.write(f"{mins[0]:.6f} {maxs[0]:.6f} xlo xhi\n")
        f.write(f"{mins[1]:.6f} {maxs[1]:.6f} ylo yhi\n")
        f.write(f"{mins[2]:.6f} {maxs[2]:.6f} zlo zhi\n\n")
        f.write("Masses\n\n1 12.011\n")
        if num_types == 2:
            f.write("2 1.008\n")
        f.write("\nAtoms # full\n\n")
        for i, (t, (x, y, z)) in enumerate(zip(atom_types, pos), start=1):
            f.write(f"{i} 1 {t} 0.0 {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"[INFO] LAMMPS data written to {filename}")


# =====================================================================
# --- MAIN ---
# =====================================================================

if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    output_dir = os.path.join(os.path.dirname(base_dir), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{STRUCTURE_TYPE}_initial.data")

    print(f"[INFO] Generating '{STRUCTURE_TYPE}' structure...")

    # Step 1: build the base monolayer (used by all three structure types)
    graphene_monolayer = create_graphene_monolayer()

    # Step 2: build the carbon scaffold for the chosen structure type
    if STRUCTURE_TYPE == "graphene":
        carbon_scaffold = graphene_monolayer

    elif STRUCTURE_TYPE == "graphite":
        carbon_scaffold = stack_graphite_layers(
            graphene_monolayer, GRAPHITE_HEIGHT, INTERLAYER_DIST, C_C_BOND
        )

    elif STRUCTURE_TYPE == "structured_graphite":
        graphite_slab   = stack_graphite_layers(
            graphene_monolayer, PILLAR_HEIGHT, INTERLAYER_DIST, C_C_BOND
        )
        carbon_scaffold = carve_pillars_from_graphite(
            graphite_slab, PILLAR_RADIUS, PILLAR_PERIOD
        )

    else:
        raise ValueError(
            f"Unknown STRUCTURE_TYPE '{STRUCTURE_TYPE}'. "
            "Choose 'graphene', 'graphite', or 'structured_graphite'."
        )

    # Step 3: passivate open edges if requested
    passivate = {
        "graphene":            PASSIVATE_GRAPHENE,
        "graphite":            PASSIVATE_GRAPHITE,
        "structured_graphite": PASSIVATE_STRUCTURED_GRAPHITE,
    }[STRUCTURE_TYPE]

    if passivate:
        print(f"[INFO] Passivating {STRUCTURE_TYPE} edges...")
        final_structure = passivate_edges(
            carbon_scaffold, ch=C_H, min_sep=MIN_SEP, inplane=True
        )
    else:
        final_structure = carbon_scaffold.copy()
        print(f"[INFO] No passivation applied for '{STRUCTURE_TYPE}'.")

    # Step 4: sanity check — warn if any atoms are unrealistically close
    geo_data = make_geo(final_structure)
    overlaps = check_overlaps(geo_data, cutoff=0.90)
    if overlaps:
        print(f"[WARNING] {len(overlaps)} close atomic contact(s) detected. "
              "Tune MIN_SEP or geometry.")

    # Step 5: write LAMMPS data file
    write_lammps_full(final_structure, output_file)

    # Step 6: report final dimensions
    positions = final_structure.get_positions()
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    dims = maxs - mins

    print(f"\n--- {STRUCTURE_TYPE} Structure Dimensions ---")
    print(f"X-range : {mins[0]:.4f} to {maxs[0]:.4f} A")
    print(f"Y-range : {mins[1]:.4f} to {maxs[1]:.4f} A")
    print(f"Z-range : {mins[2]:.4f} to {maxs[2]:.4f} A")
    print(f"Size    : {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f} A")
    print(f"Atoms   : {len(final_structure)}")
    print("-------------------------------------")