#!/usr/bin/env python3
import numpy as np
import subprocess
import textwrap
import random
import os
import shutil

# =====================================================================
# --- SCRIPT PARAMETERS ---
# =====================================================================
CUBE_SIZE = 80.0               # The length of one side of the water box in Angstroms
WATER_VOLUME_PER_MOL = 30.0    # Approximate volume per water molecule
PACKMOL_TOLERANCE = 2.0        # Minimum distance between atoms from different molecules
SEED = random.randint(1, 100000)

# =====================================================================
# --- FILE PATHS ---
# =====================================================================
DATA_DIR = "../data"
WATER_TEMPLATE_FILE = os.path.join(DATA_DIR, "water.xyz")
PACKMOL_XYZ_OUTPUT = "water_cube_packed.xyz"
PACKMOL_INPUT_FILE = "packmol_cube.inp"
PACKMOL_LOG_FILE = "packmol_cube.log"
TEMP_LAMMPS_DATA_FILE = "system_temp.data"
FINAL_LAMMPS_DATA = os.path.join(DATA_DIR, "water_cube_initial.data")


# =====================================================================
# --- FUNCTIONS ---
# =====================================================================
def create_water_template_file(filename):
    """Creates a standard SPC/E water molecule XYZ file if it does not exist."""
    content = """3
SPC/E water molecule for PACKMOL
O   0.000000  0.000000  0.000000
H   1.000000  0.000000  0.000000
H  -0.333333  0.942809  0.000000
"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(content)


def calculate_num_molecules_in_cube(cube_size, molecule_volume):
    """Calculates the number of water molecules to fill a cube."""
    return int((cube_size ** 3) / molecule_volume)


def generate_packmol_input_cube(cube_size, num_molecules, inp_file):
    """Generates the input file for Packmol to place molecules in a CUBIC box."""
    half = cube_size / 2.0
    packmol_content = f"""
    # Packmol input file for a water cube
    tolerance {PACKMOL_TOLERANCE}
    filetype xyz
    output {PACKMOL_XYZ_OUTPUT}
    seed {SEED}

    # Water molecule structure definition
    structure {WATER_TEMPLATE_FILE}
      number {num_molecules}
      # Place molecules inside the cube
      inside box {-half:.2f} {-half:.2f} {-half:.2f} {half:.2f} {half:.2f} {half:.2f}
    end structure
    """
    with open(inp_file, "w") as f:
        f.write(textwrap.dedent(packmol_content))


def run_packmol(inp_file, log_file):
    """Executes Packmol and checks for the output file."""
    cmd = f"packmol < {inp_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if not os.path.exists(PACKMOL_XYZ_OUTPUT):
            raise RuntimeError(f"Packmol finished, but output file '{PACKMOL_XYZ_OUTPUT}' is missing!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Packmol failed! Check log for details: {log_file}")
        with open(log_file, "w") as log:
            log.write("--- STDOUT ---\n")
            log.write(e.stdout)
            log.write("\n--- STDERR ---\n")
            log.write(e.stderr)
        raise


def generate_lammps_topology(num_molecules, box_size, data_file_out):
    """Generates the LAMMPS data file with topology and force field parameters."""
    total_atoms = num_molecules * 3
    total_bonds = num_molecules * 2
    total_angles = num_molecules * 1

    # SPC/E parameters
    mass_O, mass_H = 15.9994, 1.008
    charge_O, charge_H = -0.8476, 0.4238
    pair_eps_O, pair_sig_O = 0.15535, 3.166
    bond_K, bond_r0 = 1000.0, 1.0
    angle_K, angle_th0 = 100.0, 109.47

    lines = ["LAMMPS Description\n\n"]
    lines.extend([
        f"{total_atoms:12} atoms\n", f"{total_bonds:12} bonds\n", f"{total_angles:12} angles\n\n",
        f"           2 atom types\n", f"           1 bond types\n", f"           1 angle types\n\n"
    ])

    half_box = box_size / 2.0
    lines.extend([
        f"{-half_box:12.4f} {half_box:12.4f} xlo xhi\n", f"{-half_box:12.4f} {half_box:12.4f} ylo yhi\n",
        f"{-half_box:12.4f} {half_box:12.4f} zlo zhi\n\n"
    ])

    lines.extend(["Masses\n\n", f" 1 {mass_O}\n", f" 2 {mass_H}\n\n"])
    lines.extend(["Pair Coeffs\n\n", f" 1 {pair_eps_O} {pair_sig_O}\n", " 2 0.0 0.0\n\n"])
    lines.extend(["Bond Coeffs\n\n", f" 1 {bond_K} {bond_r0}\n\n"])
    lines.extend(["Angle Coeffs\n\n", f" 1 {angle_K} {angle_th0}\n\n"])

    # Use clean headers for robust parsing
    lines.append("Atoms\n\n")
    for i in range(num_molecules):
        mol_id = i + 1
        o_id, h1_id, h2_id = (i * 3) + 1, (i * 3) + 2, (i * 3) + 3
        lines.append(f"{o_id:7d} {mol_id:5d} 1 {charge_O: .4f} 0.0 0.0 0.0\n")
        lines.append(f"{h1_id:7d} {mol_id:5d} 2 {charge_H: .4f} 0.0 0.0 0.0\n")
        lines.append(f"{h2_id:7d} {mol_id:5d} 2 {charge_H: .4f} 0.0 0.0 0.0\n")

    lines.append("\nBonds\n\n")
    for i in range(num_molecules):
        bond_id1, bond_id2 = (i * 2) + 1, (i * 2) + 2
        o_id, h1_id, h2_id = (i * 3) + 1, (i * 3) + 2, (i * 3) + 3
        lines.append(f"{bond_id1:7d} 1 {o_id:5d} {h1_id:5d}\n")
        lines.append(f"{bond_id2:7d} 1 {o_id:5d} {h2_id:5d}\n")

    lines.append("\nAngles\n\n")
    for i in range(num_molecules):
        angle_id = i + 1
        o_id, h1_id, h2_id = (i * 3) + 1, (i * 3) + 2, (i * 3) + 3
        lines.append(f"{angle_id:7d} 1 {h1_id:5d} {o_id:5d} {h2_id:5d}\n")

    with open(data_file_out, "w") as f:
        f.writelines(lines)


def patch_coords_from_xyz(data_file_in, xyz_file, data_file_out):
    """Replaces coordinates in a LAMMPS data file with coordinates from an XYZ file."""
    from ase.io import read as ase_read
    
    atoms = ase_read(xyz_file)
    coords = atoms.get_positions()
    
    with open(data_file_in, "r") as f:
        lines = f.readlines()

    # Robustly find the start of the "Atoms" section
    atoms_header_index = -1
    for i, line in enumerate(lines):
        if line.strip() == "Atoms":
            atoms_header_index = i
            break
    
    if atoms_header_index == -1:
        raise RuntimeError("Could not find 'Atoms' section in the data file.")

    atoms_start_index = atoms_header_index + 2  # Skip header and blank line

    new_lines = lines[:atoms_start_index]
    for i in range(len(coords)):
        parts = lines[atoms_start_index + i].split()
        x, y, z = coords[i]
        parts[4], parts[5], parts[6] = f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"
        new_lines.append(" ".join(parts) + "\n")
    
    # Robustly find the start of the "Bonds" section to append the rest of the file
    bonds_header_index = -1
    for i, line in enumerate(lines):
        if line.strip() == "Bonds":
            bonds_header_index = i
            break
            
    if bonds_header_index != -1:
        # Append from the blank line *before* the Bonds header
        new_lines.extend(lines[bonds_header_index-1:])
    
    os.makedirs(os.path.dirname(data_file_out), exist_ok=True)
    with open(data_file_out, "w") as f:
        f.writelines(new_lines)


def cleanup_temp_files(*files):
    """Removes temporary files generated during the process."""
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {f}: {e}")


# =====================================================================
# --- MAIN EXECUTION SCRIPT ---
# =====================================================================
if __name__ == "__main__":

    if not os.path.exists(WATER_TEMPLATE_FILE):
        create_water_template_file(WATER_TEMPLATE_FILE)

    # --- Step 1: Calculate the number of molecules for the cube ---
    n_molecules = calculate_num_molecules_in_cube(CUBE_SIZE, WATER_VOLUME_PER_MOL)
    print(f"🔹 System Setup: Creating a {CUBE_SIZE} Å box with {n_molecules} water molecules.")

    # --- Step 2: Run Packmol to get packed coordinates ---
    generate_packmol_input_cube(CUBE_SIZE, n_molecules, PACKMOL_INPUT_FILE)
    run_packmol(PACKMOL_INPUT_FILE, PACKMOL_LOG_FILE)

    # --- Step 3: Generate the base LAMMPS data file with topology ---
    generate_lammps_topology(n_molecules, CUBE_SIZE, TEMP_LAMMPS_DATA_FILE)
    
    # --- Step 4: Patch the data file with coordinates from Packmol ---
    patch_coords_from_xyz(TEMP_LAMMPS_DATA_FILE, PACKMOL_XYZ_OUTPUT, FINAL_LAMMPS_DATA)
      
    # --- Step 5: Clean up intermediate files ---
    cleanup_temp_files(
        PACKMOL_INPUT_FILE,
        PACKMOL_LOG_FILE,
        PACKMOL_XYZ_OUTPUT,
        TEMP_LAMMPS_DATA_FILE
    )
    print(f"Final LAMMPS data file is located at: {FINAL_LAMMPS_DATA}")