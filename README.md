![License](https://img.shields.io/badge/license-MIT-teal)
![LAMMPS](https://img.shields.io/badge/LAMMPS-OpenCL-blue)
![Python](https://img.shields.io/badge/python-3.10+-purple)
![Platform](https://img.shields.io/badge/platform-Linux-orange)
# LAMMPS Workspace: Graphene & MXenes wetting
Molecular Dynamics (MD) simulation scripts and analysis tools for studying surface wettability, contact angles, and density profiles of water droplets on substrates (Graphene, Graphite, Structured Graphite, and MXenes).

## Overview
This repository provides a fully automated pipeline: from building atomistic substrate geometries and water boxes to running LAMMPS equilibration and extracting the final contact angle using Object-Oriented Python analysis.

---

## System Setup
Runs on a remote Ubuntu workstation (Ryzen 7 8700F + RTX 5070 Ti).  
Workflow: VS Code Remote via SSH/Tailscale.

### LAMMPS Build Info
LAMMPS is compiled locally with **OpenCL** support to bypass CUDA compatibility issues with the Blackwell architecture.

* **GPU Backend:** OpenCL (Selected for RTX 5070 Ti support)
* **Compiler:** GCC-12 / G++-12
* **MPI:** OpenMPI
* **Packages:** MOLECULE, KSPACE, MANYBODY, MISC, RIGID

### Running Simulations
Using 8 MPI threads + GPU acceleration via custom alias:
```bash
# 'lmpgpu' is aliased to: mpirun -np 8 /path/to/lmp -sf gpu -pk gpu 1
lmpgpu -var geom graphene -in scripts/04_analysis/equilibration.in
```

## Python Environment
Used for automated geometry generation, extracting Gibbs dividing surfaces from LAMMPS density maps, least-squares contact angle fitting, and rendering heatmaps.

```bash
source venv/bin/activate
pip install -r requirements.txt
```
## Project Structure & Pipeline
The simulation workflow is strictly organized into logical steps:
```text
.
├── scripts/               # Core simulation pipeline
│   ├── 01_substrate/      # Python builders & LAMMPS minimization for solid surfaces
│   ├── 02_water/          # Water droplet generation (Packmol/Python) & relaxation
│   ├── 03_assembly/       # Merging the droplet onto the substrate
│   └── 04_analysis/       # Main MD runs (equilibration, density grids) and Contact Angle math
├── data/                  # Geometries (.data, .xyz) and density fields (.dat)
├── dump/                  # Trajectory files (*.lammpstrj) (Ignored by Git)
├── output/                # Rendered plots, heatmaps, and analysis logs
├── requirements.txt       # Python dependencies
└── README.md
```
