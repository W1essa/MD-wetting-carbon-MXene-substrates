# LAMMPS Workspace: Graphene & MXenes wetting
MD simulation scripts and analysis tools.

## System Setup
Runs on a remote Ubuntu workstation (Ryzen 7 8700F + RTX 5070 Ti).
Workflow: VS Code Remote via SSH/Tailscale.

## LAMMPS Build Info
LAMMPS is compiled locally with **OpenCL** support to bypass CUDA compatibility issues with the Blackwell architecture.

* **GPU Backend:** OpenCL (Selected for RTX 5070 Ti support)
* **Compiler:** GCC-12 / G++-12
* **MPI:** OpenMPI
* **Packages:** MOLECULE, KSPACE, MANYBODY, MISC, RIGID

### Running Simulations
Using 16 MPI threads + GPU acceleration:

```bash
mpirun -np 16 /path/to/lmp -sf gpu -pk gpu 1 -in scripts/run.in
```

## Python Env
Used for geometry generation and post-processing.

```bash
source venv/bin/activate
pip install -r requirements.txt
```
## Project Structure
```text
.
├── README.md
├── requirements.txt
├── venv/                  # Python virtual environment
├── build/                 # Compiled LAMMPS binary
├── scripts/               # Simulation logic & tools
│   ├── *.in
│   ├── *.py
│   ├── *.ipynb
│   └── *.vars
├── data/                  # Geometries, merged systems, processed outputs
│   ├── *.data
│   ├── *.xyz
│   ├── *.png
│   └── *.dat
├── dump/                  # Trajectory files and raw simulation dumps
│   ├── *.lammpstrj
│   └── *
└── output/                # Optional logs or additional exported results
```