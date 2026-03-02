# LAMMPS Course & Research Workspace

This repository contains scripts, data files, and Python tools for molecular dynamics simulations, specifically focusing on the wetting of graphene and MXenes.

## Remote Workspace Architecture

To ensure high performance and flexibility, the workflow is distributed across devices using Tailscale and VS Code Remote - SSH:

* **Compute Node (Server):** A dedicated Ubuntu 24.04 machine equipped with an AMD Ryzen 7 8700F (16 threads) and an NVIDIA GeForce RTX 5070 Ti. All files, heavy computations, and the Python environment reside here.
* **Client Nodes:** Work and home laptops connect to the Compute Node remotely via Tailscale. VS Code handles the UI, allowing code editing and simulation execution from any location without transferring files.

## LAMMPS Build Configuration

LAMMPS was compiled from source (stable branch) directly on the Compute Node to maximize hardware utilization.

* **Compiler:** GCC-12 / G++-12 (Downgraded for CUDA compatibility)
* **Parallelization:** OpenMPI
* **GPU Acceleration:** Enabled via CUDA (PKG_GPU=yes, GPU_API=cuda)
* **Included Packages:** MOLECULE, KSPACE, MANYBODY, MISC, RIGID
* **Executable Location:** /usr/local/bin/lmp

### Running Simulations

To utilize all 16 threads of the Ryzen CPU, simulations are launched using MPI:

```bash
mpirun -np 16 lmp -in scripts/your_script.in
```

## Python Environment
A dedicated Python virtual environment is set up in the root directory for generating geometries and analyzing dump files.

## Setup & Activation
# Activate the environment
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

Key libraries include numpy, pandas, MDAnalysis, and ovito.