# Installation Guide

Step-by-step installation instructions for Phase-Transition-Hardness.

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| CPU | 4 cores (x86_64 or ARM64) |
| RAM | 8 GB |
| Disk | 1 GB free space |
| OS | Linux, macOS, or Windows (WSL2) |
| Python | 3.10 or higher |

### Recommended for Full Experiments

| Component | Specification |
|-----------|---------------|
| CPU | 8+ cores |
| RAM | 16+ GB |
| Disk | 10 GB free space |
| OS | Linux (Ubuntu 22.04 LTS tested) |
| Python | 3.11 or 3.12 |

## Installation Methods

### Method 1: pip (Recommended for Users)

```bash
pip install phase-transition-hardness
```

### Method 2: From Source (Recommended for Developers)

```bash
git clone https://github.com/robinbishtt/Phase-Transition-Hardness.git
cd Phase-Transition-Hardness
pip install -e .
```

### Method 3: Using Makefile

```bash
git clone https://github.com/robinbishtt/Phase-Transition-Hardness.git
cd Phase-Transition-Hardness
make install
```

### Method 4: Docker

```bash
docker pull phase-transition:latest
docker run -v $(pwd)/results:/app/results phase-transition:latest
```

### Method 5: Singularity (HPC Clusters)

```bash
singularity pull phase-transition.sif docker://phase-transition:latest
singularity run phase-transition.sif
```

## Detailed Setup

### Step 1: Install Python

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**macOS (using Homebrew):**
```bash
brew install python@3.11
```

**Windows:**
Download from [python.org](https://python.org) or use Windows Subsystem for Linux (WSL2).

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Package

```bash
pip install -e .
```

### Step 5: Verify Installation

```bash
python -c "from src.energy_model import ALPHA_S; print(f'α_s = {ALPHA_S}')"
```

Expected output: `α_s = 4.267`

## Development Installation

For contributing to the codebase:

```bash
git clone https://github.com/robinbishtt/Phase-Transition-Hardness.git
cd Phase-Transition-Hardness
make install-dev
```

This installs:
- Package in editable mode
- Development dependencies (pytest, black, mypy, etc.)
- Pre-commit hooks

## External Solver Integration (Optional)

### Kissat

```bash
git clone https://github.com/arminbiere/kissat.git
cd kissat
./configure && make
sudo cp build/kissat /usr/local/bin/
```

### CaDiCaL

```bash
git clone https://github.com/arminbiere/cadical.git
cd cadical
./configure && make
sudo cp build/cadical /usr/local/bin/
```

## Platform-Specific Notes

### Linux

No special requirements. All dependencies available via pip.

### macOS

If you encounter build errors for numpy/scipy:

```bash
brew install openblas
export OPENBLAS=$(brew --prefix openblas)
pip install numpy scipy
```

### Windows

Recommended: Use WSL2 for best compatibility.

For native Windows:
```bash
pip install numpy scipy matplotlib --only-binary :all:
```

### HPC Clusters

Load required modules:
```bash
module load python/3.11
module load gcc/11
pip install --user -r requirements.txt
```

## GPU Support (Optional)

For GPU-accelerated experiments (experimental):

```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

## Troubleshooting Installation

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
cd /path/to/Phase-Transition-Hardness
pip install -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Issue: `numpy` build fails

**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install numpy --only-binary :all:
```

### Issue: Permission denied

**Solution:**
```bash
pip install --user -e .
# or
pip install -e . --break-system-packages  # Not recommended
```

### Issue: Version conflicts

**Solution:**
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

## Verification

Run the test suite:

```bash
make test
```

All tests should pass.

## Uninstallation

```bash
pip uninstall phase-transition-hardness
```

## Next Steps

After installation:
1. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for running experiments
2. Explore [notebooks/](../notebooks/) for tutorials
3. Run `make reproduce-quick` for a quick validation
