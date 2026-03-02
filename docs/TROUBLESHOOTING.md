# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### ImportError: No module named 'src'

**Cause:** Package not installed or wrong directory.

**Solution:**
```bash
cd /path/to/Phase-Transition-Hardness
pip install -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Build errors for numpy/scipy

**Cause:** Missing system dependencies.

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get install python3-dev build-essential libopenblas-dev
pip install numpy scipy
```

**Solution (macOS):**
```bash
brew install openblas
export OPENBLAS=$(brew --prefix openblas)
pip install numpy scipy
```

### Permission denied during pip install

**Cause:** Insufficient permissions.

**Solution:**
```bash
pip install --user -e .
```

## Runtime Issues

### MemoryError during experiments

**Cause:** Insufficient RAM for requested system size.

**Solutions:**
1. Reduce system size: `--n 100 200` instead of `--n 400 800`
2. Reduce instances: `--n_instances 100` instead of `1000`
3. Use fewer parallel jobs: `--n_jobs 1`

**Example:**
```bash
python experiments/alpha_sweep.py --n 100 200 --n_instances 100 --n_jobs 1
```

### Solver timeout/cutoff

**Cause:** Hard instances at critical α values.

**Solutions:**
1. Increase max_decisions: `--max_decisions 500000`
2. Use WalkSAT for faster (but incomplete) solving
3. Reduce system size

**Example:**
```bash
python experiments/alpha_sweep.py --solver walksat --max_decisions 50000
```

### Results differ from manuscript

**Cause:** Insufficient sampling or different random seed.

**Solutions:**
1. Increase n_instances: `--n_instances 1000`
2. Use manuscript seed: `--seed 20240223`
3. Check validation: `python src/validation.py`

## Validation Issues

### Validation check fails

**Check 1 (α_s):** Outside [4.20, 4.35]
- **Fix:** Increase n_instances or extend alpha range

**Check 2 (α*):** Outside [4.10, 4.40]
- **Fix:** Increase resolution near α=4.2

**Check 3 (γ_max):** Outside [0.005, 0.05]
- **Fix:** Check solver configuration

**Check 4 (R²):** < 0.85
- **Fix:** Increase system sizes (n ≥ 400)

**Check 5 (Residual):** ≥ 0.10
- **Fix:** Increase n_instances or check data quality

**Check 6 (ν):** Outside [1.5, 3.5]
- **Fix:** Verify FSS convergence

**Check 7 (Barrier):** ≤ 0
- **Fix:** Check energy model implementation

**Check 8 (Monotonicity):** Violations > 0
- **Fix:** Statistical fluctuations expected at small n

## Experiment Issues

### Alpha sweep too slow

**Cause:** Large system sizes or many instances.

**Solutions:**
1. Use parallel processing: `--n_jobs -1`
2. Reduce alpha resolution: `--alpha_step 0.1`
3. Start with smaller n: `--n 50 100`

### FSS collapse poor quality

**Cause:** Insufficient data for quality collapse.

**Solutions:**
1. Increase system sizes: `--n 100 200 400 800`
2. Increase instances: `--n_instances 1000`
3. Finer alpha resolution: `--alpha_step 0.02`

### NaN in results

**Cause:** P_sat curve doesn't cross 0.5 in alpha range.

**Solution:**
```bash
python experiments/alpha_sweep.py --alpha_min 3.0 --alpha_max 5.5
```

## Docker Issues

### Docker build fails

**Cause:** Missing Docker or insufficient permissions.

**Solution:**
```bash
sudo docker build -t phase-transition .
# or add user to docker group
sudo usermod -aG docker $USER
```

### Docker container exits immediately

**Cause:** No command specified.

**Solution:**
```bash
docker run -it phase-transition:latest bash
```

## HPC Issues

### Singularity build fails

**Cause:** Missing Singularity or wrong base image.

**Solution:**
```bash
singularity build phase-transition.sif containers/singularity.def
```

### Job killed on cluster

**Cause:** Exceeded time or memory limits.

**Solution:**
1. Request more resources:
```bash
#SBATCH --time=24:00:00
#SBATCH --mem=64G
```
2. Reduce experiment scale
3. Use checkpointing

## Performance Issues

### Experiments slower than expected

**Cause:** Python overhead or suboptimal configuration.

**Solutions:**
1. Use PyPy instead of CPython (experimental)
2. Enable compiler optimizations
3. Profile code: `python -m cProfile script.py`

### High memory usage

**Cause:** Storing all instances in memory.

**Solution:**
```python
# Generate and process instances one at a time
for i in range(n_instances):
    instance = generate_ksat_instance(n, alpha, k, seed)
    result = process(instance)
    save(result)
```

## Reproducibility Issues

### Different results on different machines

**Cause:** Floating-point differences or library versions.

**Expected:** Results should agree within:
- α_s: ±0.05
- ν: ±0.2
- γ_max: ±0.005

**Solution:** Use Docker for exact reproducibility.

### Non-deterministic results

**Cause:** Random seed not set or parallel execution.

**Solution:**
```python
from src.utils import make_rng
rng = make_rng(42)  # Fixed seed
```

## Getting Help

If issues persist:

1. Check [existing issues](https://github.com/robinbishtt/Phase-Transition-Hardness/issues)
2. Include in new issue:
   - Command used
   - Full error message
   - System info (`python --version`, OS)
   - Expected vs actual behavior

## Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Common Error Messages

### `ValueError: n must be >= k`

**Cause:** Trying to generate k-SAT with fewer variables than clause length.

**Solution:** Use n ≥ k.

### `RuntimeError: Maximum recursion depth exceeded`

**Cause:** Very hard instance causing deep recursion in DPLL.

**Solution:** Increase recursion limit or use iterative solver.

### `KeyError: 'satisfiable'`

**Cause:** Corrupted result dictionary.

**Solution:** Check solver output format.
