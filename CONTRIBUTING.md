Thank you for your interest in contributing to this research software.



Contributors are expected to maintain professional, respectful discourse consistent with
academic norms. Issues and pull requests should be focused on scientific accuracy,
numerical correctness, and reproducibility.





- Use GitHub Issues to report bugs, numerical discrepancies, or inconsistencies with the manuscript.
- Include: Python version, OS, full traceback, minimal reproducible example.
- For scientific discrepancies, specify which manuscript figure or table the output contradicts.



1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-description
   ```

2. Ensure all existing experiments still pass:
   ```bash
   python src/validation.py --results_dir results/
   ```

3. Write clear, documented code following the existing style:
   - Module-level docstrings
   - NumPy-style function docstrings
   - Type hints on all public functions
   - No placeholder logic  all computations must be mathematically grounded

4. Add tests if introducing new numerical routines.

5. Submit PR with a clear description of what changed and why.



If contributing new experiments or analysis:
- Ground all computations in cited literature
- Avoid modifications that contradict manuscript predictions (α_s ≈ 4.27, ν ≈ 2.3, etc.)
- Extensions beyond manuscript scope are welcome but must be clearly labelled as such



- Follow PEP 8
- Maximum line length: 100 characters
- Use `numpy` for all numerical computation; avoid pure-Python loops where vectorisation is feasible
- Random seeds must be explicitly passed  no global state



- All experiments must be deterministic given the same seed
- Floating-point operations should be numerically stable (prefer log-space for probabilities)
- Bootstrap confidence intervals required for all statistical claims



This repository uses semantic versioning. Breaking changes to the public API of `src/` modules
require a major version bump.



For scientific questions, contact Robin Bisht at bishtrobin75@gmail.com.