"""Wrapper for the Kissat SAT solver (Armin Biere).

Wall-clock runtime capture
--------------------------
The paper's hardness metric is H(n,α) = E[log T]/n where T is wall-clock
seconds, NOT decision counts.  This wrapper measures T via
time.perf_counter() bracketing the subprocess call and also attempts to
parse Kissat's own timing line ("c total real time") as a cross-check.
"""

import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


class KissatWrapper:
    """Wrapper for Kissat SAT solver."""

    # Kissat's summary line format: "c total real time : X.XX seconds"
    _REAL_TIME_PREFIX = "c total real time"

    def __init__(self, executable: str = "kissat", timeout: Optional[int] = 3600):
        """
        Parameters
        ----------
        executable : str
            Path or name of the kissat binary.
        timeout : int or None
            Wall-clock timeout in seconds.  The paper uses 3600 s.
        """
        self.executable = executable
        self.timeout    = timeout
        self._check_executable()

    def _check_executable(self) -> None:
        """Verify Kissat is installed and record its version."""
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            self.version = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(
                f"Kissat not found at '{self.executable}'. "
                "Install from https://github.com/arminbiere/kissat "
                "(paper uses Kissat 3.1.0)."
            )

    def _instance_to_dimacs(self, instance: Dict) -> str:
        """Serialise instance dict to DIMACS CNF string."""
        lines = [f"p cnf {instance['n']} {instance['m']}"]
        for clause in instance["clauses"]:
            lines.append(" ".join(str(lit) for lit in clause) + " 0")
        return "\n".join(lines)

    def solve(
        self,
        instance: Dict,
        options: Optional[List[str]] = None,
    ) -> Dict:
        """Solve one instance and return result including wall-clock time.

        Returns
        -------
        dict with keys:
            satisfiable : bool or None  (None = timeout / error)
            assignment  : dict[int, bool] or None
            wall_time   : float  - seconds measured by perf_counter (primary)
            kissat_time : float or None  - seconds from Kissat's own report
            timed_out   : bool
            decisions   : int or None  - from Kissat statistics line
            conflicts   : int or None
            stdout, stderr, returncode
        """
        options = options or []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
            f.write(self._instance_to_dimacs(instance))
            cnf_path = f.name

        try:
            cmd = [self.executable] + options + [cnf_path]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                proc      = exc  # partial result object
            wall_time = time.perf_counter() - t0

            return self._parse_output(proc, wall_time=wall_time, timed_out=timed_out)
        finally:
            os.unlink(cnf_path)

    def _parse_output(self, result, wall_time: float, timed_out: bool) -> Dict:
        """Parse Kissat stdout/stderr and augment with timing."""
        output = getattr(result, "stdout", "") or ""

        satisfiable  = None
        assignment   = {}
        kissat_time  = None
        decisions    = None
        conflicts    = None

        for line in output.split("\n"):
            line = line.strip()

            if line == "s SATISFIABLE":
                satisfiable = True
            elif line == "s UNSATISFIABLE":
                satisfiable = False
            elif line.startswith("v "):
                for lit in line[2:].split():
                    if lit == "0":
                        continue
                    val = int(lit)
                    assignment[abs(val)] = val > 0

            # Kissat timing line: "c total real time : 1.23 seconds"
            elif self._REAL_TIME_PREFIX in line.lower():
                try:
                    kissat_time = float(line.split(":")[-1].replace("seconds", "").strip())
                except (ValueError, IndexError):
                    pass

            # Decision / conflict counts from Kissat statistics
            elif line.startswith("c decisions:") or "decisions:" in line:
                try:
                    decisions = int(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("c conflicts:") or "conflicts:" in line:
                try:
                    conflicts = int(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        return {
            "satisfiable":  None if timed_out else satisfiable,
            "assignment":   assignment if satisfiable else None,
            "wall_time":    wall_time,
            "kissat_time":  kissat_time,
            "timed_out":    timed_out,
            "decisions":    decisions,
            "conflicts":    conflicts,
            "stdout":       output,
            "stderr":       getattr(result, "stderr", ""),
            "returncode":   getattr(result, "returncode", None),
        }

    def solve_with_proof(self, instance: Dict, proof_path: str) -> Dict:
        """Solve and emit a DRAT proof."""
        return self.solve(instance, options=["--no-binary", f"--proof={proof_path}"])
