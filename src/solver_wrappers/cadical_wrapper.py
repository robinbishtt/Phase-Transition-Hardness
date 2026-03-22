"""Wrapper for the CaDiCaL SAT solver (Armin Biere).

Wall-clock runtime capture
--------------------------
The paper's hardness metric is H(n,α) = E[log T]/n where T is wall-clock
seconds from CaDiCaL, NOT decision counts.  This wrapper measures T via
time.perf_counter() and also attempts to parse CaDiCaL's own "c time"
reporting line.
"""

import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


class CadicalWrapper:
    """Wrapper for CaDiCaL SAT solver."""

    def __init__(self, executable: str = "cadical", timeout: Optional[int] = 3600):
        """
        Parameters
        ----------
        executable : str
            Path or name of the cadical binary.
        timeout : int or None
            Wall-clock timeout in seconds.  The paper uses 3600 s.
        """
        self.executable = executable
        self.timeout    = timeout
        self._check_executable()

    def _check_executable(self) -> None:
        """Verify CaDiCaL is installed and record its version."""
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            self.version = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(
                f"CaDiCaL not found at '{self.executable}'. "
                "Install from https://github.com/arminbiere/cadical "
                "(paper uses CaDiCaL 1.9.4)."
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
            satisfiable : bool or None  (None = timeout)
            assignment  : dict[int, bool] or None
            wall_time   : float  - seconds measured by perf_counter (primary)
            cadical_time: float or None  - from CaDiCaL's own report line
            timed_out   : bool
            decisions   : int or None
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
                proc      = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout
                )
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                proc      = exc
            wall_time = time.perf_counter() - t0

            return self._parse_output(proc, wall_time=wall_time, timed_out=timed_out)
        finally:
            os.unlink(cnf_path)

    def _parse_output(self, result, wall_time: float, timed_out: bool) -> Dict:
        """Parse CaDiCaL stdout/stderr and augment with timing."""
        output = getattr(result, "stdout", "") or ""

        satisfiable   = None
        assignment    = {}
        cadical_time  = None
        decisions     = None
        conflicts     = None

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

            # CaDiCaL: "c total process time since initialization: X.XX seconds"
            elif "total process time" in line.lower() or (
                line.startswith("c time") and ":" in line
            ):
                try:
                    cadical_time = float(
                        line.split(":")[-1].replace("seconds", "").strip().split()[0]
                    )
                except (ValueError, IndexError):
                    pass

            # CaDiCaL statistics
            elif line.startswith("c decisions:"):
                try:
                    decisions = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("c conflicts:"):
                try:
                    conflicts = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        return {
            "satisfiable":   None if timed_out else satisfiable,
            "assignment":    assignment if satisfiable else None,
            "wall_time":     wall_time,
            "cadical_time":  cadical_time,
            "timed_out":     timed_out,
            "decisions":     decisions,
            "conflicts":     conflicts,
            "stdout":        output,
            "stderr":        getattr(result, "stderr", ""),
            "returncode":    getattr(result, "returncode", None),
        }

    def solve_incremental(self, instance: Dict, assumptions: List[int]) -> Dict:
        """Solve with literal assumptions (incremental mode)."""
        options = [f"--assume={','.join(map(str, assumptions))}"]
        return self.solve(instance, options)
