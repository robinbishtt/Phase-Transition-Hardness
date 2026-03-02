"""Wrapper for CaDiCaL SAT solver."""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional, List


class CadicalWrapper:
    """Wrapper for CaDiCaL SAT solver by Armin Biere."""

    def __init__(self, executable: str = "cadical", timeout: Optional[int] = None):
        self.executable = executable
        self.timeout = timeout
        self._check_executable()

    def _check_executable(self) -> None:
        """Verify CaDiCaL is installed."""
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.version = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(
                f"CaDiCaL not found: {self.executable}. "
                "Install from https://github.com/arminbiere/cadical"
            )

    def _instance_to_dimacs(self, instance: Dict) -> str:
        """Convert instance to DIMACS CNF format."""
        n = instance["n"]
        m = instance["m"]
        clauses = instance["clauses"]

        lines = [f"p cnf {n} {m}"]
        for clause in clauses:
            line = " ".join(str(lit) for lit in clause) + " 0"
            lines.append(line)

        return "\n".join(lines)

    def solve(
        self,
        instance: Dict,
        options: Optional[List[str]] = None
    ) -> Dict:
        """Solve instance using CaDiCaL."""
        options = options or []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cnf", delete=False
        ) as f:
            f.write(self._instance_to_dimacs(instance))
            cnf_path = f.name

        try:
            cmd = [self.executable] + options + [cnf_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return self._parse_output(result)
        finally:
            os.unlink(cnf_path)

    def _parse_output(self, result: subprocess.CompletedProcess) -> Dict:
        """Parse CaDiCaL output."""
        output = result.stdout

        satisfiable = None
        assignment = {}
        decisions = 0
        conflicts = 0

        for line in output.split("\n"):
            line = line.strip()

            if line == "s SATISFIABLE":
                satisfiable = True
            elif line == "s UNSATISFIABLE":
                satisfiable = False
            elif line.startswith("v "):
                literals = line[2:].split()
                for lit in literals:
                    if lit == "0":
                        continue
                    val = int(lit)
                    var = abs(val)
                    assignment[var] = val > 0
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
            "satisfiable": satisfiable,
            "assignment": assignment if satisfiable else None,
            "decisions": decisions,
            "conflicts": conflicts,
            "stdout": output,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def solve_incremental(
        self,
        instance: Dict,
        assumptions: List[int]
    ) -> Dict:
        """Solve with assumptions (incremental mode)."""
        options = [f"--assume={','.join(map(str, assumptions))}"]
        return self.solve(instance, options)