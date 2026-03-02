"""Wrapper for Kissat SAT solver."""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional, List


class KissatWrapper:
    """Wrapper for Kissat SAT solver by Armin Biere."""

    def __init__(self, executable: str = "kissat", timeout: Optional[int] = None):
        self.executable = executable
        self.timeout = timeout
        self._check_executable()

    def _check_executable(self) -> None:
        """Verify Kissat is installed."""
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
                f"Kissat not found: {self.executable}. "
                "Install from https://github.com/arminbiere/kissat"
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
        """Solve instance using Kissat."""
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
        """Parse Kissat output."""
        output = result.stdout

        satisfiable = None
        assignment = {}

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

        return {
            "satisfiable": satisfiable,
            "assignment": assignment if satisfiable else None,
            "stdout": output,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def solve_with_proof(
        self,
        instance: Dict,
        proof_path: str
    ) -> Dict:
        """Solve and generate DRAT proof."""
        return self.solve(instance, options=["--no-binary", f"--proof={proof_path}"])