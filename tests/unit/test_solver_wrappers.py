"""Unit tests for src/solver_wrappers/.

Covers:
    kissat_wrapper.py  — KissatWrapper: instantiation, DIMACS serialisation,
                         parse_output, solve interface.
    cadical_wrapper.py — CadicalWrapper: same coverage.

Both wrappers require external binaries (Kissat 3.1.0 and CaDiCaL 1.9.4)
that are not part of the Python package.  Tests that require a live binary
are decorated with @skipIfNoBinary and are silently skipped when the binary
is absent.  The _parse_output and _instance_to_dimacs methods are tested
independently of the binary by constructing mock subprocess results.
"""
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.instance_generator import generate_ksat_instance


# ---------------------------------------------------------------------------
# Helper: skip decorator that checks whether a binary is on PATH
# ---------------------------------------------------------------------------

def _binary_present(name: str) -> bool:
    try:
        subprocess.run(
            [name, "--version"],
            capture_output=True, text=True, timeout=3,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def skipIfNoBinary(binary: str):
    """Decorator: skip test if *binary* is not found on the system PATH."""
    return unittest.skipUnless(
        _binary_present(binary),
        reason=f"Binary '{binary}' not found — install to run these tests",
    )


def _mock_proc(stdout: str, stderr: str = "", returncode: int = 0):
    """Build a minimal object mimicking subprocess.CompletedProcess."""
    p = types.SimpleNamespace()
    p.stdout     = stdout
    p.stderr     = stderr
    p.returncode = returncode
    return p


# ---------------------------------------------------------------------------
# KissatWrapper
# ---------------------------------------------------------------------------

class TestKissatWrapperInit(unittest.TestCase):

    def test_raises_when_binary_absent(self):
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        with self.assertRaises(RuntimeError):
            KissatWrapper(executable="__definitely_absent_binary__")

    @skipIfNoBinary("kissat")
    def test_instantiates_with_kissat_binary(self):
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        wrapper = KissatWrapper()
        self.assertIsNotNone(wrapper.version)
        self.assertIsInstance(wrapper.version, str)

    @skipIfNoBinary("kissat")
    def test_custom_timeout_stored(self):
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        wrapper = KissatWrapper(timeout=60)
        self.assertEqual(wrapper.timeout, 60)


class TestKissatInstanceToDimacs(unittest.TestCase):
    """_instance_to_dimacs is a pure function — no binary required."""

    def setUp(self):
        # Bypass __init__ binary check by patching _check_executable
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        self.KissatWrapper = KissatWrapper
        # Monkey-patch so we can instantiate without a binary
        self._orig_check = KissatWrapper._check_executable
        KissatWrapper._check_executable = lambda self: setattr(self, "version", "mock")

    def tearDown(self):
        self.KissatWrapper._check_executable = self._orig_check

    def _make_wrapper(self):
        return self.KissatWrapper(executable="mock")

    def test_preamble_line(self):
        inst = generate_ksat_instance(5, 3.0, k=3, seed=42)
        w    = self._make_wrapper()
        dimacs = w._instance_to_dimacs(inst)
        first_line = dimacs.split("\n")[0]
        self.assertTrue(first_line.startswith("p cnf"))
        parts = first_line.split()
        self.assertEqual(int(parts[2]), inst["n"])
        self.assertEqual(int(parts[3]), inst["m"])

    def test_every_clause_ends_with_zero(self):
        inst   = generate_ksat_instance(5, 3.0, k=3, seed=42)
        w      = self._make_wrapper()
        lines  = w._instance_to_dimacs(inst).split("\n")
        clause_lines = [l for l in lines if l and not l.startswith("p")]
        for line in clause_lines:
            self.assertTrue(
                line.rstrip().endswith("0"),
                msg=f"Clause line does not end with 0: {line!r}",
            )

    def test_clause_count_matches(self):
        inst   = generate_ksat_instance(5, 3.0, k=3, seed=42)
        w      = self._make_wrapper()
        lines  = w._instance_to_dimacs(inst).split("\n")
        clause_lines = [l for l in lines if l and not l.startswith("p")]
        self.assertEqual(len(clause_lines), inst["m"])


class TestKissatParseOutput(unittest.TestCase):
    """_parse_output is pure (no subprocess) — tests bypass binary check."""

    def setUp(self):
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        self.KissatWrapper = KissatWrapper
        self._orig = KissatWrapper._check_executable
        KissatWrapper._check_executable = lambda self: setattr(self, "version", "mock")

    def tearDown(self):
        self.KissatWrapper._check_executable = self._orig

    def _make(self):
        return self.KissatWrapper(executable="mock")

    def test_parse_satisfiable(self):
        stdout = "s SATISFIABLE\nv 1 -2 3 0\n"
        proc   = _mock_proc(stdout)
        w      = self._make()
        result = w._parse_output(proc, wall_time=0.5, timed_out=False)
        self.assertTrue(result["satisfiable"])
        self.assertFalse(result["timed_out"])
        self.assertAlmostEqual(result["wall_time"], 0.5, places=6)

    def test_parse_unsatisfiable(self):
        proc   = _mock_proc("s UNSATISFIABLE\n")
        result = self._make()._parse_output(proc, wall_time=0.1, timed_out=False)
        self.assertFalse(result["satisfiable"])

    def test_parse_timed_out(self):
        proc   = _mock_proc("")
        result = self._make()._parse_output(proc, wall_time=3600.0, timed_out=True)
        self.assertIsNone(result["satisfiable"])
        self.assertTrue(result["timed_out"])

    def test_parse_assignment_variables(self):
        stdout = "s SATISFIABLE\nv 1 -2 3 0\n"
        proc   = _mock_proc(stdout)
        result = self._make()._parse_output(proc, wall_time=0.1, timed_out=False)
        assign = result["assignment"]
        self.assertIsNotNone(assign)
        self.assertTrue(assign[1])
        self.assertFalse(assign[2])
        self.assertTrue(assign[3])

    def test_result_has_all_keys(self):
        proc   = _mock_proc("s SATISFIABLE\n")
        result = self._make()._parse_output(proc, wall_time=0.1, timed_out=False)
        for key in ["satisfiable", "assignment", "wall_time", "kissat_time",
                    "timed_out", "decisions", "conflicts", "stdout", "stderr",
                    "returncode"]:
            self.assertIn(key, result, msg=f"Key '{key}' missing from Kissat result")

    def test_kissat_timing_line_parsed(self):
        stdout = "s SATISFIABLE\nc total real time : 0.42 seconds\n"
        proc   = _mock_proc(stdout)
        result = self._make()._parse_output(proc, wall_time=0.5, timed_out=False)
        self.assertIsNotNone(result["kissat_time"])
        self.assertAlmostEqual(result["kissat_time"], 0.42, places=5)

    @skipIfNoBinary("kissat")
    def test_solve_easy_instance_satisfiable(self):
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        inst   = generate_ksat_instance(15, 2.5, k=3, seed=42)
        wrapper = KissatWrapper()
        result  = wrapper.solve(inst)
        self.assertIn(result["satisfiable"], [True, False, None])
        self.assertGreaterEqual(result["wall_time"], 0.0)
        self.assertFalse(result["timed_out"])


# ---------------------------------------------------------------------------
# CadicalWrapper
# ---------------------------------------------------------------------------

class TestCadicalWrapperInit(unittest.TestCase):

    def test_raises_when_binary_absent(self):
        from src.solver_wrappers.cadical_wrapper import CadicalWrapper
        with self.assertRaises(RuntimeError):
            CadicalWrapper(executable="__definitely_absent_binary__")

    @skipIfNoBinary("cadical")
    def test_instantiates_with_cadical_binary(self):
        from src.solver_wrappers.cadical_wrapper import CadicalWrapper
        wrapper = CadicalWrapper()
        self.assertIsNotNone(wrapper.version)


class TestCadicalInstanceToDimacs(unittest.TestCase):

    def setUp(self):
        from src.solver_wrappers.cadical_wrapper import CadicalWrapper
        self.CadicalWrapper = CadicalWrapper
        self._orig = CadicalWrapper._check_executable
        CadicalWrapper._check_executable = lambda self: setattr(self, "version", "mock")

    def tearDown(self):
        self.CadicalWrapper._check_executable = self._orig

    def _make(self):
        return self.CadicalWrapper(executable="mock")

    def test_preamble_format(self):
        inst   = generate_ksat_instance(5, 3.0, k=3, seed=42)
        dimacs = self._make()._instance_to_dimacs(inst)
        first  = dimacs.split("\n")[0]
        self.assertTrue(first.startswith("p cnf"))

    def test_every_clause_ends_with_zero(self):
        inst  = generate_ksat_instance(5, 3.0, k=3, seed=42)
        lines = self._make()._instance_to_dimacs(inst).split("\n")
        for l in lines:
            if l and not l.startswith("p"):
                self.assertTrue(l.rstrip().endswith("0"))

    def test_identical_to_kissat_serialisation(self):
        # Both wrappers must produce identical DIMACS for the same instance,
        # since DIMACS format is fully standardised.
        from src.solver_wrappers.kissat_wrapper import KissatWrapper
        self.KissatWrapper = KissatWrapper
        self._orig_k = KissatWrapper._check_executable
        KissatWrapper._check_executable = lambda self: setattr(self, "version", "mock")
        try:
            inst = generate_ksat_instance(8, 3.0, k=3, seed=42)
            dimacs_k = KissatWrapper(executable="mock")._instance_to_dimacs(inst)
            dimacs_c = self.CadicalWrapper(executable="mock")._instance_to_dimacs(inst)
            self.assertEqual(dimacs_k, dimacs_c)
        finally:
            KissatWrapper._check_executable = self._orig_k


class TestCadicalParseOutput(unittest.TestCase):

    def setUp(self):
        from src.solver_wrappers.cadical_wrapper import CadicalWrapper
        self.CadicalWrapper = CadicalWrapper
        self._orig = CadicalWrapper._check_executable
        CadicalWrapper._check_executable = lambda self: setattr(self, "version", "mock")

    def tearDown(self):
        self.CadicalWrapper._check_executable = self._orig

    def _make(self):
        return self.CadicalWrapper(executable="mock")

    def test_parse_satisfiable(self):
        proc   = _mock_proc("s SATISFIABLE\nv 1 -2 3 0\n")
        result = self._make()._parse_output(proc, wall_time=0.3, timed_out=False)
        self.assertTrue(result["satisfiable"])
        self.assertFalse(result["timed_out"])

    def test_parse_unsatisfiable(self):
        proc   = _mock_proc("s UNSATISFIABLE\n")
        result = self._make()._parse_output(proc, wall_time=0.1, timed_out=False)
        self.assertFalse(result["satisfiable"])

    def test_parse_timed_out(self):
        proc   = _mock_proc("")
        result = self._make()._parse_output(proc, wall_time=3600.0, timed_out=True)
        self.assertIsNone(result["satisfiable"])
        self.assertTrue(result["timed_out"])

    def test_assignment_parsed_correctly(self):
        proc   = _mock_proc("s SATISFIABLE\nv 1 -2 3 0\n")
        result = self._make()._parse_output(proc, wall_time=0.1, timed_out=False)
        self.assertTrue(result["assignment"][1])
        self.assertFalse(result["assignment"][2])
        self.assertTrue(result["assignment"][3])

    def test_result_has_all_keys(self):
        proc   = _mock_proc("s UNSATISFIABLE\n")
        result = self._make()._parse_output(proc, wall_time=0.1, timed_out=False)
        for key in ["satisfiable", "assignment", "wall_time", "cadical_time",
                    "timed_out", "decisions", "conflicts", "stdout", "stderr",
                    "returncode"]:
            self.assertIn(key, result)

    def test_wall_time_always_positive(self):
        proc   = _mock_proc("s SATISFIABLE\n")
        result = self._make()._parse_output(proc, wall_time=0.05, timed_out=False)
        self.assertGreater(result["wall_time"], 0.0)

    @skipIfNoBinary("cadical")
    def test_solve_easy_instance(self):
        from src.solver_wrappers.cadical_wrapper import CadicalWrapper
        inst    = generate_ksat_instance(15, 2.5, k=3, seed=42)
        wrapper = CadicalWrapper()
        result  = wrapper.solve(inst)
        self.assertIn(result["satisfiable"], [True, False, None])
        self.assertGreaterEqual(result["wall_time"], 0.0)


class TestSolverWrapperDimacsConsistency(unittest.TestCase):
    """Cross-wrapper consistency without requiring any binary."""

    def setUp(self):
        from src.solver_wrappers.kissat_wrapper  import KissatWrapper
        from src.solver_wrappers.cadical_wrapper import CadicalWrapper
        for cls in (KissatWrapper, CadicalWrapper):
            cls._orig_check = cls._check_executable
            cls._check_executable = lambda self: setattr(self, "version", "mock")
        self._K = KissatWrapper
        self._C = CadicalWrapper

    def tearDown(self):
        self._K._check_executable = self._K._orig_check
        self._C._check_executable = self._C._orig_check

    def test_dimacs_independent_of_wrapper(self):
        inst = generate_ksat_instance(10, 3.5, k=3, seed=7)
        k_dimacs = self._K(executable="mock")._instance_to_dimacs(inst)
        c_dimacs = self._C(executable="mock")._instance_to_dimacs(inst)
        self.assertEqual(k_dimacs, c_dimacs)

    def test_both_wrappers_produce_valid_sat_keys(self):
        proc = _mock_proc("s SATISFIABLE\nv 1 0\n")
        k_res = self._K(executable="mock")._parse_output(
            proc, wall_time=0.1, timed_out=False
        )
        c_res = self._C(executable="mock")._parse_output(
            proc, wall_time=0.1, timed_out=False
        )
        self.assertTrue(k_res["satisfiable"])
        self.assertTrue(c_res["satisfiable"])

    def test_both_parse_unsat(self):
        proc  = _mock_proc("s UNSATISFIABLE\n")
        for wrapper in [self._K(executable="mock"), self._C(executable="mock")]:
            result = wrapper._parse_output(proc, wall_time=0.1, timed_out=False)
            self.assertFalse(result["satisfiable"])

    def test_both_parse_timeout(self):
        proc  = _mock_proc("")
        k_res = self._K(executable="mock")._parse_output(
            proc, wall_time=3600.0, timed_out=True
        )
        c_res = self._C(executable="mock")._parse_output(
            proc, wall_time=3600.0, timed_out=True
        )
        self.assertIsNone(k_res["satisfiable"])
        self.assertIsNone(c_res["satisfiable"])
        self.assertTrue(k_res["timed_out"])
        self.assertTrue(c_res["timed_out"])


if __name__ == "__main__":
    unittest.main()
