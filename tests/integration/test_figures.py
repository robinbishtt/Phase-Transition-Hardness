"""Integration tests for figures/.

Tests each of the six figure-generating scripts by invoking their main
generation functions with minimal parameters (small DPI, PNG format) and
verifying that (a) the functions run without raising exceptions, (b) at
least one output PNG file is created in a temporary directory, and (c) the
created files are non-empty valid PNG files (IHDR magic bytes check).

All tests fall back to synthetic data — no experimental results directory
is required.  This mirrors the behaviour of the production scripts, which
include synthetic fallbacks for CI environments.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _run_main(mod, argv):
    """Invoke a script's main() with patched sys.argv, suppressing all output."""
    import io, logging
    from contextlib import redirect_stdout, redirect_stderr
    orig = sys.argv[:]
    sys.argv = [mod.__file__] + argv
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = mod.main()
    finally:
        sys.argv = orig
        logging.disable(logging.NOTSET)
    return rc if rc is not None else 0


def _is_valid_png(path: str) -> bool:
    """Return True if the file starts with the 8-byte PNG magic signature."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


class TestHardnessPlots(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_generate_runs_without_error(self):
        from figures.hardness_plots import generate_hardness_plots
        generate_hardness_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )

    def test_output_file_created(self):
        from figures.hardness_plots import generate_hardness_plots
        generate_hardness_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [f for f in os.listdir(self.tmp) if f.endswith(".png")]
        self.assertGreater(len(pngs), 0, "No PNG file was created")

    def test_output_is_valid_png(self):
        from figures.hardness_plots import generate_hardness_plots
        generate_hardness_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [os.path.join(self.tmp, f)
                for f in os.listdir(self.tmp) if f.endswith(".png")]
        for p in pngs:
            self.assertTrue(_is_valid_png(p), f"Not a valid PNG: {p}")

    def test_output_file_non_empty(self):
        from figures.hardness_plots import generate_hardness_plots
        generate_hardness_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [os.path.join(self.tmp, f)
                for f in os.listdir(self.tmp) if f.endswith(".png")]
        for p in pngs:
            self.assertGreater(os.path.getsize(p), 1000,
                                f"PNG file too small: {p}")


class TestPhaseTransitionPlots(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_generate_runs_without_error(self):
        from figures.phase_transition_plots import generate_phase_transition_plots
        generate_phase_transition_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )

    def test_output_file_created(self):
        from figures.phase_transition_plots import generate_phase_transition_plots
        generate_phase_transition_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [f for f in os.listdir(self.tmp) if f.endswith(".png")]
        self.assertGreater(len(pngs), 0)

    def test_output_is_valid_png(self):
        from figures.phase_transition_plots import generate_phase_transition_plots
        generate_phase_transition_plots(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        for f in os.listdir(self.tmp):
            if f.endswith(".png"):
                self.assertTrue(_is_valid_png(os.path.join(self.tmp, f)))


class TestScalingCollapse(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_generate_runs_without_error(self):
        from figures.scaling_collapse import generate_scaling_collapse
        generate_scaling_collapse(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )

    def test_output_file_created(self):
        from figures.scaling_collapse import generate_scaling_collapse
        generate_scaling_collapse(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [f for f in os.listdir(self.tmp) if f.endswith(".png")]
        self.assertGreater(len(pngs), 0)

    def test_output_is_valid_png(self):
        from figures.scaling_collapse import generate_scaling_collapse
        generate_scaling_collapse(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        for f in os.listdir(self.tmp):
            if f.endswith(".png"):
                self.assertTrue(_is_valid_png(os.path.join(self.tmp, f)))


class TestLandscapeVisuals(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_generate_runs_without_error(self):
        from figures.landscape_visuals import generate_landscape_visuals
        generate_landscape_visuals(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )

    def test_output_files_created(self):
        from figures.landscape_visuals import generate_landscape_visuals
        generate_landscape_visuals(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [f for f in os.listdir(self.tmp) if f.endswith(".png")]
        self.assertGreater(len(pngs), 0)

    def test_all_outputs_valid_png(self):
        from figures.landscape_visuals import generate_landscape_visuals
        generate_landscape_visuals(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        for f in os.listdir(self.tmp):
            if f.endswith(".png"):
                self.assertTrue(_is_valid_png(os.path.join(self.tmp, f)))


class TestGenerateAllFigures(unittest.TestCase):
    """figures/generate_all_figures.py — orchestrates all figure scripts via main()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _run(self):
        import figures.generate_all_figures as m
        _run_main(m, [
            "--results_dir", self.tmp,
            "--output_dir",  self.tmp,
            "--format",      "png",
            "--dpi",         "50",
        ])

    def test_generate_all_runs_without_error(self):
        self._run()

    def test_multiple_png_files_created(self):
        self._run()
        pngs = [f for f in os.listdir(self.tmp) if f.endswith(".png")]
        self.assertGreaterEqual(len(pngs), 3,
                                 f"Expected at least 3 PNGs, got {len(pngs)}")

    def test_all_outputs_valid_png(self):
        self._run()
        for f in os.listdir(self.tmp):
            if f.endswith(".png"):
                p = os.path.join(self.tmp, f)
                self.assertTrue(_is_valid_png(p), f"Not a valid PNG: {f}")


class TestExtendedDataFigures(unittest.TestCase):
    """figures/extended_data_figures.py — all suptitles must have no figure numbers."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_generate_runs_without_error(self):
        from figures.extended_data_figures import generate_extended_data_figures
        generate_extended_data_figures(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )

    def test_output_files_created(self):
        from figures.extended_data_figures import generate_extended_data_figures
        generate_extended_data_figures(
            results_dir=self.tmp, output_dir=self.tmp,
            fmt="png", dpi=50,
        )
        pngs = [f for f in os.listdir(self.tmp) if f.endswith(".png")]
        self.assertGreater(len(pngs), 0)

    def test_source_code_has_no_figure_number_in_suptitle(self):
        import re
        with open("figures/extended_data_figures.py") as f:
            src = f.read()
        # The pattern "Extended Data Fig. N" must not appear inside suptitle() calls
        matches = re.findall(
            r'suptitle\([^)]*Extended Data Fig\.\s*\d', src
        )
        self.assertEqual(
            len(matches), 0,
            msg=f"Figure numbers found in suptitle calls: {matches}",
        )


if __name__ == "__main__":
    unittest.main()
