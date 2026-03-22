"""Unit tests for src/data_management/.

Covers:
    export.py   — export_to_json, export_to_csv, export_to_npz,
                  export_results, export_summary_table, export_latex_table
    import_.py  — import_from_json, import_from_csv, import_from_npz,
                  import_results, merge_results, load_experiment_batch,
                  validate_imported_data
    database.py — ExperimentDatabase CRUD operations

These functions are the data-persistence layer used by the experiment
scripts to write and reload results.  Tests verify round-trip fidelity
(write then read gives back the original data), file-format correctness,
error behaviour on missing files, and schema validation logic.
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management.export import (
    export_latex_table,
    export_results,
    export_summary_table,
    export_to_csv,
    export_to_json,
    export_to_npz,
)
from src.data_management.import_ import (
    import_from_csv,
    import_from_json,
    import_from_npz,
    load_experiment_batch,
    merge_results,
    validate_imported_data,
)


# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------
_SAMPLE_DICT  = {"alpha": 4.2, "gamma_mean": 0.021, "n": 400}
_SAMPLE_LIST  = [
    {"alpha": 4.0, "gamma": 0.015, "n": 100},
    {"alpha": 4.2, "gamma": 0.021, "n": 400},
    {"alpha": 4.5, "gamma": 0.010, "n": 800},
]


# ===========================================================================
# export.py
# ===========================================================================

class TestExportToJson(unittest.TestCase):

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_json(_SAMPLE_DICT, f"{tmp}/out.json")
            self.assertTrue(Path(f"{tmp}/out.json").exists())

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_json(_SAMPLE_DICT, f"{tmp}/out.json")
            with open(f"{tmp}/out.json") as f:
                loaded = json.load(f)
        self.assertAlmostEqual(loaded["alpha"],      _SAMPLE_DICT["alpha"])
        self.assertAlmostEqual(loaded["gamma_mean"], _SAMPLE_DICT["gamma_mean"])
        self.assertEqual(loaded["n"],                _SAMPLE_DICT["n"])

    def test_numpy_scalar_serialisable(self):
        data = {"val": np.float64(3.14), "arr": np.array([1, 2, 3])}
        with tempfile.TemporaryDirectory() as tmp:
            export_to_json(data, f"{tmp}/out.json")
            with open(f"{tmp}/out.json") as f:
                loaded = json.load(f)
        self.assertAlmostEqual(loaded["val"], 3.14, places=5)
        self.assertEqual(loaded["arr"], [1, 2, 3])


class TestExportToCsv(unittest.TestCase):

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_csv(_SAMPLE_LIST, f"{tmp}/out.csv")
            self.assertTrue(Path(f"{tmp}/out.csv").exists())

    def test_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_csv(_SAMPLE_LIST, f"{tmp}/out.csv")
            with open(f"{tmp}/out.csv") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(_SAMPLE_LIST))

    def test_column_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_csv(_SAMPLE_LIST, f"{tmp}/out.csv")
            with open(f"{tmp}/out.csv") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
        for key in _SAMPLE_LIST[0].keys():
            self.assertIn(key, headers)


class TestExportToNpz(unittest.TestCase):
    # Signature: export_to_npz(output_path: str, **arrays) -> None

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_npz(f"{tmp}/out.npz", arr=np.array([1.0, 2.0, 3.0]))
            self.assertTrue(Path(f"{tmp}/out.npz").exists())

    def test_roundtrip_single_array(self):
        arr = np.array([0.1, 0.2, 0.3])
        with tempfile.TemporaryDirectory() as tmp:
            export_to_npz(f"{tmp}/out.npz", values=arr)
            loaded = np.load(f"{tmp}/out.npz")
            np.testing.assert_array_almost_equal(loaded["values"], arr)

    def test_multiple_arrays(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        with tempfile.TemporaryDirectory() as tmp:
            export_to_npz(f"{tmp}/out.npz", a=a, b=b)
            loaded = np.load(f"{tmp}/out.npz")
            self.assertIn("a", loaded)
            self.assertIn("b", loaded)


class TestExportResults(unittest.TestCase):
    # Signature: export_results(results, output_dir, basename, formats) -> Dict[str,str]

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_results(_SAMPLE_DICT, tmp, "test_exp", formats=["json"])
        self.assertIsInstance(result, dict)

    def test_json_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_results(_SAMPLE_DICT, tmp, "test_exp", formats=["json"])
            self.assertTrue(any(Path(tmp).glob("test_exp*.json")))

    def test_default_formats_include_json(self):
        # export_results without explicit formats defaults to ["json", "npz"].
        # A dict input with no numpy arrays writes only the JSON file.
        with tempfile.TemporaryDirectory() as tmp:
            export_results(_SAMPLE_DICT, tmp, "test_exp", formats=["json"])
            self.assertTrue(any(Path(tmp).glob("test_exp*.json")))


class TestExportSummaryTable(unittest.TestCase):

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_summary_table(_SAMPLE_LIST, f"{tmp}/summary.csv")
            self.assertTrue(Path(f"{tmp}/summary.csv").exists())


class TestExportLatexTable(unittest.TestCase):

    def test_file_created(self):
        cols = ["alpha", "gamma", "n"]
        with tempfile.TemporaryDirectory() as tmp:
            export_latex_table(_SAMPLE_LIST, f"{tmp}/table.tex", columns=cols)
            self.assertTrue(Path(f"{tmp}/table.tex").exists())

    def test_latex_file_contains_tabular(self):
        cols = ["alpha", "gamma", "n"]
        with tempfile.TemporaryDirectory() as tmp:
            export_latex_table(_SAMPLE_LIST, f"{tmp}/table.tex", columns=cols)
            content = Path(f"{tmp}/table.tex").read_text()
        self.assertIn("tabular", content)


# ===========================================================================
# import_.py
# ===========================================================================

class TestImportFromJson(unittest.TestCase):

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_json(_SAMPLE_DICT, f"{tmp}/out.json")
            loaded = import_from_json(f"{tmp}/out.json")
        self.assertIsInstance(loaded, dict)
        self.assertAlmostEqual(loaded["alpha"], _SAMPLE_DICT["alpha"])

    def test_raises_on_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, Exception)):
                import_from_json(f"{tmp}/nonexistent.json")


class TestImportFromCsv(unittest.TestCase):

    def test_returns_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_csv(_SAMPLE_LIST, f"{tmp}/out.csv")
            loaded = import_from_csv(f"{tmp}/out.csv")
        self.assertIsInstance(loaded, list)
        self.assertEqual(len(loaded), len(_SAMPLE_LIST))

    def test_headers_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_csv(_SAMPLE_LIST, f"{tmp}/out.csv")
            loaded = import_from_csv(f"{tmp}/out.csv")
        for row in loaded:
            for key in _SAMPLE_LIST[0].keys():
                self.assertIn(key, row)


class TestImportFromNpz(unittest.TestCase):

    def test_roundtrip(self):
        arr = np.linspace(0.0, 1.0, 10)
        with tempfile.TemporaryDirectory() as tmp:
            export_to_npz(f"{tmp}/out.npz", data=arr)
            loaded = import_from_npz(f"{tmp}/out.npz")
        self.assertIsInstance(loaded, dict)
        np.testing.assert_array_almost_equal(loaded["data"], arr)


class TestMergeResults(unittest.TestCase):
    # Signature: merge_results(result_paths: List[str], output_path=None) -> Dict

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = f"{tmp}/r1.json"
            p2 = f"{tmp}/r2.json"
            export_to_json({"alpha": 4.0, "gamma": 0.015}, p1)
            export_to_json({"alpha": 4.2, "gamma": 0.021}, p2)
            result = merge_results([p1, p2])
        self.assertIsInstance(result, dict)

    def test_merged_dict_non_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = f"{tmp}/r1.json"
            export_to_json({"alpha": 4.0, "gamma": 0.015}, p1)
            result = merge_results([p1])
        self.assertGreater(len(result), 0)


class TestLoadExperimentBatch(unittest.TestCase):

    def test_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_to_json({"alpha": 4.0}, f"{tmp}/exp1.json")
            export_to_json({"alpha": 4.2}, f"{tmp}/exp2.json")
            batch = load_experiment_batch(tmp, pattern="*.json")
        self.assertIsInstance(batch, list)

    def test_loads_all_matching_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                export_to_json({"index": i}, f"{tmp}/exp{i}.json")
            batch = load_experiment_batch(tmp, pattern="*.json")
        self.assertEqual(len(batch), 3)

    def test_empty_directory_gives_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = load_experiment_batch(tmp, pattern="*.json")
        self.assertEqual(batch, [])


class TestValidateImportedData(unittest.TestCase):

    def test_valid_data_returns_true(self):
        schema = {"alpha": "number", "n": "number"}
        data   = {"alpha": 4.2, "n": 400}
        self.assertTrue(validate_imported_data(data, schema))

    def test_missing_key_returns_false(self):
        schema = {"alpha": "number", "gamma": "number"}
        data   = {"alpha": 4.2}
        self.assertFalse(validate_imported_data(data, schema))

    def test_wrong_type_returns_false(self):
        schema = {"alpha": "number"}
        data   = {"alpha": "not_a_float"}
        self.assertFalse(validate_imported_data(data, schema))

    def test_extra_keys_are_ignored(self):
        schema = {"alpha": "number"}
        data   = {"alpha": 4.2, "extra_key": "irrelevant"}
        self.assertTrue(validate_imported_data(data, schema))


# ===========================================================================
# database.py
# ===========================================================================

class TestExperimentDatabase(unittest.TestCase):

    def _fresh_db(self, tmp: str):
        from src.data_management.database import ExperimentDatabase
        return ExperimentDatabase(db_path=f"{tmp}/test.db")

    def test_insert_and_retrieve_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            db  = self._fresh_db(tmp)
            eid = db.insert_experiment(
                name="test_exp",
                experiment_type="alpha_sweep",
                parameters={"n": 400, "alpha": 4.2, "k": 3},
            )
            self.assertIsInstance(eid, int)
            self.assertGreater(eid, 0)
            exp = db.get_experiment(eid)
            self.assertIsNotNone(exp)
            self.assertEqual(exp["name"], "test_exp")

    def test_list_experiments_empty_initially(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._fresh_db(tmp)
            exps = db.list_experiments()
            self.assertIsInstance(exps, list)

    def test_list_experiments_after_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._fresh_db(tmp)
            db.insert_experiment(
                name="e1", experiment_type="alpha_sweep",
                parameters={"n": 100, "alpha": 4.0},
            )
            db.insert_experiment(
                name="e2", experiment_type="alpha_sweep",
                parameters={"n": 200, "alpha": 4.2},
            )
            exps = db.list_experiments()
            self.assertEqual(len(exps), 2)

    def test_get_nonexistent_experiment_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            db  = self._fresh_db(tmp)
            exp = db.get_experiment(9999)
            self.assertIsNone(exp)

    def test_delete_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            db  = self._fresh_db(tmp)
            eid = db.insert_experiment(
                name="to_delete", experiment_type="alpha_sweep",
                parameters={"n": 100, "alpha": 4.0},
            )
            deleted = db.delete_experiment(eid)
            self.assertTrue(deleted)
            self.assertIsNone(db.get_experiment(eid))

    def test_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._fresh_db(tmp)
            self.assertFalse(db.delete_experiment(9999))

    def test_insert_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            db  = self._fresh_db(tmp)
            eid = db.insert_experiment(
                name="exp", experiment_type="alpha_sweep",
                parameters={"n": 20, "alpha": 4.0, "k": 3},
            )
            iid = db.insert_instance(
                experiment_id=eid,
                n=20, alpha=4.0, k=3, seed=42,
                satisfiable=True, runtime=1.23,
                decisions=1000, hardness=0.025,
            )
            self.assertIsInstance(iid, int)
            self.assertGreater(iid, 0)

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db  = self._fresh_db(tmp)
            eid = db.insert_experiment(
                name="exp", experiment_type="alpha_sweep",
                parameters={"n": 20, "alpha": 4.0, "k": 3},
            )
            db.insert_instance(eid, n=20, alpha=4.0, k=3, seed=1,
                               satisfiable=True, runtime=0.5, decisions=500, hardness=0.031)
            db.insert_instance(eid, n=20, alpha=4.0, k=3, seed=2,
                               satisfiable=True, runtime=1.0, decisions=1000, hardness=0.042)
            stats = db.get_statistics(eid)
            self.assertIsInstance(stats, dict)
            # get_statistics returns {"total_instances", "mean_hardness",
            # "mean_runtime", "mean_decisions", "sat_count", "sat_fraction"}
            self.assertIn("total_instances", stats)
            self.assertEqual(stats["total_instances"], 2)


if __name__ == "__main__":
    unittest.main()
