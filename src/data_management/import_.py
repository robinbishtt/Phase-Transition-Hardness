"""Import utilities for experiment results."""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


def import_from_json(path: str) -> Dict:
    """Import data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def import_from_csv(path: str) -> List[Dict]:
    """Import data from CSV file."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def import_from_npz(path: str) -> Dict[str, np.ndarray]:
    """Import numpy arrays from NPZ file."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def import_results(
    path: str,
    format: Optional[str] = None
) -> Dict:
    """Import results from file (auto-detect format)."""
    path = Path(path)

    if format is None:
        format = path.suffix.lower().lstrip(".")

    if format == "json":
        return import_from_json(path)
    elif format in ["npz", "npy"]:
        return import_from_npz(path)
    elif format == "csv":
        return {"data": import_from_csv(path)}
    else:
        raise ValueError(f"Unsupported format: {format}")


def merge_results(
    result_paths: List[str],
    output_path: Optional[str] = None
) -> Dict:
    """Merge multiple result files."""
    merged = {
        "experiments": [],
        "arrays": {}
    }

    for path in result_paths:
        data = import_results(path)

        if "experiments" in data:
            merged["experiments"].extend(data["experiments"])

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if key not in merged["arrays"]:
                    merged["arrays"][key] = []
                merged["arrays"][key].append(value)

    for key in merged["arrays"]:
        arrays = merged["arrays"][key]
        if arrays:
            merged["arrays"][key] = np.concatenate(arrays, axis=0)

    if output_path:
        if output_path.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(merged, f, indent=2, default=str)
        elif output_path.endswith(".npz"):
            np.savez_compressed(output_path, **merged["arrays"])

    return merged


def load_experiment_batch(
    directory: str,
    pattern: str = "*.json"
) -> List[Dict]:
    """Load all experiment results matching pattern."""
    directory = Path(directory)
    results = []

    for path in directory.glob(pattern):
        try:
            data = import_results(path)
            results.append(data)
        except Exception:
            continue

    return results


def validate_imported_data(data: Dict, schema: Dict) -> bool:
    """Validate imported data against schema."""
    for key, expected_type in schema.items():
        if key not in data:
            return False

        if expected_type == "array":
            if not isinstance(data[key], (list, np.ndarray)):
                return False
        elif expected_type == "number":
            if not isinstance(data[key], (int, float, np.number)):
                return False
        elif expected_type == "string":
            if not isinstance(data[key], str):
                return False

    return True