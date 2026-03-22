"""Export utilities for experiment results."""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


def export_to_json(
    data: Dict,
    output_path: str,
    indent: int = 2
) -> None:
    """Export data to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def export_to_csv(
    data: List[Dict],
    output_path: str,
    fieldnames: Optional[List[str]] = None
) -> None:
    """Export list of dictionaries to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def export_to_npz(
    output_path: str,
    **arrays
) -> None:
    """Export numpy arrays to NPZ file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def export_results(
    results: Dict,
    output_dir: str,
    basename: str,
    formats: List[str] = None
) -> Dict[str, str]:
    """Export results in multiple formats."""
    formats = formats or ["json", "npz"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    if "json" in formats:
        json_path = output_dir / f"{basename}.json"
        export_to_json(results, json_path)
        exported["json"] = str(json_path)

    if "npz" in formats:
        npz_path = output_dir / f"{basename}.npz"
        arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
        if arrays:
            export_to_npz(npz_path, **arrays)
            exported["npz"] = str(npz_path)

    return exported


def export_summary_table(
    results_list: List[Dict],
    output_path: str
) -> None:
    """Export summary table of multiple experiments."""
    rows = []

    for result in results_list:
        row = {
            "experiment": result.get("name", "unknown"),
            "n": result.get("n", ""),
            "alpha": result.get("alpha", ""),
            "k": result.get("k", 3),
            "n_instances": result.get("n_instances", ""),
            "mean_hardness": result.get("mean_hardness", ""),
            "std_hardness": result.get("std_hardness", ""),
        }
        rows.append(row)

    export_to_csv(rows, output_path)


def export_latex_table(
    data: List[Dict],
    output_path: str,
    columns: List[str],
    headers: Optional[List[str]] = None
) -> None:
    """Export data as LaTeX table."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if headers is None:
        headers = columns

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{" + "c" * len(columns) + "}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline"
    ]

    for row in data:
        values = [str(row.get(col, "")) for col in columns]
        lines.append(" & ".join(values) + " \\\\")

    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))