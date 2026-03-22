"""SQLite database for experiment result management."""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ExperimentDatabase:
    """SQLite database for storing and querying experiment results.

    NOTE: The hostname and git_commit fields have been intentionally removed
    from the schema to prevent accidental identity leakage in double-blind
    review submissions.
    """

    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    results TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    n INTEGER,
                    alpha REAL,
                    k INTEGER,
                    seed INTEGER,
                    satisfiable INTEGER,
                    runtime REAL,
                    decisions INTEGER,
                    hardness REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_instances_exp
                ON instances(experiment_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_instances_params
                ON instances(n, alpha, k)
            """)

            conn.commit()

    def insert_experiment(
        self,
        name: str,
        experiment_type: str,
        parameters: Dict,
        results: Optional[Dict] = None,
    ) -> int:
        """Insert new experiment record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                (name, experiment_type, parameters, results, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    experiment_type,
                    json.dumps(parameters),
                    json.dumps(results) if results else None,
                    datetime.now().isoformat(),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def insert_instance(
        self,
        experiment_id: int,
        n: int,
        alpha: float,
        k: int,
        seed: int,
        satisfiable: Optional[bool],
        runtime: Optional[float],
        decisions: Optional[int],
        hardness: Optional[float]
    ) -> int:
        """Insert instance result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO instances
                (experiment_id, n, alpha, k, seed, satisfiable, runtime, decisions, hardness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id, n, alpha, k, seed,
                    1 if satisfiable else 0 if satisfiable is False else None,
                    runtime, decisions, hardness,
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "experiment_type": row["experiment_type"],
                "parameters": json.loads(row["parameters"]),
                "results": json.loads(row["results"]) if row["results"] else None,
                "timestamp": row["timestamp"],
            }

    def get_instances(
        self,
        experiment_id: Optional[int] = None,
        n: Optional[int] = None,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
    ) -> List[Dict]:
        """Query instances with filters."""
        query = "SELECT * FROM instances WHERE 1=1"
        params: List[Any] = []

        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if n is not None:
            query += " AND n = ?"
            params.append(n)
        if alpha is not None:
            query += " AND alpha = ?"
            params.append(alpha)
        if k is not None:
            query += " AND k = ?"
            params.append(k)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [
                {
                    "id": r["id"],
                    "experiment_id": r["experiment_id"],
                    "n": r["n"],
                    "alpha": r["alpha"],
                    "k": r["k"],
                    "seed": r["seed"],
                    "satisfiable": r["satisfiable"],
                    "runtime": r["runtime"],
                    "decisions": r["decisions"],
                    "hardness": r["hardness"],
                }
                for r in cursor.fetchall()
            ]

    def get_statistics(self, experiment_id: int) -> Dict:
        """Get aggregate statistics for experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    AVG(hardness)   as mean_hardness,
                    AVG(runtime)    as mean_runtime,
                    AVG(decisions)  as mean_decisions,
                    SUM(CASE WHEN satisfiable = 1 THEN 1 ELSE 0 END) as sat_count
                FROM instances WHERE experiment_id = ?
                """,
                (experiment_id,)
            )
            row = cursor.fetchone()
            return {
                "total_instances": row[0],
                "mean_hardness":   row[1],
                "mean_runtime":    row[2],
                "mean_decisions":  row[3],
                "sat_count":       row[4],
                "sat_fraction":    row[4] / row[0] if row[0] > 0 else 0,
            }

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete experiment and associated instances."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM instances WHERE experiment_id = ?", (experiment_id,)
            )
            cursor = conn.execute(
                "DELETE FROM experiments WHERE id = ?", (experiment_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, name, experiment_type, timestamp "
                "FROM experiments ORDER BY timestamp DESC"
            )
            return [
                {
                    "id":   r["id"],
                    "name": r["name"],
                    "type": r["experiment_type"],
                    "timestamp": r["timestamp"],
                }
                for r in cursor.fetchall()
            ]
