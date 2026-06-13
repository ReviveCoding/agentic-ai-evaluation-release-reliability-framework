from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from agentic_eval_framework.utils.trace_integrity import verify_trace_fingerprint


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    scenario_id TEXT,
    status TEXT,
    release_decision TEXT,
    failure_type TEXT,
    latency_ms REAL,
    payload_json TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT,
    step_id TEXT,
    predicted_tool TEXT,
    latency_ms REAL,
    payload_json TEXT,
    PRIMARY KEY (run_id, step_id)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    run_id TEXT,
    step_id TEXT,
    tool_name TEXT,
    evidence_id TEXT,
    observation_json TEXT
);
CREATE TABLE IF NOT EXISTS evaluator_scores (
    run_id TEXT,
    evaluator_name TEXT,
    score REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_runs_decision ON runs(release_decision);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
"""


class SQLiteTraceStore:
    def __init__(self, db_path: str | Path = "outputs/traces.sqlite", commit_every: int = 1) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self.commit_every = max(1, int(commit_every))
        self._pending = 0

    def upsert_run(self, run: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM steps WHERE run_id = ?", (run["run_id"],))
            self.conn.execute("DELETE FROM tool_calls WHERE run_id = ?", (run["run_id"],))
            self.conn.execute("DELETE FROM evaluator_scores WHERE run_id = ?", (run["run_id"],))
            self.conn.execute(
                "REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run["run_id"], run["scenario_id"], run.get("status"),
                    run.get("release_decision"), run.get("failure_type"),
                    run.get("latency_ms", 0.0), json.dumps(run, ensure_ascii=False),
                ),
            )
            for step in run.get("steps", []):
                self.conn.execute(
                    "REPLACE INTO steps VALUES (?, ?, ?, ?, ?)",
                    (run["run_id"], step["step_id"], step.get("predicted_tool"), step.get("latency_ms", 0.0), json.dumps(step, ensure_ascii=False)),
                )
                obs = step.get("observation", {})
                self.conn.execute(
                    "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?)",
                    (run["run_id"], step["step_id"], step.get("predicted_tool"), obs.get("evidence_id"), json.dumps(obs, ensure_ascii=False)),
                )
            for name, score in run.get("scores", {}).items():
                self.conn.execute("INSERT INTO evaluator_scores VALUES (?, ?, ?)", (run["run_id"], name, float(score)))
            self._pending += 1
            if self._pending >= self.commit_every:
                self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if getattr(self, "conn", None) is not None:
            self.conn.commit()
            self._pending = 0

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self.conn.execute("SELECT payload_json FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def list_failed_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute("SELECT payload_json FROM runs WHERE release_decision != 'PASS'")
            rows = cur.fetchall()
        return [json.loads(row[0]) for row in rows]


    def integrity_check(self) -> str:
        with self._lock:
            row = self.conn.execute("PRAGMA quick_check").fetchone()
        return str(row[0]) if row else "unknown"

    def count_runs(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0]) if row else 0


    def payload_integrity_summary(self, limit: int | None = None) -> dict[str, int]:
        with self._lock:
            sql = "SELECT payload_json FROM runs ORDER BY rowid DESC"
            params: tuple[Any, ...] = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (max(0, int(limit)),)
            rows = self.conn.execute(sql, params).fetchall()
        valid = 0
        invalid = 0
        missing = 0
        for row in rows:
            payload = json.loads(row[0])
            if not payload.get("trace_fingerprint"):
                missing += 1
            elif verify_trace_fingerprint(payload):
                valid += 1
            else:
                invalid += 1
        return {"checked": len(rows), "valid": valid, "invalid": invalid, "missing": missing}

    def close(self) -> None:
        with self._lock:
            if getattr(self, "conn", None) is not None:
                self._flush_unlocked()
                self.conn.close()
                self.conn = None  # type: ignore[assignment]

    def __enter__(self) -> "SQLiteTraceStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
