"""Internal SQLite database layer for Stock_app.

Stores all specific stock predictions, market screening calls, and screen hits
separately from the ML_stock repository.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import DB_PATH
from cone_utils import to_yyyy_mm_dd


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data before serializing to SQLite JSON columns."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _sanitize_for_json(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return _sanitize_for_json(obj.tolist())
        except Exception:
            pass
    return obj


def _safe_json_dumps(obj: Any) -> str:
    """Serializes object to strict JSON with allow_nan=False after sanitization."""
    sanitized = _sanitize_for_json(obj)
    return json.dumps(sanitized, ensure_ascii=False, allow_nan=False)


class StockAppDatabase:
    """Thread-safe SQLite storage for predictions and screening runs."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS screening_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    universe TEXT NOT NULL,
                    top_n INTEGER NOT NULL,
                    min_price REAL NOT NULL,
                    interval TEXT NOT NULL DEFAULT '1d',
                    total_candidates INTEGER NOT NULL DEFAULT 0,
                    total_screened INTEGER NOT NULL DEFAULT 0,
                    checkpoint_path TEXT,
                    duration_sec REAL DEFAULT 0.0,
                    results_json TEXT NOT NULL,
                    run_id TEXT DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    sector TEXT,
                    universe TEXT,
                    is_index INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    anchor_date TEXT NOT NULL,
                    anchor_close REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    screen_run_id INTEGER,
                    screen_category TEXT,
                    model_type TEXT,
                    checkpoint_path TEXT,
                    run_id TEXT DEFAULT 'default',
                    lookback INTEGER NOT NULL DEFAULT 60,
                    num_samples INTEGER NOT NULL DEFAULT 1,
                    mc_dropout INTEGER NOT NULL DEFAULT 0,
                    market_resampling INTEGER NOT NULL DEFAULT 1,
                    milestones_json TEXT NOT NULL,
                    quantiles_json TEXT NOT NULL,
                    cone_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    FOREIGN KEY (screen_run_id) REFERENCES screening_runs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_ticker
                    ON predictions(ticker, anchor_date DESC, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                    ON predictions(created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_predictions_source
                    ON predictions(source, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_predictions_screen_run
                    ON predictions(screen_run_id);

                CREATE INDEX IF NOT EXISTS idx_screening_runs_created_at
                    ON screening_runs(created_at DESC);

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Automatic schema migration: ensure run_id column exists
            pred_cols = [c[1] for c in conn.execute("PRAGMA table_info(predictions)").fetchall()]
            if "run_id" not in pred_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN run_id TEXT DEFAULT 'default';")

            screen_cols = [c[1] for c in conn.execute("PRAGMA table_info(screening_runs)").fetchall()]
            if "run_id" not in screen_cols:
                conn.execute("ALTER TABLE screening_runs ADD COLUMN run_id TEXT DEFAULT 'default';")

            # Unconditionally backfill any rows with NULL run_id to 'default'
            conn.execute("UPDATE predictions SET run_id = 'default' WHERE run_id IS NULL;")
            conn.execute("UPDATE screening_runs SET run_id = 'default' WHERE run_id IS NULL;")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_screening_runs_run_id ON screening_runs(run_id);")
            conn.commit()

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves a persistent setting value from the app_settings table."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Stores or updates a persistent setting in the app_settings table."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
            conn.commit()

    @staticmethod
    def _row_to_prediction(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        return {
            "id": d["id"],
            "ticker": d["ticker"],
            "name": d.get("name") or d["ticker"],
            "sector": d.get("sector") or "Equities",
            "universe": d.get("universe") or "custom",
            "is_index": bool(d.get("is_index", 0)),
            "created_at": d["created_at"],
            "anchor_date": to_yyyy_mm_dd(d["anchor_date"]),
            "anchor_close": float(d["anchor_close"]),
            "source": d["source"],
            "screen_run_id": d.get("screen_run_id"),
            "screen_category": d.get("screen_category"),
            "model_type": d.get("model_type") or "patchtst",
            "checkpoint_path": d.get("checkpoint_path") or "",
            "run_id": d.get("run_id") or "default",
            "lookback": int(d.get("lookback", 60)),
            "num_samples": int(d.get("num_samples", 1)),
            "mc_dropout": bool(d.get("mc_dropout", 0)),
            "market_resampling": bool(d.get("market_resampling", 1)),
            "milestones": json.loads(d["milestones_json"]),
            "quantiles": json.loads(d["quantiles_json"]),
            "cone": json.loads(d["cone_json"]),
            "summary": json.loads(d["summary_json"]),
        }

    def save_prediction(
        self,
        ticker: str,
        anchor_date: str,
        anchor_close: float,
        milestones: List[int],
        quantiles: List[float],
        cone: Dict[str, Any],
        summary: Dict[str, Any],
        name: Optional[str] = None,
        sector: Optional[str] = None,
        universe: Optional[str] = None,
        is_index: bool = False,
        source: str = "manual",
        screen_run_id: Optional[int] = None,
        screen_category: Optional[str] = None,
        model_type: str = "market_query",
        checkpoint_path: str = "",
        run_id: Optional[str] = None,
        lookback: int = 60,
        num_samples: int = 1,
        mc_dropout: bool = False,
        market_resampling: bool = True,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inserts a prediction record and returns the stored dict."""
        ticker_clean = ticker.upper().strip()
        now_iso = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_identifier = run_id or "default"

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    ticker, name, sector, universe, is_index,
                    created_at, anchor_date, anchor_close,
                    source, screen_run_id, screen_category,
                    model_type, checkpoint_path, run_id, lookback,
                    num_samples, mc_dropout, market_resampling,
                    milestones_json, quantiles_json, cone_json, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker_clean,
                    name or ticker_clean,
                    sector or "Equities",
                    universe or "custom",
                    1 if is_index else 0,
                    now_iso,
                    to_yyyy_mm_dd(anchor_date),
                    float(anchor_close),
                    source,
                    screen_run_id,
                    screen_category,
                    model_type,
                    checkpoint_path,
                    run_identifier,
                    int(lookback),
                    int(num_samples),
                    1 if mc_dropout else 0,
                    1 if market_resampling else 0,
                    _safe_json_dumps(milestones),
                    _safe_json_dumps(quantiles),
                    _safe_json_dumps(cone),
                    _safe_json_dumps(summary),
                ),
            )
            pred_id = cur.lastrowid
            conn.commit()

            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
            return self._row_to_prediction(row)

    def save_screening_run(
        self,
        universe: str,
        top_n: int,
        min_price: float,
        interval: str,
        total_candidates: int,
        total_screened: int,
        checkpoint_path: str,
        duration_sec: float,
        results_payload: Dict[str, Any],
        hit_predictions: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves a screening run and stores prediction records for its hits."""
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_identifier = run_id or "default"

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO screening_runs (
                    created_at, universe, top_n, min_price, interval,
                    total_candidates, total_screened, checkpoint_path, run_id,
                    duration_sec, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    universe,
                    int(top_n),
                    float(min_price),
                    interval,
                    int(total_candidates),
                    int(total_screened),
                    checkpoint_path,
                    run_identifier,
                    round(float(duration_sec), 3),
                    _safe_json_dumps(results_payload),
                ),
            )
            screening_db_id = cur.lastrowid

            # Insert associated screen-hit predictions in the same transaction
            for hp in hit_predictions:
                conn.execute(
                    """
                    INSERT INTO predictions (
                        ticker, name, sector, universe, is_index,
                        created_at, anchor_date, anchor_close,
                        source, screen_run_id, screen_category,
                        model_type, checkpoint_path, run_id, lookback,
                        num_samples, mc_dropout, market_resampling,
                        milestones_json, quantiles_json, cone_json, summary_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hp["ticker"].upper().strip(),
                        hp.get("name") or hp["ticker"],
                        hp.get("sector") or "Equities",
                        hp.get("universe") or universe,
                        1 if hp.get("is_index") else 0,
                        now_iso,
                        to_yyyy_mm_dd(hp["anchor_date"]),
                        float(hp["anchor_close"]),
                        "screen",
                        screening_db_id,
                        hp.get("screen_category", "screen"),
                        hp.get("model_type", "market_query"),
                        checkpoint_path,
                        run_identifier,
                        int(hp.get("lookback", 60)),
                        1,
                        0,
                        0,
                        _safe_json_dumps(hp["milestones"]),
                        _safe_json_dumps(hp["quantiles"]),
                        _safe_json_dumps(hp["cone"]),
                        _safe_json_dumps(hp["summary"]),
                    ),
                )

            conn.commit()
            return self.get_screening_run(screening_db_id)  # type: ignore

    def get_stock_predictions(
        self, ticker: str, limit: int = 200, model_run: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns all stored predictions (manual + screen hits) for a stock, optionally filtered by model_run."""
        ticker_clean = ticker.upper().strip()
        clauses = ["ticker = ?"]
        params: List[Any] = [ticker_clean]
        if model_run and model_run != "all":
            clauses.append("run_id = ?")
            params.append(model_run)
        where_sql = f"WHERE {' AND '.join(clauses)}"
        params.append(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM predictions
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [self._row_to_prediction(r) for r in rows]

    def get_prediction_by_id(self, pred_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (int(pred_id),)
            ).fetchone()
            return self._row_to_prediction(row) if row else None

    def delete_prediction(self, pred_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM predictions WHERE id = ?", (int(pred_id),)
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_screening_run(self, run_id: int) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM predictions WHERE screen_run_id = ?", (int(run_id),)
            )
            cur = conn.execute(
                "DELETE FROM screening_runs WHERE id = ?", (int(run_id),)
            )
            conn.commit()
            return cur.rowcount > 0

    def get_tracked_stocks(self, model_run: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns a summary list of all stocks that currently have predictions or screen hits."""
        clauses = []
        params: List[Any] = []
        if model_run and model_run != "all":
            clauses.append("run_id = ?")
            params.append(model_run)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock, self._connect() as conn:
            query = f"""
            WITH base AS (
                SELECT *
                FROM predictions
                {where_sql}
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker
                        ORDER BY anchor_date DESC, id DESC
                    ) AS rn
                FROM base
            ),
            stats AS (
                SELECT
                    ticker,
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN source = 'manual' THEN 1 ELSE 0 END) AS manual_count,
                    SUM(CASE WHEN source = 'screen' THEN 1 ELSE 0 END) AS screen_hit_count,
                    GROUP_CONCAT(DISTINCT CASE WHEN screen_category IS NOT NULL AND screen_category != '' THEN screen_category ELSE NULL END) AS screen_categories_csv,
                    MAX(created_at) AS latest_created_at
                FROM base
                GROUP BY ticker
            )
            SELECT
                r.*,
                s.total_predictions,
                s.manual_count,
                s.screen_hit_count,
                s.screen_categories_csv
            FROM ranked r
            JOIN stats s ON r.ticker = s.ticker
            WHERE r.rn = 1
            ORDER BY s.latest_created_at DESC, r.ticker ASC
            """
            rows = conn.execute(query, tuple(params)).fetchall()

            tracked: List[Dict[str, Any]] = []
            for r in rows:
                latest_pred = self._row_to_prediction(r)
                raw_cats = r["screen_categories_csv"]
                categories = [c.strip() for c in raw_cats.split(",") if c.strip()] if raw_cats else []

                tracked.append(
                    {
                        "ticker": latest_pred["ticker"],
                        "name": latest_pred["name"],
                        "sector": latest_pred["sector"],
                        "universe": latest_pred["universe"],
                        "is_index": latest_pred["is_index"],
                        "total_predictions": int(r["total_predictions"]),
                        "manual_count": int(r["manual_count"]),
                        "screen_hit_count": int(r["screen_hit_count"]),
                        "screen_categories": categories,
                        "latest_prediction_id": latest_pred["id"],
                        "latest_created_at": latest_pred["created_at"],
                        "latest_anchor_date": latest_pred["anchor_date"],
                        "latest_anchor_close": latest_pred["anchor_close"],
                        "latest_source": latest_pred["source"],
                        "latest_screen_category": latest_pred["screen_category"],
                        "latest_run_id": latest_pred.get("run_id") or "default",
                        "latest_summary": latest_pred["summary"],
                    }
                )
            return tracked

    def get_prediction_history(
        self,
        limit: int = 100,
        source: Optional[str] = None,
        ticker: Optional[str] = None,
        model_run: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns history of specific prediction calls (and/or screen hit predictions)."""
        clauses = []
        params: List[Any] = []
        if source and source != "all":
            clauses.append("source = ?")
            params.append(source)
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker.upper().strip())
        if model_run and model_run != "all":
            clauses.append("run_id = ?")
            params.append(model_run)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM predictions
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [self._row_to_prediction(r) for r in rows]

    def get_screening_history(
        self, limit: int = 50, model_run: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns summary list of all historical screening calls."""
        clauses = []
        params: List[Any] = []
        if model_run and model_run != "all":
            clauses.append("run_id = ?")
            params.append(model_run)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id, created_at, universe, top_n, min_price, interval,
                    total_candidates, total_screened, checkpoint_path, run_id,
                    duration_sec, results_json
                FROM screening_runs
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

            history: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                res = json.loads(d.pop("results_json", "{}"))
                top_bull = res.get("top_bullish", [])
                top_bear = res.get("top_bearish", [])
                top_vol = res.get("top_volatility", [])
                indexes = res.get("indexes", [])
                history.append(
                    {
                        **d,
                        "run_id": d.get("run_id") or "default",
                        "bullish_count": len(top_bull),
                        "bearish_count": len(top_bear),
                        "volatility_count": len(top_vol),
                        "index_count": len(indexes),
                        "top_bullish_preview": [
                            {
                                "ticker": x["ticker"],
                                "name": x.get("name", x["ticker"]),
                                "cum2d_ret_50": x.get("cum2d_ret_50", 0.0),
                            }
                            for x in top_bull[:3]
                        ],
                        "top_bearish_preview": [
                            {
                                "ticker": x["ticker"],
                                "name": x.get("name", x["ticker"]),
                                "cum2d_ret_50": x.get("cum2d_ret_50", 0.0),
                            }
                            for x in top_bear[:3]
                        ],
                    }
                )
            return history

    def get_screening_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Returns full details of a specific screening call."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM screening_runs WHERE id = ?", (int(run_id),)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            res = json.loads(d.pop("results_json", "{}"))
            return {
                **d,
                "run_id": d.get("run_id") or "default",
                "results": res,
            }

    def get_counts(self, model_run: Optional[str] = None) -> Dict[str, int]:
        clauses = []
        params: List[Any] = []
        if model_run and model_run != "all":
            clauses.append("run_id = ?")
            params.append(model_run)
        where_pred = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        where_screen = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock, self._connect() as conn:
            total_preds = conn.execute(
                f"SELECT COUNT(*) FROM predictions {where_pred}", tuple(params)
            ).fetchone()[0]
            manual_clauses = ["source = 'manual'"] + clauses
            manual_preds = conn.execute(
                f"SELECT COUNT(*) FROM predictions WHERE {' AND '.join(manual_clauses)}",
                tuple(params),
            ).fetchone()[0]
            screen_clauses = ["source = 'screen'"] + clauses
            screen_preds = conn.execute(
                f"SELECT COUNT(*) FROM predictions WHERE {' AND '.join(screen_clauses)}",
                tuple(params),
            ).fetchone()[0]
            screen_runs = conn.execute(
                f"SELECT COUNT(*) FROM screening_runs {where_screen}", tuple(params)
            ).fetchone()[0]
            tracked_stocks = conn.execute(
                f"SELECT COUNT(DISTINCT ticker) FROM predictions {where_pred}",
                tuple(params),
            ).fetchone()[0]
            return {
                "total_predictions": int(total_preds),
                "manual_predictions": int(manual_preds),
                "screen_hit_predictions": int(screen_preds),
                "screening_runs": int(screen_runs),
                "tracked_stocks": int(tracked_stocks),
            }
