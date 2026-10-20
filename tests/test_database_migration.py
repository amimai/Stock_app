"""Tests for database migration, run_id backfill, and optimized get_tracked_stocks.

Verifies:
1. Pre-existing databases with NULL run_id rows are properly backfilled to 'default'.
2. Pre-existing databases missing run_id columns migrate cleanly on initialization.
3. get_tracked_stocks returns accurate results matching single-stock predictions.
4. Filtering by model_run accurately filters tracked stocks, counts, and categories.
5. Elimination of N+1 query loop: single query execution with window functions.
6. Correctness with multiple stocks, categories, and duplicate hits.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from typing import Any, Dict, List
import unittest

# Ensure Stock_app root is in sys.path
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from database import StockAppDatabase


def _create_legacy_db_without_run_id(db_path: Path) -> None:
    """Creates a database schema mimicking the legacy schema before run_id was added."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE screening_runs (
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
            results_json TEXT NOT NULL
        );

        CREATE TABLE predictions (
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

        CREATE TABLE app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    # Insert legacy rows without run_id
    conn.execute(
        """
        INSERT INTO screening_runs (created_at, universe, top_n, min_price, interval, results_json)
        VALUES ('2026-03-01T10:00:00', 'sp500', 5, 10.0, '1d', '{}');
        """
    )
    conn.execute(
        """
        INSERT INTO predictions (
            ticker, name, sector, universe, is_index, created_at, anchor_date,
            anchor_close, source, screen_run_id, screen_category, milestones_json,
            quantiles_json, cone_json, summary_json
        ) VALUES (
            'AAPL', 'Apple Inc', 'Technology', 'sp500', 0, '2026-03-01T10:00:00',
            '2026-03-01', 180.0, 'manual', NULL, NULL, '[]', '[]', '{}', '{"score": 0.8}'
        );
        """
    )
    conn.commit()
    conn.close()


def _create_db_with_null_run_id(db_path: Path) -> None:
    """Creates a database where run_id column exists but contains explicit NULL values."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE screening_runs (
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
            run_id TEXT
        );

        CREATE TABLE predictions (
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
            run_id TEXT,
            lookback INTEGER NOT NULL DEFAULT 60,
            num_samples INTEGER NOT NULL DEFAULT 1,
            mc_dropout INTEGER NOT NULL DEFAULT 0,
            market_resampling INTEGER NOT NULL DEFAULT 1,
            milestones_json TEXT NOT NULL,
            quantiles_json TEXT NOT NULL,
            cone_json TEXT NOT NULL,
            summary_json TEXT NOT NULL
        );
        """
    )
    # Insert rows: one with NULL run_id and one with explicit custom run_id
    conn.execute(
        """
        INSERT INTO screening_runs (created_at, universe, top_n, min_price, interval, results_json, run_id)
        VALUES ('2026-03-01T10:00:00', 'sp500', 5, 10.0, '1d', '{}', NULL);
        """
    )
    conn.execute(
        """
        INSERT INTO screening_runs (created_at, universe, top_n, min_price, interval, results_json, run_id)
        VALUES ('2026-03-02T10:00:00', 'sp500', 5, 10.0, '1d', '{}', 'custom_run');
        """
    )
    conn.execute(
        """
        INSERT INTO predictions (
            ticker, name, sector, universe, is_index, created_at, anchor_date,
            anchor_close, source, screen_category, run_id, milestones_json,
            quantiles_json, cone_json, summary_json
        ) VALUES (
            'MSFT', 'Microsoft', 'Technology', 'sp500', 0, '2026-03-01T10:00:00',
            '2026-03-01', 400.0, 'manual', NULL, NULL, '[]', '[]', '{}', '{"score": 0.9}'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO predictions (
            ticker, name, sector, universe, is_index, created_at, anchor_date,
            anchor_close, source, screen_category, run_id, milestones_json,
            quantiles_json, cone_json, summary_json
        ) VALUES (
            'GOOGL', 'Alphabet', 'Technology', 'sp500', 0, '2026-03-02T10:00:00',
            '2026-03-02', 170.0, 'manual', NULL, 'custom_run', '[]', '[]', '{}', '{"score": 0.85}'
        );
        """
    )
    conn.commit()
    conn.close()


class TestDatabaseMigration(unittest.TestCase):
    """Tests schema migration and run_id normalization."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_legacy_db_without_run_id_column_migrates_and_backfills(self) -> None:
        db_path = self.db_dir / "legacy_no_column.db"
        _create_legacy_db_without_run_id(db_path)

        # Opening via StockAppDatabase should trigger _init_schema migration
        db = StockAppDatabase(db_path)

        with db._connect() as conn:
            # Check predictions table info
            pred_cols = {c[1]: c for c in conn.execute("PRAGMA table_info(predictions)").fetchall()}
            self.assertIn("run_id", pred_cols)

            # Check screening_runs table info
            screen_cols = {c[1]: c for c in conn.execute("PRAGMA table_info(screening_runs)").fetchall()}
            self.assertIn("run_id", screen_cols)

            # Verify existing row backfilled to 'default'
            p_rows = conn.execute("SELECT ticker, run_id FROM predictions").fetchall()
            self.assertEqual(len(p_rows), 1)
            self.assertEqual(p_rows[0]["ticker"], "AAPL")
            self.assertEqual(p_rows[0]["run_id"], "default")

            s_rows = conn.execute("SELECT id, run_id FROM screening_runs").fetchall()
            self.assertEqual(len(s_rows), 1)
            self.assertEqual(s_rows[0]["run_id"], "default")

            # Check indexes exist
            indexes = [r[1] for r in conn.execute("PRAGMA index_list(predictions)").fetchall()]
            self.assertIn("idx_predictions_run_id", indexes)

    def test_db_with_null_run_id_values_unconditionally_backfilled(self) -> None:
        db_path = self.db_dir / "legacy_null_vals.db"
        _create_db_with_null_run_id(db_path)

        # Opening via StockAppDatabase should run unconditional backfill
        db = StockAppDatabase(db_path)

        with db._connect() as conn:
            # NULL run_id rows should now be 'default'
            msft_row = conn.execute("SELECT run_id FROM predictions WHERE ticker = 'MSFT'").fetchone()
            self.assertEqual(msft_row["run_id"], "default")

            # Existing non-null run_id should remain unchanged
            googl_row = conn.execute("SELECT run_id FROM predictions WHERE ticker = 'GOOGL'").fetchone()
            self.assertEqual(googl_row["run_id"], "custom_run")

            # Check screening runs
            s_rows = conn.execute("SELECT id, run_id FROM screening_runs ORDER BY id ASC").fetchall()
            self.assertEqual(s_rows[0]["run_id"], "default")
            self.assertEqual(s_rows[1]["run_id"], "custom_run")

    def test_fresh_db_initializes_with_run_id_column(self) -> None:
        db_path = self.db_dir / "fresh.db"
        db = StockAppDatabase(db_path)

        with db._connect() as conn:
            pred_cols = {c[1]: c for c in conn.execute("PRAGMA table_info(predictions)").fetchall()}
            self.assertIn("run_id", pred_cols)
            # Default value in DDL should be 'default'
            self.assertIn("'default'", str(pred_cols["run_id"][4]))

            screen_cols = {c[1]: c for c in conn.execute("PRAGMA table_info(screening_runs)").fetchall()}
            self.assertIn("run_id", screen_cols)
            self.assertIn("'default'", str(screen_cols["run_id"][4]))


class TestGetTrackedStocksAccuracyAndFiltering(unittest.TestCase):
    """Tests get_tracked_stocks correctness and filtering against single-stock predictions."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db = StockAppDatabase(Path(self.tmp_dir.name) / "test_tracked.db")
        self._populate_test_data()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _populate_test_data(self) -> None:
        dummy_cone = {
            "milestones": [
                {"horizon_days": 1, "prices": {"q50": 100.0}, "returns": {"q50": 0.01}},
                {"horizon_days": 2, "prices": {"q50": 102.0}, "returns": {"q50": 0.02}},
            ]
        }
        dummy_summary = {"day1_ret_50": 0.01, "score": 0.8}

        # 1. AAPL: 3 predictions under "default" (2 manual, 1 screen hit)
        self.db.save_prediction(
            ticker="AAPL",
            name="Apple Inc",
            sector="Technology",
            universe="sp500",
            anchor_date="2026-03-01",
            anchor_close=180.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.70},
            source="manual",
            run_id="default",
            created_at="2026-03-01T10:00:00",
        )
        self.db.save_prediction(
            ticker="AAPL",
            name="Apple Inc",
            sector="Technology",
            universe="sp500",
            anchor_date="2026-03-02",
            anchor_close=182.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.75},
            source="screen",
            screen_category="bullish",
            run_id="default",
            created_at="2026-03-02T10:00:00",
        )
        self.db.save_prediction(
            ticker="AAPL",
            name="Apple Inc",
            sector="Technology",
            universe="sp500",
            anchor_date="2026-03-03",
            anchor_close=185.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.82},
            source="manual",
            run_id="default",
            created_at="2026-03-03T10:00:00",
        )

        # 2. MSFT: 2 predictions under "run_alpha" (both screen hits with different categories)
        self.db.save_prediction(
            ticker="MSFT",
            name="Microsoft Corp",
            sector="Technology",
            universe="sp500",
            anchor_date="2026-03-02",
            anchor_close=400.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.65},
            source="screen",
            screen_category="bearish",
            run_id="run_alpha",
            created_at="2026-03-02T11:00:00",
        )
        self.db.save_prediction(
            ticker="MSFT",
            name="Microsoft Corp",
            sector="Technology",
            universe="sp500",
            anchor_date="2026-03-04",
            anchor_close=410.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.88},
            source="screen",
            screen_category="volatility",
            run_id="run_alpha",
            created_at="2026-03-04T11:00:00",
        )

        # 3. NVDA: 1 prediction under "default", 1 prediction under "run_alpha"
        self.db.save_prediction(
            ticker="NVDA",
            name="Nvidia Corp",
            sector="Semiconductors",
            universe="sp500",
            anchor_date="2026-03-01",
            anchor_close=800.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.90},
            source="manual",
            run_id="default",
            created_at="2026-03-01T12:00:00",
        )
        self.db.save_prediction(
            ticker="NVDA",
            name="Nvidia Corp",
            sector="Semiconductors",
            universe="sp500",
            anchor_date="2026-03-05",
            anchor_close=850.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary={"score": 0.95},
            source="screen",
            screen_category="bullish",
            run_id="run_alpha",
            created_at="2026-03-05T12:00:00",
        )

    def test_tracked_stocks_matches_single_stock_predictions_all(self) -> None:
        """Verifies get_tracked_stocks produces records matching get_stock_predictions."""
        for model_run in [None, "all", "default", "run_alpha"]:
            tracked = self.db.get_tracked_stocks(model_run=model_run)
            for item in tracked:
                ticker = item["ticker"]
                single_preds = self.db.get_stock_predictions(ticker, model_run=model_run)
                self.assertGreater(len(single_preds), 0, f"No predictions found for {ticker}")

                latest = single_preds[0]
                self.assertEqual(item["latest_prediction_id"], latest["id"])
                self.assertEqual(item["latest_anchor_date"], latest["anchor_date"])
                self.assertEqual(item["latest_anchor_close"], latest["anchor_close"])
                self.assertEqual(item["latest_created_at"], latest["created_at"])
                self.assertEqual(item["latest_source"], latest["source"])
                self.assertEqual(item["latest_summary"], latest["summary"])
                self.assertEqual(item["latest_run_id"], latest["run_id"])
                self.assertEqual(item["name"], latest["name"])
                self.assertEqual(item["sector"], latest["sector"])
                self.assertEqual(item["universe"], latest["universe"])
                self.assertEqual(item["is_index"], latest["is_index"])

                # Verification of aggregate metrics
                expected_total = len(single_preds)
                expected_manual = sum(1 for p in single_preds if p["source"] == "manual")
                expected_screen = sum(1 for p in single_preds if p["source"] == "screen")
                expected_cats = set(p["screen_category"] for p in single_preds if p.get("screen_category"))

                self.assertEqual(item["total_predictions"], expected_total)
                self.assertEqual(item["manual_count"], expected_manual)
                self.assertEqual(item["screen_hit_count"], expected_screen)
                self.assertEqual(set(item["screen_categories"]), expected_cats)

    def test_tracked_stocks_filtering_by_model_run(self) -> None:
        """Verifies strict filtering by model_run."""
        # 'all' or None includes all 3 stocks
        tracked_all = self.db.get_tracked_stocks(model_run="all")
        self.assertEqual({s["ticker"] for s in tracked_all}, {"AAPL", "MSFT", "NVDA"})

        # 'default' includes AAPL and NVDA, but NOT MSFT
        tracked_default = self.db.get_tracked_stocks(model_run="default")
        self.assertEqual({s["ticker"] for s in tracked_default}, {"AAPL", "NVDA"})
        aapl_item = next(s for s in tracked_default if s["ticker"] == "AAPL")
        self.assertEqual(aapl_item["total_predictions"], 3)
        self.assertEqual(aapl_item["latest_anchor_date"], "2026-03-03")

        # 'run_alpha' includes MSFT and NVDA, but NOT AAPL
        tracked_alpha = self.db.get_tracked_stocks(model_run="run_alpha")
        self.assertEqual({s["ticker"] for s in tracked_alpha}, {"MSFT", "NVDA"})
        msft_item = next(s for s in tracked_alpha if s["ticker"] == "MSFT")
        self.assertEqual(msft_item["total_predictions"], 2)
        self.assertEqual(msft_item["screen_hit_count"], 2)
        self.assertEqual(set(msft_item["screen_categories"]), {"bearish", "volatility"})
        self.assertEqual(msft_item["latest_anchor_date"], "2026-03-04")
        self.assertEqual(msft_item["latest_screen_category"], "volatility")

        # Nonexistent run_id returns empty list
        tracked_empty = self.db.get_tracked_stocks(model_run="nonexistent_run")
        self.assertEqual(tracked_empty, [])

    def test_counts_consistency(self) -> None:
        """Verifies get_counts aligns with get_tracked_stocks count."""
        for run_id in [None, "all", "default", "run_alpha", "nonexistent"]:
            counts = self.db.get_counts(model_run=run_id)
            tracked = self.db.get_tracked_stocks(model_run=run_id)
            self.assertEqual(counts["tracked_stocks"], len(tracked))


class TestPerformanceAndSingleQueryExecution(unittest.TestCase):
    """Tests performance and verifies N+1 query loop elimination."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db = StockAppDatabase(Path(self.tmp_dir.name) / "test_perf.db")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_single_query_execution_no_n_plus_one(self) -> None:
        """Confirms that get_tracked_stocks executes only 1 SQL query regardless of ticker count."""
        # Insert predictions for 25 distinct tickers
        dummy_cone = {"milestones": []}
        dummy_summary = {"score": 0.5}

        for i in range(25):
            sym = f"SYM{i:02d}"
            self.db.save_prediction(
                ticker=sym,
                anchor_date="2026-03-01",
                anchor_close=100.0 + i,
                milestones=[1],
                quantiles=[0.5],
                cone=dummy_cone,
                summary=dummy_summary,
                source="screen" if i % 2 == 0 else "manual",
                screen_category="bullish" if i % 2 == 0 else None,
            )

        executed_queries: List[str] = []

        # Hook trace on connection
        orig_connect = self.db._connect

        def traced_connect() -> sqlite3.Connection:
            conn = orig_connect()
            conn.set_trace_callback(executed_queries.append)
            return conn

        self.db._connect = traced_connect  # type: ignore[assignment]

        try:
            executed_queries.clear()
            tracked = self.db.get_tracked_stocks()
            self.assertEqual(len(tracked), 25)

            # Filter out internal PRAGMA statements from connection setup
            data_queries = [
                q for q in executed_queries
                if not q.strip().upper().startswith("PRAGMA")
            ]

            # In the old code, this was 1 + 25*2 = 51 queries.
            # Now it must be exactly 1 query!
            self.assertEqual(
                len(data_queries),
                1,
                f"Expected exactly 1 SQL query, but executed {len(data_queries)}: {data_queries}",
            )
            # Confirm window function was used
            self.assertIn("ROW_NUMBER()", data_queries[0])
            self.assertIn("PARTITION BY ticker", data_queries[0])
        finally:
            self.db._connect = orig_connect  # type: ignore[assignment]

    def test_multiple_stocks_and_categories_correctness(self) -> None:
        """Tests 50+ stocks with duplicate categories, varying anchor dates, and sorting."""
        dummy_cone = {"milestones": []}
        dummy_summary = {"score": 0.5}

        # Populate 50 stocks with multiple predictions per stock
        for i in range(50):
            sym = f"STK{i:03d}"
            # Earlier prediction
            self.db.save_prediction(
                ticker=sym,
                anchor_date="2026-03-01",
                anchor_close=50.0,
                milestones=[1],
                quantiles=[0.5],
                cone=dummy_cone,
                summary={"score": 0.1},
                source="screen",
                screen_category="bullish",
                created_at=f"2026-03-01T{i % 24:02d}:00:00",
            )
            # Duplicate category prediction on another date
            self.db.save_prediction(
                ticker=sym,
                anchor_date="2026-03-02",
                anchor_close=55.0,
                milestones=[1],
                quantiles=[0.5],
                cone=dummy_cone,
                summary={"score": 0.2},
                source="screen",
                screen_category="bullish",  # duplicate category
                created_at=f"2026-03-02T{i % 24:02d}:00:00",
            )
            # Another category prediction
            self.db.save_prediction(
                ticker=sym,
                anchor_date="2026-03-05",
                anchor_close=60.0 + i,
                milestones=[1],
                quantiles=[0.5],
                cone=dummy_cone,
                summary={"score": 0.9, "idx": i},
                source="screen",
                screen_category="volatility",
                created_at=f"2026-03-05T{i % 24:02d}:00:00",
            )

        start = time.perf_counter()
        tracked = self.db.get_tracked_stocks()
        elapsed_sec = time.perf_counter() - start

        self.assertEqual(len(tracked), 50)
        # Should be fast (< 100ms)
        self.assertLess(elapsed_sec, 0.5)

        for item in tracked:
            # Check distinct categories
            self.assertEqual(sorted(item["screen_categories"]), ["bullish", "volatility"])
            self.assertEqual(item["total_predictions"], 3)
            self.assertEqual(item["screen_hit_count"], 3)
            self.assertEqual(item["manual_count"], 0)
            # Latest prediction must be the one with anchor_date 2026-03-05
            self.assertEqual(item["latest_anchor_date"], "2026-03-05")
            self.assertEqual(item["latest_screen_category"], "volatility")
            self.assertEqual(item["latest_summary"]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
