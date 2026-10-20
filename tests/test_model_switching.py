"""Test suite for Stock_app Model Switching, Lineage Tracking, and Leaderboard API.

Tests:
1. StockAppDatabase run_id schema migration, app_settings persistence, and queries with model_run filters.
2. MLStockBridge model discovery (list_available_models), detail inspection (get_model_detail), and active model switching (set_active_model).
3. HTTP Endpoints for GET /api/models, GET /api/models/<run_id>, POST /api/models/select, and GET /api/overview?model_run=...
4. Model lineage preservation in predictions and screening runs.
"""

from __future__ import annotations

from http import HTTPStatus
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Dict
import unittest
import urllib.request
from unittest.mock import patch, MagicMock

# Ensure Stock_app root is in sys.path
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app import StockAppRequestHandler, bridge, db
from database import StockAppDatabase
from ml_bridge import MLStockBridge


class TestDatabaseModelLineage(unittest.TestCase):
    """Verifies SQLite database methods for model lineage tracking and filtering."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_lineage.db"
        self.test_db = StockAppDatabase(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_app_settings_get_and_set(self) -> None:
        # Initial default value
        val = self.test_db.get_setting("active_run_id", "default")
        self.assertEqual(val, "default")

        # Set new setting
        self.test_db.set_setting("active_run_id", "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1")
        val2 = self.test_db.get_setting("active_run_id")
        self.assertEqual(val2, "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1")

        # Overwrite setting
        self.test_db.set_setting("active_run_id", "default")
        val3 = self.test_db.get_setting("active_run_id")
        self.assertEqual(val3, "default")

    def test_prediction_filtering_by_run_id(self) -> None:
        dummy_cone = {
            "anchor_date": "2026-03-01",
            "anchor_close": 100.0,
            "milestones": [
                {
                    "horizon_days": 1,
                    "returns": {"q50": 0.01},
                    "prices": {"q50": 101.0},
                }
            ],
        }
        dummy_summary = {"day1_ret_50": 0.01, "score": 0.75}

        # Prediction with default run
        p1 = self.test_db.save_prediction(
            ticker="AAPL",
            anchor_date="2026-03-01",
            anchor_close=100.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary=dummy_summary,
            run_id="default",
        )
        # Prediction with Run A
        p2 = self.test_db.save_prediction(
            ticker="AAPL",
            anchor_date="2026-03-02",
            anchor_close=102.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary=dummy_summary,
            run_id="run_alpha",
        )
        # Prediction with Run B for MSFT
        p3 = self.test_db.save_prediction(
            ticker="MSFT",
            anchor_date="2026-03-02",
            anchor_close=300.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=dummy_cone,
            summary=dummy_summary,
            run_id="run_beta",
        )

        # Unfiltered history should contain all 3
        all_preds = self.test_db.get_prediction_history(model_run="all")
        self.assertEqual(len(all_preds), 3)

        # Filtered by run_alpha
        alpha_preds = self.test_db.get_prediction_history(model_run="run_alpha")
        self.assertEqual(len(alpha_preds), 1)
        self.assertEqual(alpha_preds[0]["id"], p2["id"])
        self.assertEqual(alpha_preds[0]["run_id"], "run_alpha")

        # Filtered by run_beta
        beta_preds = self.test_db.get_prediction_history(model_run="run_beta")
        self.assertEqual(len(beta_preds), 1)
        self.assertEqual(beta_preds[0]["ticker"], "MSFT")

        # Tracked stocks: "all" should return AAPL and MSFT
        tracked_all = self.test_db.get_tracked_stocks(model_run="all")
        self.assertEqual(len(tracked_all), 2)

        # Tracked stocks: run_beta should return only MSFT
        tracked_beta = self.test_db.get_tracked_stocks(model_run="run_beta")
        self.assertEqual(len(tracked_beta), 1)
        self.assertEqual(tracked_beta[0]["ticker"], "MSFT")

        # Counts by model_run
        counts_all = self.test_db.get_counts(model_run="all")
        self.assertEqual(counts_all["total_predictions"], 3)
        self.assertEqual(counts_all["tracked_stocks"], 2)

        counts_alpha = self.test_db.get_counts(model_run="run_alpha")
        self.assertEqual(counts_alpha["total_predictions"], 1)
        self.assertEqual(counts_alpha["tracked_stocks"], 1)

    def test_screening_run_filtering_by_run_id(self) -> None:
        payload = {"top_bullish": [], "top_bearish": []}
        self.test_db.save_screening_run(
            universe="sp500",
            top_n=5,
            min_price=10.0,
            interval="1d",
            total_candidates=100,
            total_screened=100,
            checkpoint_path="/path/a",
            duration_sec=2.5,
            results_payload=payload,
            hit_predictions=[],
            run_id="run_1",
        )
        self.test_db.save_screening_run(
            universe="sp500",
            top_n=5,
            min_price=10.0,
            interval="1d",
            total_candidates=100,
            total_screened=100,
            checkpoint_path="/path/b",
            duration_sec=3.0,
            results_payload=payload,
            hit_predictions=[],
            run_id="run_2",
        )

        all_screens = self.test_db.get_screening_history(model_run="all")
        self.assertEqual(len(all_screens), 2)

        run1_screens = self.test_db.get_screening_history(model_run="run_1")
        self.assertEqual(len(run1_screens), 1)
        self.assertEqual(run1_screens[0]["run_id"], "run_1")


class TestBridgeModelDiscoveryAndSwitching(unittest.TestCase):
    """Verifies MLStockBridge discovers archived models and supports active model switching."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge = bridge

    def test_list_available_models(self) -> None:
        models = self.bridge.list_available_models()
        self.assertTrue(len(models) >= 1)

        # First model should be default root
        default_model = models[0]
        self.assertEqual(default_model["run_id"], "default")
        self.assertEqual(default_model["badge_label"], "Active Root")

        # Check for archived runs from runs_index.csv
        run_ids = [m["run_id"] for m in models]
        self.assertIn("2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1", run_ids)

        # Inspect the 3-head model metadata
        m_3head = next(m for m in models if m["run_id"] == "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1")
        self.assertEqual(m_3head["badge_label"], "3-Head 4L")
        self.assertEqual(m_3head["num_layers"], 4)
        self.assertTrue(m_3head["exists"])
        self.assertIsNotNone(m_3head.get("test_score"))

    def test_get_model_detail(self) -> None:
        run_id = "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        detail = self.bridge.get_model_detail(run_id)

        self.assertTrue(detail["ok"])
        self.assertEqual(detail["run_id"], run_id)
        self.assertTrue(detail["exists"])
        self.assertIn("summary", detail)
        self.assertIn("report_txt", detail)
        self.assertIn("history_rows", detail)

        # Check that report_txt or summary has valid data
        self.assertTrue(len(detail["report_txt"]) > 0 or len(detail["summary"]) > 0)

    def test_set_active_model_and_revert(self) -> None:
        original_active = self.bridge.active_run_id

        target_run = "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        res = self.bridge.set_active_model(target_run, promote_to_root=False)
        self.assertTrue(res["ok"])
        self.assertEqual(self.bridge.active_run_id, target_run)
        self.assertIn(target_run, str(self.bridge.checkpoint_path))

        # Check persistence in database
        saved_db_setting = self.bridge.db.get_setting("active_run_id")
        self.assertEqual(saved_db_setting, target_run)

        # Revert back to original (or "default")
        revert_id = original_active if original_active else "default"
        res_revert = self.bridge.set_active_model(revert_id, promote_to_root=False)
        self.assertTrue(res_revert["ok"])
        self.assertEqual(self.bridge.active_run_id, revert_id)

    def test_set_invalid_model_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.bridge.set_active_model("non_existent_fake_run_12345")

    def test_path_traversal_rejection(self) -> None:
        traversal_attempts = [
            "../../etc/passwd",
            "/tmp/malicious.pt",
            "../checkpoints/patchtst_best.pt",
            "/etc/shadow",
        ]
        for attempt in traversal_attempts:
            with self.assertRaises(ValueError):
                self.bridge._resolve_checkpoint_path(attempt)
            with self.assertRaises(ValueError):
                self.bridge.set_active_model(attempt)

    def test_unknown_run_id_handling(self) -> None:
        unknown_id = "non_existent_run_99999"
        with self.assertRaises(FileNotFoundError):
            self.bridge._resolve_checkpoint_path(unknown_id)
        with self.assertRaises(FileNotFoundError):
            self.bridge._resolve_screener_and_run_id(unknown_id)

        detail = self.bridge.get_model_detail(unknown_id)
        self.assertFalse(detail["ok"])
        self.assertFalse(detail["exists"])
        self.assertIn("error", detail)

    def test_market_pool_cache_cleared_on_switch(self) -> None:
        import numpy as np
        # Populate market pool cache with mock data
        self.bridge._market_pool_cache = [("MOCK", np.zeros((60, 16), dtype=np.float32))]
        self.assertEqual(len(self.bridge._market_pool_cache), 1)

        # Switch model
        target_run = "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        self.bridge.set_active_model(target_run, promote_to_root=False)
        self.assertTrue(
            self.bridge._market_pool_cache is None or len(self.bridge._market_pool_cache) == 0
        )

        # Switch back to default
        self.bridge._market_pool_cache = [("MOCK", np.zeros((60, 16), dtype=np.float32))]
        self.bridge.set_active_model("default", promote_to_root=False)
        self.assertTrue(
            self.bridge._market_pool_cache is None or len(self.bridge._market_pool_cache) == 0
        )

    def test_normalized_model_detail_metrics(self) -> None:
        run_id = "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        detail = self.bridge.get_model_detail(run_id)
        self.assertTrue(detail["ok"])
        summary = detail["summary"]
        self.assertIn("test_evaluation", summary)
        self.assertIn("test_metrics", summary)
        te = summary["test_evaluation"]
        tm = summary["test_metrics"]
        # Both should have aliases populated
        if "test_cov50" in tm:
            self.assertIn("coverage_50pct_band", te)
            self.assertEqual(tm["test_cov50"], te["coverage_50pct_band"])
        if "test_score" in tm:
            self.assertIn("score", te)
            self.assertEqual(tm["test_score"], te["score"])


class TestHttpModelSwitchingEndpoints(unittest.TestCase):
    """End-to-end integration tests for HTTP API endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        from http.server import ThreadingHTTPServer

        # Bind to port 0 for an OS-assigned ephemeral port to prevent collisions
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StockAppRequestHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_api_models(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertTrue(data["ok"])
        self.assertIn("models", data)
        self.assertIn("active_run_id", data)
        self.assertTrue(len(data["models"]) >= 1)

    def test_get_api_model_detail(self) -> None:
        run_id = "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        url = f"http://127.0.0.1:{self.port}/api/models/{run_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertTrue(data["ok"])
        self.assertEqual(data["run_id"], run_id)
        self.assertIn("summary", data)

    def test_post_api_models_select_and_revert(self) -> None:
        target_run = "2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        url = f"http://127.0.0.1:{self.port}/api/models/select"

        body = json.dumps({"run_id": target_run, "promote_to_root": False}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertTrue(data["ok"])
        self.assertEqual(data["active_run_id"], target_run)

        # Revert back to default
        body_revert = json.dumps({"run_id": "default", "promote_to_root": False}).encode("utf-8")
        req_revert = urllib.request.Request(url, data=body_revert, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_revert, timeout=10) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            data_revert = json.loads(resp.read().decode("utf-8"))

        self.assertTrue(data_revert["ok"])
        self.assertEqual(data_revert["active_run_id"], "default")

    def test_post_api_models_select_unknown_returns_404(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/models/select"
        body = json.dumps({"run_id": "non_existent_fake_run_9999", "promote_to_root": False}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, HTTPStatus.NOT_FOUND)
        except urllib.error.HTTPError as err:
            self.assertEqual(err.code, HTTPStatus.NOT_FOUND)
            err_data = json.loads(err.read().decode("utf-8"))
            self.assertFalse(err_data["ok"])
            self.assertIn("error", err_data)

    def test_post_api_models_select_traversal_returns_400(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/models/select"
        body = json.dumps({"run_id": "../../etc/passwd", "promote_to_root": False}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, HTTPStatus.BAD_REQUEST)
        except urllib.error.HTTPError as err:
            self.assertEqual(err.code, HTTPStatus.BAD_REQUEST)
            err_data = json.loads(err.read().decode("utf-8"))
            self.assertFalse(err_data["ok"])
            self.assertIn("error", err_data)

    def test_get_api_overview_with_model_filter(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/overview?model_run=2026-10-19_16-37-33_mq_4L_m4_3head_1-1-1"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            data = json.loads(resp.read().decode("utf-8"))

        self.assertTrue(data["ok"])
        self.assertIn("tracked_stocks", data)
        self.assertIn("prediction_history", data)
        self.assertIn("counts", data)
        self.assertIn("active_run_id", data)


if __name__ == "__main__":
    unittest.main()
