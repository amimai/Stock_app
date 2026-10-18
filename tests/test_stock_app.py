"""Comprehensive automated test suite for Stock_app.

Tests:
1. get_stock_detail with all-NaN rows, partial-NaN rows, and JSON serialization.
2. Real data test for 000660.KS (get_stock_detail & GET /api/stock/000660.KS).
3. enrich_prediction_with_realized and build_cone_and_summary with NaN edge cases.
4. _sanitize_json_value and StockAppRequestHandler._send_json strict RFC compliance.
5. StockAppDatabase CRUD operations for predictions and screening runs.
"""

from __future__ import annotations

from http import HTTPStatus
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List
import unittest
import urllib.request

# Ensure Stock_app root is in sys.path
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd

from app import StockAppRequestHandler, _sanitize_json_value
from cone_utils import build_cone_and_summary, enrich_prediction_with_realized
from database import StockAppDatabase, _safe_json_dumps, _sanitize_for_json
from ml_bridge import MLStockBridge


class TestSanitizeJsonValue(unittest.TestCase):
    """Test suite for _sanitize_json_value and JSON serialization helpers."""

    def test_primitives_preservation(self) -> None:
        self.assertIsNone(_sanitize_json_value(None))
        self.assertEqual(_sanitize_json_value(True), True)
        self.assertEqual(_sanitize_json_value(False), False)
        self.assertEqual(_sanitize_json_value(0), 0)
        self.assertEqual(_sanitize_json_value(42), 42)
        self.assertEqual(_sanitize_json_value(-100), -100)
        self.assertEqual(_sanitize_json_value(3.1415), 3.1415)
        self.assertEqual(_sanitize_json_value("hello"), "hello")

    def test_non_finite_floats_converted_to_none(self) -> None:
        self.assertIsNone(_sanitize_json_value(float("nan")))
        self.assertIsNone(_sanitize_json_value(float("inf")))
        self.assertIsNone(_sanitize_json_value(float("-inf")))
        self.assertIsNone(_sanitize_json_value(np.nan))
        self.assertIsNone(_sanitize_json_value(np.float32("nan")))
        self.assertIsNone(_sanitize_json_value(np.float64("nan")))
        self.assertIsNone(_sanitize_json_value(np.inf))
        self.assertIsNone(_sanitize_json_value(-np.inf))

    def test_numpy_and_pandas_types(self) -> None:
        self.assertEqual(_sanitize_json_value(np.int64(123)), 123)
        self.assertEqual(_sanitize_json_value(np.bool_(True)), True)
        self.assertEqual(_sanitize_json_value(np.float64(45.67)), 45.67)
        self.assertIsNone(_sanitize_json_value(pd.NA))
        self.assertIsNone(_sanitize_json_value(pd.NaT))

        arr = np.array([1.5, np.nan, 3.0])
        self.assertEqual(_sanitize_json_value(arr), [1.5, None, 3.0])

    def test_nested_structures(self) -> None:
        raw = {
            "name": "Test",
            "active": True,
            "count": 5,
            "prices": [10.0, float("nan"), 12.5, float("inf")],
            "cone": {
                "upper": float("nan"),
                "median": 10.5,
                "stds": (0.1, np.nan, 0.3),
            },
        }
        sanitized = _sanitize_json_value(raw)
        self.assertEqual(sanitized["name"], "Test")
        self.assertEqual(sanitized["active"], True)
        self.assertEqual(sanitized["count"], 5)
        self.assertEqual(sanitized["prices"], [10.0, None, 12.5, None])
        self.assertIsNone(sanitized["cone"]["upper"])
        self.assertEqual(sanitized["cone"]["median"], 10.5)
        self.assertEqual(sanitized["cone"]["stds"], [0.1, None, 0.3])

        # Strictly valid JSON without NaN
        dumped = json.dumps(sanitized, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)
        loaded = json.loads(dumped)
        self.assertIsNone(loaded["cone"]["upper"])
        self.assertIsNone(loaded["prices"][1])


class DummySocket:
    """Mock socket for BaseHTTPRequestHandler testing."""

    def __init__(self) -> None:
        self.data = bytearray()

    def makefile(self, *args: Any, **kwargs: Any) -> io.BytesIO:
        return io.BytesIO(self.data)

    def sendall(self, data: bytes) -> None:
        self.data.extend(data)


class DummyHandler(StockAppRequestHandler):
    """Subclass of StockAppRequestHandler with mocked network I/O."""

    def __init__(self, response_stream: io.BytesIO) -> None:
        self.rfile = io.BytesIO()
        self.wfile = response_stream
        self._headers_buffer: List[bytes] = []

    def send_response(self, code: int, message: Optional[str] = None) -> None:
        self.response_code = code

    def send_header(self, keyword: str, value: str) -> None:
        self._headers_buffer.append(f"{keyword}: {value}\r\n".encode("latin-1"))

    def end_headers(self) -> None:
        pass


class TestRequestHandlerSendJson(unittest.TestCase):
    """Test suite for StockAppRequestHandler._send_json strict JSON compliance."""

    def test_send_json_with_nan_and_inf(self) -> None:
        out_stream = io.BytesIO()
        handler = DummyHandler(out_stream)

        payload = {
            "ok": True,
            "metric": float("nan"),
            "variance": np.nan,
            "infinity": float("inf"),
            "series": [1.0, float("nan"), 3.0],
        }

        # Must not raise ValueError: Out of range float values are not JSON compliant
        handler._send_json(payload, status=HTTPStatus.OK)

        raw_bytes = out_stream.getvalue()
        self.assertTrue(len(raw_bytes) > 0)

        raw_str = raw_bytes.decode("utf-8")
        self.assertNotIn("NaN", raw_str)
        self.assertNotIn("Infinity", raw_str)

        parsed = json.loads(raw_str)
        self.assertTrue(parsed["ok"])
        self.assertIsNone(parsed["metric"])
        self.assertIsNone(parsed["variance"])
        self.assertIsNone(parsed["infinity"])
        self.assertEqual(parsed["series"], [1.0, None, 3.0])


class TestConeUtilsNaN(unittest.TestCase):
    """Test suite for cone_utils NaN handling and edge cases."""

    def setUp(self) -> None:
        self.anchor_date = "2026-01-05"
        self.anchor_close = 100.0
        self.milestones = [1, 2, 5, 20]
        self.quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    def test_build_cone_and_summary_clean(self) -> None:
        pred_rets = np.zeros((4, 7), dtype=np.float64)
        pred_rets[0] = [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03]
        pred_rets[1] = [-0.04, -0.03, -0.015, 0.005, 0.02, 0.035, 0.05]
        pred_rets[2] = [-0.06, -0.04, -0.02, 0.01, 0.03, 0.05, 0.07]
        pred_rets[3] = [-0.10, -0.07, -0.03, 0.02, 0.05, 0.09, 0.12]

        cone, summary = build_cone_and_summary(
            anchor_date=self.anchor_date,
            anchor_close=self.anchor_close,
            milestones=self.milestones,
            quantiles=self.quantiles,
            pred_cum_returns=pred_rets,
        )

        self.assertEqual(cone["anchor_date"], self.anchor_date)
        self.assertEqual(cone["anchor_close"], 100.0)
        self.assertEqual(len(cone["milestones"]), 4)
        self.assertIn("1d", summary["horizons"])

        # Strict JSON dump check
        json.dumps(cone, allow_nan=False)
        json.dumps(summary, allow_nan=False)

    def test_build_cone_and_summary_with_nan_inputs(self) -> None:
        pred_rets = np.zeros((4, 7), dtype=np.float64)
        pred_rets[0, 0] = np.nan
        pred_rets[1, 3] = np.nan
        stds = np.full((4, 7), np.nan)

        cone, summary = build_cone_and_summary(
            anchor_date=self.anchor_date,
            anchor_close=self.anchor_close,
            milestones=self.milestones,
            quantiles=self.quantiles,
            pred_cum_returns=pred_rets,
            sample_std=stds,
        )

        dumped_cone = json.dumps(cone, allow_nan=False)
        dumped_sum = json.dumps(summary, allow_nan=False)
        self.assertNotIn("NaN", dumped_cone)
        self.assertNotIn("NaN", dumped_sum)

    def test_enrich_prediction_with_realized_empty_or_none(self) -> None:
        pred = {
            "anchor_date": "2026-01-05",
            "anchor_close": 100.0,
            "cone": {"milestones": [{"horizon_days": 1, "returns": {}, "prices": {}}]},
        }
        res_none = enrich_prediction_with_realized(pred, None)  # type: ignore
        self.assertFalse(res_none["realized"]["has_data"])
        self.assertEqual(res_none["realized"]["elapsed_trading_days"], 0)

        res_empty = enrich_prediction_with_realized(pred, pd.DataFrame())
        self.assertFalse(res_empty["realized"]["has_data"])

    def test_enrich_prediction_with_realized_all_nan_close(self) -> None:
        pred = {
            "anchor_date": "2026-01-05",
            "anchor_close": 100.0,
            "cone": {"milestones": [{"horizon_days": 1, "returns": {}, "prices": {}}]},
        }
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "Open": [np.nan] * 5,
                "High": [np.nan] * 5,
                "Low": [np.nan] * 5,
                "Close": [np.nan] * 5,
                "Volume": [0] * 5,
            },
            index=dates,
        )

        res = enrich_prediction_with_realized(pred, df)
        self.assertFalse(res["realized"]["has_data"])
        self.assertEqual(res["realized"]["elapsed_trading_days"], 0)

    def test_enrich_prediction_with_realized_partial_nan(self) -> None:
        pred = {
            "anchor_date": "2026-01-05",
            "anchor_close": 100.0,
            "cone": {
                "milestones": [
                    {
                        "horizon_days": 2,
                        "returns": {"q50": 0.02},
                        "prices": {"q10": 98.0, "q25": 99.0, "q50": 102.0, "q75": 104.0, "q90": 106.0, "q05": 97.0, "q95": 108.0},
                    }
                ]
            },
        }

        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
        df = pd.DataFrame(
            {
                # 2026-01-06 has NaN open/high/low but valid close
                "Open": [100.0, np.nan, 102.0, 103.0],
                "High": [101.0, np.nan, 103.0, 104.0],
                "Low": [99.0, np.nan, 101.0, 102.0],
                "Close": [100.0, 101.5, 102.5, 103.5],
            },
            index=dates,
        )

        enriched = enrich_prediction_with_realized(pred, df)
        self.assertTrue(enriched["realized"]["has_data"])
        self.assertEqual(enriched["realized"]["elapsed_trading_days"], 3)

        # Check daily cone
        daily_cone = enriched["cone"]["daily_cone"]
        for pt in daily_cone:
            if pt["is_realized"]:
                self.assertIsNotNone(pt["actual_close"])
                self.assertFalse(math.isnan(pt["actual_open"]))
                self.assertFalse(math.isnan(pt["actual_high"]))
                self.assertFalse(math.isnan(pt["actual_low"]))
                self.assertFalse(math.isnan(pt["actual_close"]))

        # Check context bars
        for bar in enriched["context_bars"]:
            self.assertFalse(math.isnan(bar["open"]))
            self.assertFalse(math.isnan(bar["high"]))
            self.assertFalse(math.isnan(bar["low"]))
            self.assertFalse(math.isnan(bar["close"]))

        # Check serialization
        dumped = json.dumps(enriched, allow_nan=False)
        self.assertNotIn("NaN", dumped)


class MockDataLoader:
    """Mock YahooParquetDataLoader for testing MLStockBridge."""

    def __init__(self, df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> None:
        self.df = df
        self.meta = meta or {
            "ticker": "MOCK",
            "bars": len(df),
            "start": str(df.index[0])[:10] if not df.empty else "",
            "end": str(df.index[-1])[:10] if not df.empty else "",
        }

    def load(self, ticker: str, interval: str = "1d", auto_fetch: bool = False, end: Optional[str] = None) -> pd.DataFrame:
        return self.df.copy()

    def get_metadata(self, ticker: str, interval: str = "1d") -> Dict[str, Any]:
        return dict(self.meta)

    def fetch_and_save(self, ticker: str, interval: str = "1d") -> pd.DataFrame:
        return self.df.copy()


class TestStockBridgeDetail(unittest.TestCase):
    """Test suite for MLStockBridge.get_stock_detail with NaN data."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = StockAppDatabase(self.db_path)
        self.bridge = MLStockBridge(self.db)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_stock_detail_with_all_nan_holiday_rows(self) -> None:
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Open": [10.0, 11.0, np.nan, 12.0, 13.0, np.nan, 14.0, 15.0, 16.0, 17.0],
                "High": [10.5, 11.5, np.nan, 12.5, 13.5, np.nan, 14.5, 15.5, 16.5, 17.5],
                "Low": [9.5, 10.5, np.nan, 11.5, 12.5, np.nan, 13.5, 14.5, 15.5, 16.5],
                "Close": [10.0, 11.0, np.nan, 12.0, 13.0, np.nan, 14.0, 15.0, 16.0, 17.0],
                "Volume": [1000, 1100, 0, 1200, 1300, 0, 1400, 1500, 1600, 1700],
            },
            index=dates,
        )

        self.bridge.loader = MockDataLoader(df)  # type: ignore

        detail = self.bridge.get_stock_detail("TEST_NAN", bars_limit=50)

        # 2 rows with NaN close should have been dropped, leaving 8 bars
        self.assertEqual(len(detail["bars"]), 8)
        for b in detail["bars"]:
            self.assertTrue(math.isfinite(b["open"]))
            self.assertTrue(math.isfinite(b["high"]))
            self.assertTrue(math.isfinite(b["low"]))
            self.assertTrue(math.isfinite(b["close"]))
            self.assertIsInstance(b["volume"], int)

        # Check serialization with allow_nan=False
        dumped = json.dumps(detail, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("null", [b["close"] for b in detail["bars"]])

    def test_get_stock_detail_with_partial_nan_rows(self) -> None:
        dates = pd.date_range("2026-02-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                # Open, High, Low, Volume are NaN on row 2, but Close is valid
                "Open": [100.0, np.nan, 102.0, 103.0, 104.0],
                "High": [101.0, np.nan, 103.0, 104.0, 105.0],
                "Low": [99.0, np.nan, 101.0, 102.0, 103.0],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000, np.nan, 1200, 1300, 1400],
            },
            index=dates,
        )

        self.bridge.loader = MockDataLoader(df)  # type: ignore

        detail = self.bridge.get_stock_detail("TEST_PARTIAL", bars_limit=50)
        self.assertEqual(len(detail["bars"]), 5)

        # Row 1 (index 1) should fall back to close for open, high, low, and 0 for volume
        bar1 = detail["bars"][1]
        self.assertEqual(bar1["close"], 101.5)
        self.assertEqual(bar1["open"], 101.5)
        self.assertEqual(bar1["high"], 101.5)
        self.assertEqual(bar1["low"], 101.5)
        self.assertEqual(bar1["volume"], 0)

        dumped = json.dumps(detail, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("NaN", dumped)

    def test_real_data_000660_ks(self) -> None:
        """Test on actual cached 000660.KS Parquet file which contains the 2025-09-19 holiday NaN row."""
        try:
            detail = self.bridge.get_stock_detail("000660.KS", bars_limit=320)
        except Exception as e:
            self.skipTest(f"000660.KS Parquet file unavailable: {e}")

        self.assertTrue(detail["in_parquet"])
        self.assertTrue(len(detail["bars"]) > 0)
        self.assertIsNotNone(detail["latest_bar"])

        # Check every bar is strictly finite
        for b in detail["bars"]:
            self.assertTrue(math.isfinite(b["open"]))
            self.assertTrue(math.isfinite(b["high"]))
            self.assertTrue(math.isfinite(b["low"]))
            self.assertTrue(math.isfinite(b["close"]))
            self.assertIsInstance(b["volume"], int)

        # Must serialize cleanly with allow_nan=False
        dumped = json.dumps(detail, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)


class TestRealDataHttpEndpoint(unittest.TestCase):
    """End-to-end integration test running HTTP server and fetching /api/stock/000660.KS."""

    @classmethod
    def setUpClass(cls) -> None:
        from http.server import ThreadingHTTPServer
        import app

        cls.port = 18099
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), StockAppRequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_stock_000660_ks_http(self) -> None:
        url = f"http://127.0.0.1:{self.port}/api/stock/000660.KS"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            raw_bytes = resp.read()

        raw_text = raw_bytes.decode("utf-8")
        self.assertNotIn("NaN", raw_text)
        self.assertNotIn("Infinity", raw_text)

        parsed = json.loads(raw_text)
        self.assertTrue(parsed["ok"])
        self.assertIn("stock", parsed)
        stock = parsed["stock"]
        self.assertEqual(stock["ticker"], "000660.KS")
        self.assertTrue(len(stock["bars"]) > 0)


class TestDatabaseCRUD(unittest.TestCase):
    """Test suite for StockAppDatabase CRUD operations and NaN sanitization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_app.db"
        self.db = StockAppDatabase(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_safe_json_dumps_and_sanitize(self) -> None:
        data = {
            "a": float("nan"),
            "b": [1.0, float("inf"), np.nan],
            "c": {"sub": np.float64("nan"), "valid": 42},
        }
        res_str = _safe_json_dumps(data)
        self.assertNotIn("NaN", res_str)
        self.assertNotIn("Infinity", res_str)
        loaded = json.loads(res_str)
        self.assertIsNone(loaded["a"])
        self.assertEqual(loaded["b"], [1.0, None, None])
        self.assertIsNone(loaded["c"]["sub"])
        self.assertEqual(loaded["c"]["valid"], 42)

    def test_save_and_retrieve_prediction_with_nan(self) -> None:
        # Save a prediction where cone and summary contain NaN values
        cone_with_nan = {
            "anchor_date": "2026-03-01",
            "anchor_close": 150.0,
            "milestones": [
                {
                    "horizon_days": 1,
                    "returns": {"q50": float("nan"), "q10": -0.02},
                    "prices": {"q50": 150.0, "q10": float("nan")},
                }
            ],
        }
        summary_with_nan = {
            "day1_ret_50": float("nan"),
            "skew_2d": np.nan,
            "score": 0.85,
        }

        saved = self.db.save_prediction(
            ticker="AAPL",
            anchor_date="2026-03-01",
            anchor_close=150.0,
            milestones=[1, 2],
            quantiles=[0.1, 0.5, 0.9],
            cone=cone_with_nan,
            summary=summary_with_nan,
            source="manual",
        )

        self.assertIsNotNone(saved["id"])
        pred_id = saved["id"]

        # Retrieve by id
        fetched = self.db.get_prediction_by_id(pred_id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched["ticker"], "AAPL")
        self.assertIsNone(fetched["cone"]["milestones"][0]["returns"]["q50"])
        self.assertIsNone(fetched["cone"]["milestones"][0]["prices"]["q10"])
        self.assertIsNone(fetched["summary"]["day1_ret_50"])
        self.assertIsNone(fetched["summary"]["skew_2d"])
        self.assertEqual(fetched["summary"]["score"], 0.85)

        # Ensure retrieved record serializes cleanly with allow_nan=False
        dumped = json.dumps(fetched, allow_nan=False)
        self.assertNotIn("NaN", dumped)

        # Test get_stock_predictions
        preds = self.db.get_stock_predictions("AAPL")
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["id"], pred_id)

        # Test tracked stocks
        tracked = self.db.get_tracked_stocks()
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["ticker"], "AAPL")
        self.assertEqual(tracked[0]["total_predictions"], 1)

        # Test delete prediction
        deleted = self.db.delete_prediction(pred_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.db.get_prediction_by_id(pred_id))
        self.assertEqual(len(self.db.get_stock_predictions("AAPL")), 0)

    def test_save_and_retrieve_screening_run_with_nan(self) -> None:
        results_payload = {
            "top_bullish": [
                {
                    "ticker": "MSFT",
                    "cum2d_ret_50": 0.035,
                    "invalid_metric": float("nan"),
                }
            ],
            "top_bearish": [],
            "top_volatility": [],
        }

        hit_preds = [
            {
                "ticker": "MSFT",
                "name": "Microsoft Corp",
                "sector": "Technology",
                "universe": "sp500",
                "is_index": False,
                "anchor_date": "2026-03-01",
                "anchor_close": 400.0,
                "screen_category": "bullish",
                "milestones": [1, 2],
                "quantiles": [0.1, 0.5, 0.9],
                "cone": {"daily": [np.nan]},
                "summary": {"score": float("nan")},
            }
        ]

        run = self.db.save_screening_run(
            universe="sp500",
            top_n=10,
            min_price=10.0,
            interval="1d",
            total_candidates=500,
            total_screened=495,
            checkpoint_path="/path/to/checkpoint.pt",
            duration_sec=12.34,
            results_payload=results_payload,
            hit_predictions=hit_preds,
        )

        run_id = run["id"]
        self.assertIsNotNone(run_id)

        fetched_run = self.db.get_screening_run(run_id)
        self.assertIsNotNone(fetched_run)
        assert fetched_run is not None
        self.assertEqual(fetched_run["universe"], "sp500")
        self.assertIsNone(fetched_run["results"]["top_bullish"][0]["invalid_metric"])

        # History list
        history = self.db.get_screening_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["bullish_count"], 1)

        # Hit prediction created
        preds = self.db.get_stock_predictions("MSFT")
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["source"], "screen")
        self.assertEqual(preds[0]["screen_run_id"], run_id)
        self.assertIsNone(preds[0]["summary"]["score"])

        # Delete screening run
        deleted = self.db.delete_screening_run(run_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.db.get_screening_run(run_id))
        self.assertEqual(len(self.db.get_stock_predictions("MSFT")), 0)

    def test_database_counts(self) -> None:
        counts_empty = self.db.get_counts()
        self.assertEqual(counts_empty["total_predictions"], 0)
        self.assertEqual(counts_empty["screening_runs"], 0)
        self.assertEqual(counts_empty["tracked_stocks"], 0)

        self.db.save_prediction(
            ticker="NVDA",
            anchor_date="2026-03-01",
            anchor_close=800.0,
            milestones=[1],
            quantiles=[0.5],
            cone={},
            summary={},
            source="manual",
        )

        counts = self.db.get_counts()
        self.assertEqual(counts["total_predictions"], 1)
        self.assertEqual(counts["manual_predictions"], 1)
        self.assertEqual(counts["screen_hit_predictions"], 0)
        self.assertEqual(counts["tracked_stocks"], 1)


if __name__ == "__main__":
    unittest.main()
