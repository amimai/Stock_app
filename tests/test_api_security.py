"""Security, CSRF, and Route Validation Tests for Stock_app.

Verifies:
1. Directory traversal rejection in path resolution and model selection.
2. Unknown model run_id handling (404 / FileNotFoundError) without DB misattribution.
3. Invalid route ID format integer conversion safety (400 Bad Request).
4. Market pool cache invalidation on model switching.
5. CSRF protection via Origin and Host header validation.
6. Quantile reconstruction without 0-width probability bands.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict
import unittest
import urllib.error
import urllib.request

import numpy as np

# Ensure Stock_app root is in sys.path
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app import StockAppRequestHandler, bridge, db


class TestApiSecurityAndValidation(unittest.TestCase):
    """End-to-end security and route validation test suite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StockAppRequestHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _make_request(
        self,
        path: str,
        method: str = "GET",
        data: Any = None,
        headers: Dict[str, str] | None = None,
    ) -> tuple[int, Dict[str, Any]]:
        url = f"http://127.0.0.1:{self.port}{path}"
        req_headers = headers.copy() if headers else {}
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            if "Content-Type" not in req_headers:
                req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp.status, resp_data
        except urllib.error.HTTPError as err:
            err_data = json.loads(err.read().decode("utf-8"))
            return err.code, err_data

    # 1. Invalid Route ID Format Tests
    def test_get_screen_invalid_id_format(self) -> None:
        status, data = self._make_request("/api/screen/not-an-int")
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "Invalid ID format")

    def test_get_prediction_invalid_id_format(self) -> None:
        status, data = self._make_request("/api/prediction/abc123xyz")
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "Invalid ID format")

    def test_delete_prediction_invalid_id_format(self) -> None:
        status, data = self._make_request("/api/prediction/invalid-id", method="DELETE")
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "Invalid ID format")

    def test_delete_screen_invalid_id_format(self) -> None:
        status, data = self._make_request("/api/screen/invalid-id", method="DELETE")
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "Invalid ID format")

    # 2. Directory Traversal Rejection Tests
    def test_path_traversal_in_select_model(self) -> None:
        status, data = self._make_request(
            "/api/models/select",
            method="POST",
            data={"run_id": "../../etc/passwd", "promote_to_root": False},
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(data["ok"])
        self.assertIn("Path traversal detected", data["error"])

    def test_path_traversal_absolute_path_in_select_model(self) -> None:
        status, data = self._make_request(
            "/api/models/select",
            method="POST",
            data={"run_id": "/tmp/malicious.pt", "promote_to_root": False},
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertFalse(data["ok"])
        self.assertIn("Path traversal detected", data["error"])

    def test_get_model_detail_traversal(self) -> None:
        status, data = self._make_request("/api/models/..%2F..%2Fetc%2Fpasswd")
        self.assertEqual(status, HTTPStatus.NOT_FOUND)
        self.assertFalse(data["ok"])
        self.assertIn("not found", data["error"])

    # 3. Unknown run_id Handling & DB Attribution Safety
    def test_unknown_run_id_returns_404_in_select(self) -> None:
        status, data = self._make_request(
            "/api/models/select",
            method="POST",
            data={"run_id": "totally_fake_run_12345", "promote_to_root": False},
        )
        self.assertEqual(status, HTTPStatus.NOT_FOUND)
        self.assertFalse(data["ok"])
        self.assertIn("not found", data["error"])

    def test_unknown_run_id_returns_404_in_get_detail(self) -> None:
        status, data = self._make_request("/api/models/totally_fake_run_12345")
        self.assertEqual(status, HTTPStatus.NOT_FOUND)
        self.assertFalse(data["ok"])
        self.assertIn("not found", data["error"])

    def test_unknown_run_id_in_predict_no_misattribution(self) -> None:
        # Attempt prediction with non-existent model run
        fake_run = "fake_model_run_injection_attempt"
        status, data = self._make_request(
            "/api/predict",
            method="POST",
            data={"ticker": "AAPL", "model_run": fake_run},
        )
        self.assertEqual(status, HTTPStatus.NOT_FOUND)
        self.assertFalse(data["ok"])

        # Check DB to ensure no prediction was saved under the fake run_id
        saved_preds = db.get_prediction_history(model_run=fake_run)
        self.assertEqual(len(saved_preds), 0)

    # 4. CSRF Origin / Host Header Validation Tests
    def test_csrf_external_origin_rejected(self) -> None:
        # POST with external Origin should be rejected with 403 Forbidden
        headers = {"Origin": "http://evil.com"}
        status, data = self._make_request(
            "/api/models/select",
            method="POST",
            data={"run_id": "default", "promote_to_root": False},
            headers=headers,
        )
        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertFalse(data["ok"])
        self.assertIn("CSRF check failed", data["error"])

    def test_csrf_delete_external_origin_rejected(self) -> None:
        headers = {"Origin": "https://malicious-site.org"}
        status, data = self._make_request(
            "/api/prediction/1",
            method="DELETE",
            headers=headers,
        )
        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertFalse(data["ok"])
        self.assertIn("CSRF check failed", data["error"])

    def test_csrf_matching_origin_allowed(self) -> None:
        # POST with matching Origin header should succeed
        headers = {"Origin": f"http://127.0.0.1:{self.port}"}
        status, data = self._make_request(
            "/api/models/select",
            method="POST",
            data={"run_id": "default", "promote_to_root": False},
            headers=headers,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertTrue(data["ok"])

    def test_csrf_loopback_localhost_allowed(self) -> None:
        # POST with localhost Origin against 127.0.0.1 host should succeed
        headers = {"Origin": f"http://localhost:{self.port}"}
        status, data = self._make_request(
            "/api/models/select",
            method="POST",
            data={"run_id": "default", "promote_to_root": False},
            headers=headers,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertTrue(data["ok"])

    # 5. Model Switching Cache Invalidation
    def test_market_pool_cache_invalidation_on_model_switch(self) -> None:
        # Seed the cache with a dummy element
        bridge._market_pool_cache = [("DUMMY", np.ones((60, 16), dtype=np.float32))]
        self.assertIsNotNone(bridge._market_pool_cache)

        # Switch active model
        bridge.set_active_model("default")
        self.assertTrue(bridge._market_pool_cache is None or len(bridge._market_pool_cache) == 0)

    # 6. Quantile Non-zero Band Width Test
    def test_screen_row_to_prediction_payload_no_zero_width_bands(self) -> None:
        dummy_row = {
            "ticker": "TEST",
            "name": "Test Co",
            "sector": "Technology",
            "universe": "sp500",
            "last_date": "2026-03-01",
            "last_close": 100.0,
            "day1_ret_50": 0.02,
            "cum2d_ret_10": 0.01,
            "cum2d_ret_50": 0.03,
            "cum2d_ret_90": 0.05,
            "volatility_spread": 0.04,
        }
        milestones = [1, 2]
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        payload = bridge._screen_row_to_prediction_payload(
            row=dummy_row,
            milestones=milestones,
            quantiles=quantiles,
            screen_category="bullish",
            model_type="patchtst",
            lookback=60,
        )

        cone = payload["cone"]
        for milestone_item in cone["milestones"]:
            returns_map = milestone_item["returns"]
            prices_map = milestone_item["prices"]

            # Verify that 50% band (q25 to q75) has non-zero width
            if "q25" in returns_map and "q75" in returns_map:
                band_50_width = returns_map["q75"] - returns_map["q25"]
                self.assertGreater(band_50_width, 0.0)

            # Verify that 80% band (q10 to q90) has non-zero width
            if "q10" in returns_map and "q90" in returns_map:
                band_80_width = returns_map["q90"] - returns_map["q10"]
                self.assertGreater(band_80_width, 0.0)

            # Prices should strictly ascend across quantiles
            self.assertLess(prices_map["q10"], prices_map["q50"])
            self.assertLess(prices_map["q50"], prices_map["q90"])


if __name__ == "__main__":
    unittest.main()
