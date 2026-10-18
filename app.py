"""Web server and REST API for Stock_app.

Provides a zero-extra-dependency multithreaded HTTP/JSON server that runs
directly in the ML_stock Python environment (`python app.py`).
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import mimetypes
from pathlib import Path
import traceback
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

from config import APP_DIR, DB_PATH, HOST, PORT
from database import StockAppDatabase
from ml_bridge import MLStockBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stock_app")

STATIC_DIR: Path = APP_DIR / "static"

db = StockAppDatabase(DB_PATH)
bridge = MLStockBridge(db)


def _sanitize_json_value(obj: Any) -> Any:
    """Recursively walks dicts, lists, and tuples, replacing non-finite floats and NaNs with None."""
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
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(v) for v in obj]
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_json_value(v) for v in obj.tolist()]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _sanitize_json_value(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return _sanitize_json_value(obj.tolist())
        except Exception:
            pass
    return obj


class StockAppRequestHandler(BaseHTTPRequestHandler):
    """Handles REST API requests and serves static frontend files."""

    server_version = "StockAppHTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        sanitized = _sanitize_json_value(payload)
        body = json.dumps(sanitized, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self, message: str, status: int = HTTPStatus.BAD_REQUEST, details: Optional[str] = None
    ) -> None:
        payload: Dict[str, Any] = {"ok": False, "error": message}
        if details:
            payload["details"] = details
        self._send_json(payload, status=status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def _serve_static(self, rel_path: str) -> None:
        target = (STATIC_DIR / rel_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._send_error_json("Not found", status=HTTPStatus.NOT_FOUND)
            return

        ctype, _ = mimetypes.guess_type(str(target))
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        try:
            if path == "/":
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                return self._serve_static(rel)

            if path == "/api/overview":
                source_filter = qs.get("source", ["all"])[0]
                return self._send_json(
                    {
                        "ok": True,
                        "status": bridge.get_system_status(),
                        "tracked_stocks": db.get_tracked_stocks(),
                        "prediction_history": db.get_prediction_history(
                            limit=int(qs.get("pred_limit", ["150"])[0]),
                            source=source_filter,
                        ),
                        "screening_history": db.get_screening_history(
                            limit=int(qs.get("screen_limit", ["50"])[0])
                        ),
                    }
                )

            if path == "/api/parquet/search":
                q = qs.get("q", [""])[0]
                universe = qs.get("universe", ["all"])[0]
                limit = int(qs.get("limit", ["150"])[0])
                items = bridge.search_parquet_tickers(
                    query=q, universe=universe, limit=limit
                )
                return self._send_json({"ok": True, "tickers": items, "count": len(items)})

            if path.startswith("/api/stock/"):
                ticker = unquote(path[len("/api/stock/") :]).strip().upper()
                bars_limit = int(qs.get("bars", ["360"])[0])
                auto_fetch = qs.get("auto_fetch", ["0"])[0] in ("1", "true", "yes")
                detail = bridge.get_stock_detail(
                    ticker=ticker, bars_limit=bars_limit, auto_fetch=auto_fetch
                )
                return self._send_json({"ok": True, "stock": detail})

            if path.startswith("/api/screen/"):
                run_id = int(path[len("/api/screen/") :])
                run_data = db.get_screening_run(run_id)
                if not run_data:
                    return self._send_error_json(
                        f"Screening run #{run_id} not found.", status=HTTPStatus.NOT_FOUND
                    )
                return self._send_json({"ok": True, "screening_run": run_data})

            if path.startswith("/api/prediction/"):
                pred_id = int(path[len("/api/prediction/") :])
                pred = db.get_prediction_by_id(pred_id)
                if not pred:
                    return self._send_error_json(
                        f"Prediction #{pred_id} not found.", status=HTTPStatus.NOT_FOUND
                    )
                return self._send_json({"ok": True, "prediction": pred})

            self._send_error_json(f"Unknown endpoint: {path}", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.error("GET %s failed: %s", path, exc)
            self._send_error_json(
                str(exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                details=traceback.format_exc(),
            )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            body = self._read_json_body()

            if path == "/api/predict":
                ticker = str(body.get("ticker", "")).strip().upper()
                if not ticker:
                    return self._send_error_json("Field 'ticker' is required.")
                as_of_date = body.get("as_of_date") or None
                if as_of_date:
                    as_of_date = str(as_of_date).strip()[:10]
                num_samples = int(body.get("num_samples", 1))
                mc_dropout = bool(body.get("mc_dropout", False))
                market_resampling = bool(body.get("market_resampling", True))
                refresh_data = bool(body.get("refresh_data", False))

                pred = bridge.run_stock_prediction(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    num_samples=num_samples,
                    mc_dropout=mc_dropout,
                    market_resampling=market_resampling,
                    refresh_data=refresh_data,
                )
                return self._send_json({"ok": True, "prediction": pred})

            if path == "/api/screen":
                universe = str(body.get("universe", "all")).strip().lower()
                top_n = int(body.get("top_n", 10))
                min_price = float(body.get("min_price", 5.0))
                refresh_data = bool(body.get("refresh_data", False))
                save_all = bool(body.get("save_all_screened", False))

                run_data = bridge.run_market_screening(
                    universe=universe,
                    top_n=top_n,
                    min_price=min_price,
                    refresh_data=refresh_data,
                    save_all_screened=save_all,
                )
                return self._send_json({"ok": True, "screening_run": run_data})

            if path == "/api/cache/refresh":
                tickers = body.get("tickers") or []
                if isinstance(tickers, str):
                    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
                force_refresh = bool(body.get("force_refresh", False))
                res = bridge.update_cached_data(
                    tickers=tickers, interval="1d", force_refresh=force_refresh
                )
                return self._send_json({"ok": True, "result": res})

            self._send_error_json(f"Unknown POST endpoint: {path}", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.error("POST %s failed: %s", path, exc)
            self._send_error_json(
                str(exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                details=traceback.format_exc(),
            )

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path.startswith("/api/prediction/"):
                pred_id = int(path[len("/api/prediction/") :])
                deleted = db.delete_prediction(pred_id)
                return self._send_json({"ok": True, "deleted": deleted})

            if path.startswith("/api/screen/"):
                run_id = int(path[len("/api/screen/") :])
                deleted = db.delete_screening_run(run_id)
                return self._send_json({"ok": True, "deleted": deleted})

            self._send_error_json(f"Unknown DELETE endpoint: {path}", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_json(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock_app Web Application Server")
    parser.add_argument("--host", default=HOST, help=f"Bind host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Bind port (default: {PORT})")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), StockAppRequestHandler)
    print("=" * 72)
    print(f"  Stock_app Web UI running at: http://{args.host}:{args.port}")
    print(f"  Connected ML_stock repo    : {bridge.data_dir.parent.parent}")
    print(f"  Internal SQLite Database   : {DB_PATH}")
    print("=" * 72)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Stock_app server...")
        server.server_close()


if __name__ == "__main__":
    main()
