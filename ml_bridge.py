"""Bridge module connecting Stock_app to the external ML_stock repository.

All Parquet cache reads, metadata queries, and cache updates are strictly
delegated to ML_stock's `YahooParquetDataLoader` (src/data_loader.py).
Model inference and market screening hook directly into ML_stock's PatchTST /
MarketQueryPatchTST and MarketScreener classes.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from cone_utils import (
    build_cone_and_summary,
    enrich_prediction_with_realized,
    to_yyyy_mm_dd,
)
from config import (
    ML_STOCK_CHECKPOINT,
    ML_STOCK_CHECKPOINTS_DIR,
    ML_STOCK_CONFIG_DIR,
    ML_STOCK_DATA_DIR,
    ML_STOCK_DIR,
)
from database import StockAppDatabase

# Ensure ML_stock is importable without modifying its repository
if str(ML_STOCK_DIR) not in sys.path:
    sys.path.insert(0, str(ML_STOCK_DIR))

import src.metadata as ml_metadata  # noqa: E402
import src.universe as ml_universe  # noqa: E402
from predict import enable_mc_dropout, load_cached_market_context_pool  # noqa: E402
from screen import MarketScreener, load_universe_metadata  # noqa: E402
from src.data_loader import YahooParquetDataLoader  # noqa: E402
from src.features import (  # noqa: E402
    FEATURE_COLUMNS,
    LEGACY_FEATURE_COLUMNS,
    FeatureExtractor,
)
from src.metadata import StockMetadataRegistry  # noqa: E402
from src.models.market_query import MarketQueryPatchTST  # noqa: E402
from src.models.patchtst import PatchTST  # noqa: E402
from src.universe import UniverseManager  # noqa: E402

# Point ML_stock's default config dir to its absolute path so calls work from Stock_app CWD
ml_universe.DEFAULT_CONFIG_DIR = ML_STOCK_CONFIG_DIR
ml_metadata.DEFAULT_CONFIG_DIR = ML_STOCK_CONFIG_DIR

logger = logging.getLogger(__name__)


class MLStockBridge:
    """Interfaces with ML_stock data_loader.py, PatchTST models, and MarketScreener."""

    def __init__(self, db: StockAppDatabase) -> None:
        self.db = db
        self.data_dir = ML_STOCK_DATA_DIR
        self.config_dir = ML_STOCK_CONFIG_DIR
        self.checkpoints_dir = ML_STOCK_CHECKPOINTS_DIR
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        # Delegate all Parquet operations to ML_stock's YahooParquetDataLoader
        self.loader = YahooParquetDataLoader(data_dir=self.data_dir)
        self.universe_mgr = UniverseManager(config_dir=self.config_dir)
        self.meta_lookup: Dict[str, Dict[str, str]] = load_universe_metadata(
            self.config_dir
        )

        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._screeners: Dict[str, Tuple[MarketScreener, float]] = {}
        self._market_pool_cache: Optional[List[Tuple[str, np.ndarray]]] = None

        # Resolve active model run from database settings or default to root
        saved_active = self.db.get_setting("active_run_id", "default")
        self.active_run_id = saved_active or "default"
        self.checkpoint_path = self._resolve_checkpoint_path(self.active_run_id)

    def _resolve_checkpoint_path(self, run_id: Optional[str] = None) -> Path:
        """Resolves the checkpoint file Path for a given run_id or default."""
        clean_id = (run_id or "").strip()
        if not clean_id or clean_id in ("default", "root", "active"):
            return ML_STOCK_CHECKPOINT.resolve()

        # Reject path traversal attempts (e.g. '..' or absolute paths outside checkpoints_dir)
        if ".." in clean_id or clean_id.startswith("/") or clean_id.startswith("\\"):
            raise ValueError(f"Path traversal detected in model run_id: '{run_id}'")

        checkpoints_dir_resolved = self.checkpoints_dir.resolve()
        cand = (self.checkpoints_dir / clean_id / "patchtst_best.pt").resolve()

        # Verify that candidate is strictly within checkpoints_dir
        try:
            cand.relative_to(checkpoints_dir_resolved)
        except ValueError:
            raise ValueError(f"Path traversal detected in model run_id: '{run_id}'")

        if cand.exists() and cand.is_file():
            return cand

        raise FileNotFoundError(f"Model checkpoint '{run_id}' not found")

    def _get_screener_for_path(self, ckpt_path: Path) -> MarketScreener:
        """Lazily loads or retrieves cached MarketScreener for a specific checkpoint file."""
        with self._lock:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
            mtime = ckpt_path.stat().st_mtime
            key = str(ckpt_path.resolve())
            if key in self._screeners:
                scr, cached_mtime = self._screeners[key]
                if cached_mtime == mtime:
                    return scr

            logger.info(f"Loading ML_stock model from {ckpt_path}...")
            screener = MarketScreener(
                checkpoint_path=ckpt_path,
                data_dir=self.data_dir,
                config_dir=self.config_dir,
                device_name=self.device_name,
            )
            self._screeners[key] = (screener, mtime)
            return screener

    def _ensure_screener(self) -> MarketScreener:
        """Lazily loads or reloads MarketScreener for the current active checkpoint."""
        return self._get_screener_for_path(self.checkpoint_path)

    def _resolve_screener_and_run_id(
        self, model_run: Optional[str] = None
    ) -> Tuple[MarketScreener, Path, str]:
        """Resolves screener, checkpoint path, and run_id given an optional model_run override."""
        target_run_id = (model_run or "").strip()
        if not target_run_id or target_run_id in ("active", ""):
            target_run_id = self.active_run_id
            target_path = self.checkpoint_path
        elif target_run_id in ("default", "root"):
            target_run_id = "default"
            target_path = ML_STOCK_CHECKPOINT.resolve()
        else:
            target_path = self._resolve_checkpoint_path(target_run_id)

        if not target_path.exists():
            raise FileNotFoundError(f"Model checkpoint '{target_run_id}' not found at {target_path}")

        screener = self._get_screener_for_path(target_path)
        return screener, target_path, target_run_id

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Lists all available model runs from runs_index.csv and checkpoints directory."""
        models: List[Dict[str, Any]] = []
        runs_index_file = self.checkpoints_dir / "runs_index.csv"
        seen_ids = set()

        if runs_index_file.exists():
            try:
                with open(runs_index_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        r_id = r.get("run_id", "").strip()
                        if not r_id or r_id in seen_ids:
                            continue
                        seen_ids.add(r_id)
                        ckpt_p = Path(
                            r.get(
                                "checkpoint_path",
                                str(self.checkpoints_dir / r_id / "patchtst_best.pt"),
                            )
                        )

                        # Human-friendly badge
                        model_type = r.get("model_type", "market_query")
                        short_l = int(r.get("short_layers", 1) or 1)
                        num_l = int(r.get("num_layers", 4) or 4)
                        if "3head" in r_id or "3head" in r.get("run_tag", ""):
                            badge = f"3-Head {num_l}L"
                        elif "1head" in r_id or "1head" in r.get("run_tag", ""):
                            badge = f"1-Head {num_l}L"
                        else:
                            badge = f"{num_l}L"

                        is_act = (r_id == self.active_run_id)

                        def _parse_float(k: str) -> Optional[float]:
                            v = r.get(k)
                            if v is not None and v != "":
                                try:
                                    return round(float(v), 5)
                                except Exception:
                                    return None
                            return None

                        models.append(
                            {
                                "run_id": r_id,
                                "run_tag": r.get("run_tag", r_id),
                                "model_type": model_type,
                                "num_layers": num_l,
                                "market_layers": int(r.get("market_layers", 4) or 4),
                                "short_layers": short_l,
                                "long_layers": int(r.get("long_layers", 1) or 1),
                                "vol_layers": int(r.get("vol_layers", 1) or 1),
                                "best_epoch": int(r.get("best_epoch", 1) or 1),
                                "epochs_ran": int(r.get("epochs_ran", 1) or 1),
                                "val_loss": _parse_float("val_loss"),
                                "val_score": _parse_float("val_score"),
                                "val_dir_acc": _parse_float("val_dir_acc"),
                                "val_ic": _parse_float("val_ic"),
                                "test_loss": _parse_float("test_loss"),
                                "test_score": _parse_float("test_score"),
                                "test_mae": _parse_float("test_mae"),
                                "test_dir_acc": _parse_float("test_dir_acc"),
                                "test_ic": _parse_float("test_ic"),
                                "test_cov80": _parse_float("test_cov80"),
                                "test_calib_mae": _parse_float("test_calib_mae"),
                                "duration_sec": _parse_float("duration_sec"),
                                "checkpoint_path": str(ckpt_p),
                                "badge_label": badge,
                                "display_name": f"{badge} • {r_id}",
                                "is_active": is_act,
                                "exists": ckpt_p.exists(),
                            }
                        )
            except Exception as exc:
                logger.error(f"Failed parsing runs_index.csv: {exc}")

        # Also discover any subfolders not in runs_index
        if self.checkpoints_dir.exists():
            for sub in sorted(self.checkpoints_dir.iterdir(), reverse=True):
                if sub.is_dir() and sub.name not in seen_ids:
                    pt = sub / "patchtst_best.pt"
                    if pt.exists():
                        seen_ids.add(sub.name)
                        models.append(
                            {
                                "run_id": sub.name,
                                "run_tag": sub.name,
                                "model_type": "market_query",
                                "badge_label": "Archived",
                                "display_name": sub.name,
                                "checkpoint_path": str(pt),
                                "is_active": (sub.name == self.active_run_id),
                                "exists": True,
                            }
                        )

        # Include default root option
        if ML_STOCK_CHECKPOINT.exists():
            models.insert(
                0,
                {
                    "run_id": "default",
                    "run_tag": "active_default",
                    "model_type": "active",
                    "badge_label": "Active Root",
                    "display_name": "Default Root Checkpoint (patchtst_best.pt)",
                    "checkpoint_path": str(ML_STOCK_CHECKPOINT),
                    "is_active": (self.active_run_id == "default"),
                    "exists": True,
                },
            )

        return models

    def get_model_detail(self, run_id: str) -> Dict[str, Any]:
        """Returns full summary, training history rows, and ASCII report for a model run."""
        clean_id = (run_id or "").strip()
        if not clean_id:
            clean_id = self.active_run_id

        if clean_id in ("default", "root"):
            folder = self.checkpoints_dir
            ckpt_path = ML_STOCK_CHECKPOINT
        else:
            # Reject path traversal
            if ".." in clean_id or clean_id.startswith("/") or clean_id.startswith("\\"):
                return {"ok": False, "exists": False, "error": f"Model run '{run_id}' not found"}

            checkpoints_dir_resolved = self.checkpoints_dir.resolve()
            folder = (self.checkpoints_dir / clean_id).resolve()
            try:
                folder.relative_to(checkpoints_dir_resolved)
            except ValueError:
                return {"ok": False, "exists": False, "error": f"Model run '{run_id}' not found"}

            if not folder.exists() or not folder.is_dir():
                return {"ok": False, "exists": False, "error": f"Model run '{run_id}' not found"}

            try:
                ckpt_path = self._resolve_checkpoint_path(clean_id)
            except (FileNotFoundError, ValueError):
                ckpt_path = folder / "patchtst_best.pt"

        summary_data: Dict[str, Any] = {}
        report_txt = ""
        history_rows: List[Dict[str, Any]] = []

        summary_path = folder / "run_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed loading run_summary.json: {exc}")

        report_path = folder / "report.txt"
        if report_path.exists():
            try:
                report_txt = report_path.read_text(encoding="utf-8")
            except Exception:
                pass

        hist_path = folder / "history.csv"
        if hist_path.exists():
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    history_rows = list(csv.DictReader(f))
            except Exception:
                pass

        has_checkpoint = ckpt_path.exists()
        if has_checkpoint and not summary_data:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                hp = ckpt.get("hyperparameters", {})
                summary_data = {
                    "run_id": clean_id,
                    "architecture": hp,
                    "best_val_loss": ckpt.get("best_val_loss"),
                    "test_metrics": ckpt.get("test_metrics"),
                }
            except Exception:
                pass

        # Normalize metrics in summary_data so both test_evaluation and test_metrics exist with aliases
        test_eval = summary_data.get("test_evaluation") or summary_data.get("test_metrics") or {}
        merged_eval: Dict[str, Any] = {}
        if isinstance(test_eval, dict):
            merged_eval.update(test_eval)
        if isinstance(summary_data.get("test_metrics"), dict):
            merged_eval.update(summary_data["test_metrics"])
        if isinstance(summary_data.get("test_evaluation"), dict):
            merged_eval.update(summary_data["test_evaluation"])

        alias_map = [
            ("loss", "test_loss"),
            ("score", "test_score"),
            ("mae", "test_mae"),
            ("directional_accuracy", "test_dir_acc"),
            ("dir_acc", "test_dir_acc"),
            ("information_coefficient", "test_ic"),
            ("ic", "test_ic"),
            ("coverage_50pct_band", "test_cov50"),
            ("coverage_80pct_band", "test_cov80"),
            ("coverage_90pct_band", "test_cov90"),
            ("calibration_mae", "test_calib_mae"),
        ]
        for long_k, short_k in alias_map:
            if short_k in summary_data and short_k not in merged_eval:
                merged_eval[short_k] = summary_data[short_k]
            if long_k in summary_data and long_k not in merged_eval:
                merged_eval[long_k] = summary_data[long_k]
            if long_k in merged_eval and short_k not in merged_eval:
                merged_eval[short_k] = merged_eval[long_k]
            elif short_k in merged_eval and long_k not in merged_eval:
                merged_eval[long_k] = merged_eval[short_k]

        if merged_eval:
            summary_data["test_evaluation"] = merged_eval
            summary_data["test_metrics"] = merged_eval

        return {
            "ok": True,
            "run_id": clean_id,
            "checkpoint_path": str(ckpt_path),
            "exists": has_checkpoint,
            "is_active": (clean_id == self.active_run_id),
            "summary": summary_data,
            "report_txt": report_txt,
            "history_rows": history_rows,
        }

    def set_active_model(
        self, run_id: str, promote_to_root: bool = False
    ) -> Dict[str, Any]:
        """Switches the active model used for subsequent predictions and screening calls."""
        with self._lock:
            clean_id = (run_id or "").strip()
            if not clean_id:
                raise ValueError("Model run_id cannot be empty")

            if clean_id in ("default", "root"):
                self.active_run_id = "default"
                self.checkpoint_path = ML_STOCK_CHECKPOINT
            else:
                # Reject path traversal
                if ".." in clean_id or clean_id.startswith("/") or clean_id.startswith("\\"):
                    raise ValueError(f"Path traversal detected in model run_id: '{run_id}'")

                checkpoints_dir_resolved = self.checkpoints_dir.resolve()
                target_path = (self.checkpoints_dir / clean_id / "patchtst_best.pt").resolve()
                try:
                    target_path.relative_to(checkpoints_dir_resolved)
                except ValueError:
                    raise ValueError(f"Path traversal detected in model run_id: '{run_id}'")

                if not target_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint for run '{clean_id}' not found at {target_path}"
                    )
                self.active_run_id = clean_id
                self.checkpoint_path = target_path

            self.db.set_setting("active_run_id", self.active_run_id)

            # Invalidate market pool cache on model switch
            if self._market_pool_cache is not None:
                self._market_pool_cache.clear()
            self._market_pool_cache = None

            # Reload / cache screener for active path
            screener = self._ensure_screener()

            if promote_to_root and clean_id not in ("default", "root"):
                # Validate source is strictly within checkpoints_dir
                src_path = self.checkpoint_path.resolve()
                try:
                    src_path.relative_to(self.checkpoints_dir.resolve())
                except ValueError:
                    raise ValueError(f"Cannot promote checkpoint outside checkpoints dir: {src_path}")

                logger.info(
                    f"Promoting checkpoint {self.checkpoint_path} to {ML_STOCK_CHECKPOINT}..."
                )
                # Make backup (.pt.bak) before copying
                if ML_STOCK_CHECKPOINT.exists():
                    backup_path = ML_STOCK_CHECKPOINT.with_suffix(".pt.bak")
                    shutil.copy2(ML_STOCK_CHECKPOINT, backup_path)

                shutil.copy2(self.checkpoint_path, ML_STOCK_CHECKPOINT)
                src_hist = self.checkpoints_dir / clean_id / "history.csv"
                dst_hist = self.checkpoints_dir / "history.csv"
                if src_hist.exists():
                    if dst_hist.exists():
                        shutil.copy2(dst_hist, dst_hist.with_suffix(".csv.bak"))
                    shutil.copy2(src_hist, dst_hist)

                # Invalidate screeners cache for root checkpoint path
                root_resolved = str(ML_STOCK_CHECKPOINT.resolve())
                self._screeners.pop(root_resolved, None)
                self._screeners.pop(str(ML_STOCK_CHECKPOINT), None)

            return {
                "ok": True,
                "active_run_id": self.active_run_id,
                "checkpoint_path": str(self.checkpoint_path),
                "model_type": screener.hp.get("model_type", "patchtst"),
                "promoted_to_root": promote_to_root,
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Returns ML_stock connection status, model info, and Parquet cache stats."""
        stored = self.loader.list_stored_tickers(interval="1d")
        universes = self.universe_mgr.list_universes()
        ckpt_exists = self.checkpoint_path.exists()
        model_info: Dict[str, Any] = {"checkpoint_exists": ckpt_exists}
        if ckpt_exists:
            try:
                scr = self._ensure_screener()
                hp = scr.hp
                model_info.update(
                    {
                        "model_type": hp.get("model_type", "patchtst"),
                        "seq_len": hp.get("seq_len", 60),
                        "horizon": hp.get("horizon", 4),
                        "milestones": hp.get("milestones", [1, 2]),
                        "quantiles": hp.get("quantiles", [0.1, 0.5, 0.9]),
                        "d_model": hp.get("d_model"),
                        "device": self.device_name,
                        "active_run_id": self.active_run_id,
                    }
                )
            except Exception as exc:
                model_info["error"] = str(exc)

        return {
            "ml_stock_dir": str(ML_STOCK_DIR),
            "data_dir": str(self.data_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "active_run_id": self.active_run_id,
            "parquet_ticker_count": len(stored),
            "universes": universes,
            "model": model_info,
            "db_counts": self.db.get_counts(),
        }

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        sym = ticker.upper().strip()
        meta = self.meta_lookup.get(sym, {})
        return {
            "ticker": sym,
            "name": meta.get("name", sym),
            "sector": meta.get("sector", "Aggregate Index" if sym.startswith("^") else "Equities"),
            "universe": meta.get("universe", "indexes" if sym.startswith("^") else "custom"),
            "is_index": sym.startswith("^") or meta.get("universe") == "indexes",
        }

    def search_parquet_tickers(
        self,
        query: str = "",
        universe: str = "all",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Searches tickers stored in ML_stock's Parquet memory with metadata."""
        stored_set = set(self.loader.list_stored_tickers(interval="1d"))
        tracked_map = {
            item["ticker"]: item for item in self.db.get_tracked_stocks()
        }

        q = query.strip().upper()
        u_filter = universe.strip().lower()

        results: List[Dict[str, Any]] = []
        for sym in sorted(stored_set):
            info = self.get_ticker_info(sym)
            if u_filter and u_filter != "all" and info["universe"].lower() != u_filter:
                continue
            if q:
                name_upper = info["name"].upper()
                sec_upper = info["sector"].upper()
                if q not in sym and q not in name_upper and q not in sec_upper:
                    continue

            t_stat = tracked_map.get(sym)
            results.append(
                {
                    **info,
                    "in_parquet": True,
                    "has_predictions": t_stat is not None,
                    "total_predictions": t_stat["total_predictions"] if t_stat else 0,
                    "screen_hit_count": t_stat["screen_hit_count"] if t_stat else 0,
                    "latest_anchor_date": t_stat["latest_anchor_date"] if t_stat else None,
                }
            )
            if len(results) >= limit:
                break

        # Sort exact prefix matches first when query is provided
        if q:
            results.sort(
                key=lambda x: (
                    0 if x["ticker"] == q else (1 if x["ticker"].startswith(q) else 2),
                    x["ticker"],
                )
            )
        return results

    def update_cached_data(
        self,
        tickers: List[str],
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Delegates cache updates strictly to ML_stock's data_loader.py."""
        cleaned = [t.upper().strip() for t in tickers if t and t.strip()]
        if not cleaned:
            raise ValueError("No ticker symbols provided for cache update.")

        if len(cleaned) == 1:
            sym = cleaned[0]
            df = self.loader.fetch_and_save(
                ticker=sym, interval=interval, force_refresh=force_refresh
            )
            meta = (
                self.loader.get_metadata(sym, interval=interval)
                if not df.empty
                else {}
            )
            return {
                "updated": [sym] if not df.empty else [],
                "rows": len(df),
                "metadata": meta,
            }

        dfs = self.loader.fetch_multiple(
            tickers=cleaned, interval=interval, force_refresh=force_refresh
        )
        updated = [k for k, v in dfs.items() if v is not None and not v.empty]
        return {
            "requested": len(cleaned),
            "updated_count": len(updated),
            "updated": updated,
        }

    def get_stock_detail(
        self,
        ticker: str,
        bars_limit: int = 320,
        auto_fetch: bool = False,
        model_run: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Loads stock OHLCV chart data via data_loader.py and enriches all stored predictions."""
        sym = ticker.upper().strip()
        info = self.get_ticker_info(sym)

        df = pd.DataFrame()
        parquet_meta: Dict[str, Any] = {}
        try:
            df = self.loader.load(ticker=sym, interval="1d", auto_fetch=auto_fetch)
            parquet_meta = self.loader.get_metadata(ticker=sym, interval="1d")
        except FileNotFoundError:
            if auto_fetch:
                df = self.loader.fetch_and_save(ticker=sym, interval="1d")
                if not df.empty:
                    parquet_meta = self.loader.get_metadata(ticker=sym, interval="1d")

        bars: List[Dict[str, Any]] = []
        if not df.empty:
            close_c = "Close" if "Close" in df.columns else df.columns[0]
            df = df.dropna(subset=[close_c]).copy()

        if not df.empty:
            sub_df = df.iloc[-bars_limit:] if bars_limit > 0 else df
            close_c = "Close" if "Close" in sub_df.columns else sub_df.columns[0]
            open_c = "Open" if "Open" in sub_df.columns else close_c
            high_c = "High" if "High" in sub_df.columns else close_c
            low_c = "Low" if "Low" in sub_df.columns else close_c
            vol_c = "Volume" if "Volume" in sub_df.columns else None

            for ts, row in sub_df.iterrows():
                c_val = float(row[close_c])
                o_val = float(row[open_c]) if pd.notna(row[open_c]) else c_val
                h_val = float(row[high_c]) if pd.notna(row[high_c]) else max(o_val, c_val)
                l_val = float(row[low_c]) if pd.notna(row[low_c]) else min(o_val, c_val)
                v_raw = row[vol_c] if vol_c is not None else None
                v_val = int(v_raw) if (v_raw is not None and pd.notna(v_raw) and math.isfinite(float(v_raw))) else 0

                bars.append(
                    {
                        "date": to_yyyy_mm_dd(ts),
                        "open": round(float(o_val), 4),
                        "high": round(float(h_val), 4),
                        "low": round(float(l_val), 4),
                        "close": round(float(c_val), 4),
                        "volume": int(v_val),
                    }
                )

        raw_preds = self.db.get_stock_predictions(sym, limit=200, model_run=model_run)
        enriched_preds = [
            enrich_prediction_with_realized(p, df) for p in raw_preds
        ]

        latest_bar = bars[-1] if bars else None
        prev_bar = bars[-2] if len(bars) >= 2 else None
        day_change_pct = None
        if latest_bar and prev_bar and prev_bar["close"] > 0:
            day_change_pct = round(
                (latest_bar["close"] - prev_bar["close"]) / prev_bar["close"], 6
            )

        return {
            **info,
            "in_parquet": not df.empty,
            "parquet_metadata": parquet_meta,
            "latest_bar": latest_bar,
            "day_change_pct": day_change_pct,
            "bars": bars,
            "predictions": enriched_preds,
        }

    def _load_market_pool_as_of(
        self,
        target_ticker: str,
        lookback: int,
        fe: FeatureExtractor,
        as_of_date: Optional[str] = None,
        max_pool_size: int = 128,
    ) -> List[Tuple[str, np.ndarray]]:
        """Loads market context pool, optionally sliced to as_of_date for historical predictions."""
        if not as_of_date:
            if self._market_pool_cache is None:
                self._market_pool_cache = load_cached_market_context_pool(
                    loader=self.loader,
                    target_ticker=target_ticker,
                    lookback=lookback,
                    fe=fe,
                    max_pool_size=max_pool_size,
                )
            return [
                item for item in self._market_pool_cache if item[0] != target_ticker.upper().strip()
            ]

        # Point-in-time slice for historical evaluation
        pool: List[Tuple[str, np.ndarray]] = []
        target_sym = target_ticker.upper().strip()
        stored = self.loader.list_stored_tickers(interval="1d")
        anchors = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        candidates = [s for s in anchors if s in stored and s != target_sym]
        rng = np.random.default_rng(42)
        rem = [s for s in stored if s not in candidates and s != target_sym]
        if rem:
            candidates.extend(rng.permutation(rem).tolist())

        active_cols = fe.feature_cols
        for sym in candidates[:max_pool_size]:
            try:
                df_c = self.loader.load(
                    ticker=sym, interval="1d", end=as_of_date, auto_fetch=False
                )
                if df_c.empty or len(df_c) < lookback:
                    continue
                feat_c = fe.transform(df_c, include_targets=False)
                clean_c = feat_c.dropna(subset=active_cols)
                if len(clean_c) >= lookback:
                    arr = clean_c.iloc[-lookback:][active_cols].values.astype(np.float32)
                    pool.append((sym, arr))
            except Exception:
                continue
        return pool

    def run_stock_prediction(
        self,
        ticker: str,
        as_of_date: Optional[str] = None,
        num_samples: int = 1,
        mc_dropout: bool = False,
        market_resampling: bool = True,
        refresh_data: bool = False,
        model_run: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs a PatchTST forecast on a stock, saves to Stock_app DB, and returns enriched result."""
        sym = ticker.upper().strip()
        screener, used_path, used_run_id = self._resolve_screener_and_run_id(model_run)
        model = screener.model
        hp = screener.hp
        lookback = screener.lookback
        quantiles = [float(q) for q in screener.quantiles]
        milestones = [int(m) for m in hp.get("milestones", [1, 2])]
        is_legacy_2day = milestones == [1, 2] and hp.get("milestones") is None

        if refresh_data:
            self.loader.fetch_and_save(ticker=sym, interval="1d")

        # Load full series via data_loader.py (auto_fetch=True if not yet cached)
        full_df = self.loader.load(ticker=sym, interval="1d", auto_fetch=True)
        if full_df.empty:
            raise ValueError(f"No historical market data available for '{sym}'.")

        # Slice if point-in-time historical prediction requested
        if as_of_date:
            eval_df = self.loader._filter_date_range(full_df, start=None, end=as_of_date)
        else:
            eval_df = full_df

        fe = FeatureExtractor(feature_cols=screener.feature_cols)
        feat_df = fe.transform(eval_df, include_targets=False)
        clean_df = feat_df.dropna(subset=screener.feature_cols)

        if len(clean_df) < lookback:
            raise ValueError(
                f"Insufficient clean history for {sym} (need {lookback} trading days, got {len(clean_df)})."
            )

        window_df = clean_df.iloc[-lookback:]
        x_np = window_df[screener.feature_cols].values.astype(np.float32)
        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)

        anchor_date = to_yyyy_mm_dd(window_df.index[-1])
        anchor_close = float(window_df["raw_close"].iloc[-1])

        is_market_query = isinstance(model, MarketQueryPatchTST)
        market_context_size = int(hp.get("market_context_size", 64))
        use_meta = bool(hp.get("use_metadata", False))
        meta_registry = StockMetadataRegistry(
            config_dir=self.config_dir, ticker_vocab=hp.get("ticker_vocab")
        )

        market_pool: List[Tuple[str, np.ndarray]] = []
        if is_market_query:
            market_pool = self._load_market_pool_as_of(
                target_ticker=sym,
                lookback=lookback,
                fe=fe,
                as_of_date=anchor_date if as_of_date else None,
                max_pool_size=max(100, market_context_size * 2),
            )

        num_samples = max(1, int(num_samples))
        rng = np.random.default_rng(42)
        sample_runs: List[np.ndarray] = []

        with self._inference_lock:
            if mc_dropout:
                enable_mc_dropout(model)
            else:
                model.eval()

            try:
                with torch.no_grad():
                    for _ in range(num_samples):
                        if is_market_query and len(market_pool) >= 4:
                            n_take = min(market_context_size - 1, len(market_pool))
                            if num_samples > 1 and market_resampling and len(market_pool) > n_take:
                                idx_sel = rng.choice(len(market_pool), size=n_take, replace=False)
                                sub_pool = [market_pool[i] for i in idx_sel]
                            else:
                                sub_pool = market_pool[:n_take]

                            full_tuples = [(sym, x_np)] + sub_pool
                            actual_n = len(full_tuples)
                            pad_count = max(0, market_context_size - actual_n)

                            m_arrs = [item[1] for item in full_tuples]
                            for _ in range(pad_count):
                                m_arrs.append(np.zeros_like(x_np))

                            m_stack = np.stack(m_arrs, axis=0)
                            x_m = torch.from_numpy(m_stack).unsqueeze(0).to(self.device)

                            mask_t = None
                            if pad_count > 0:
                                mask_np = np.zeros(market_context_size, dtype=bool)
                                mask_np[actual_n:] = True
                                mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

                            meta_m_t = None
                            meta_q_t = None
                            if use_meta:
                                m_meta = [meta_registry.get_metadata_indices(item[0]) for item in full_tuples]
                                for _ in range(pad_count):
                                    m_meta.append((0, 0, 0, 0))
                                meta_m_t = torch.tensor([m_meta], dtype=torch.long, device=self.device)
                                meta_q_t = torch.tensor(
                                    [[meta_registry.get_metadata_indices(sym)]],
                                    dtype=torch.long,
                                    device=self.device,
                                )

                            h_m = model.encode_market(
                                x_m, key_padding_mask=mask_t, meta_market=meta_m_t
                            )
                            preds = model.forward_with_market_state(
                                x.unsqueeze(1),
                                h_market=h_m,
                                key_padding_mask=mask_t,
                                meta_queries=meta_q_t,
                            ).squeeze(1)
                        else:
                            preds = model(x)

                        sample_runs.append(preds.squeeze(0).cpu().numpy())
            finally:
                model.eval()

        stacked = np.stack(sample_runs, axis=0)
        pred_returns = np.mean(stacked, axis=0)
        sample_std = np.std(stacked, axis=0) if num_samples > 1 else np.zeros_like(pred_returns)

        # Convert legacy incremental Day 2 return into cumulative log return if legacy checkpoint
        if is_legacy_2day and pred_returns.shape[0] >= 2:
            pred_returns = pred_returns.copy()
            pred_returns[1] = pred_returns[0] + pred_returns[1]

        cone, summary = build_cone_and_summary(
            anchor_date=anchor_date,
            anchor_close=anchor_close,
            milestones=milestones,
            quantiles=quantiles,
            pred_cum_returns=pred_returns,
            sample_std=sample_std,
        )

        info = self.get_ticker_info(sym)
        saved = self.db.save_prediction(
            ticker=sym,
            name=info["name"],
            sector=info["sector"],
            universe=info["universe"],
            is_index=info["is_index"],
            anchor_date=anchor_date,
            anchor_close=anchor_close,
            source="manual",
            model_type="market_query" if is_market_query else "patchtst",
            checkpoint_path=str(used_path),
            run_id=used_run_id,
            lookback=lookback,
            num_samples=num_samples,
            mc_dropout=mc_dropout,
            market_resampling=market_resampling,
            milestones=milestones,
            quantiles=quantiles,
            cone=cone,
            summary=summary,
        )

        return enrich_prediction_with_realized(saved, full_df)

    def _screen_row_to_prediction_payload(
        self,
        row: Dict[str, Any],
        milestones: List[int],
        quantiles: List[float],
        screen_category: str,
        model_type: str,
        lookback: int,
    ) -> Dict[str, Any]:
        """Reconstructs full probability cone and summary from a MarketScreener result row."""
        anchor_date = to_yyyy_mm_dd(row["last_date"])
        anchor_close = float(row["last_close"])
        is_legacy = milestones == [1, 2] and "ret_1d_50" not in row

        q_round = [round(float(q), 2) for q in quantiles]
        pred_mat = np.zeros((len(milestones), len(quantiles)), dtype=np.float64)

        def _interp_quantile(q: float, v10: float, v50: float, v90: float) -> float:
            min_spread = 0.002
            if v90 - v10 < min_spread:
                v10 = min(v10, v50 - min_spread / 2)
                v90 = max(v90, v50 + min_spread / 2)
            if v50 < v10:
                v50 = 0.5 * (v10 + v90)
            if v50 > v90:
                v50 = 0.5 * (v10 + v90)

            if q <= 0.10:
                return v10 - (v50 - v10) * ((0.10 - q) / 0.40)
            elif q <= 0.50:
                return v10 + (v50 - v10) * ((q - 0.10) / 0.40)
            elif q <= 0.90:
                return v50 + (v90 - v50) * ((q - 0.50) / 0.40)
            else:
                return v90 + (v90 - v50) * ((q - 0.90) / 0.40)

        if is_legacy:
            r1_50 = float(row.get("day1_ret_50", 0.0))
            c2_10 = float(row.get("cum2d_ret_10", 0.0))
            c2_50 = float(row.get("cum2d_ret_50", 0.0))
            c2_90 = float(row.get("cum2d_ret_90", 0.0))
            half_sp = 0.5 * (c2_90 - c2_10)
            if half_sp <= 0.001:
                half_sp = max(0.5 * abs(float(row.get("volatility_spread", 0.0))), 0.005)
            r1_10 = float(row.get("day1_ret_10", r1_50 - half_sp * 0.6))
            r1_90 = float(row.get("day1_ret_90", r1_50 + half_sp * 0.6))
            for q_i, q_val in enumerate(q_round):
                pred_mat[0, q_i] = _interp_quantile(q_val, r1_10, r1_50, r1_90)
                pred_mat[1, q_i] = _interp_quantile(q_val, c2_10, c2_50, c2_90)
        else:
            for h_i, m in enumerate(milestones):
                r10 = float(row.get(f"ret_{m}d_10", 0.0))
                r50 = float(row.get(f"ret_{m}d_50", 0.0))
                r90 = float(row.get(f"ret_{m}d_90", 0.0))
                if r10 == 0.0 and r90 == 0.0:
                    if m == 1 and "day1_ret_50" in row:
                        r50 = float(row.get("day1_ret_50", 0.0))
                        r10 = float(row.get("day1_ret_10", r50 - 0.01))
                        r90 = float(row.get("day1_ret_90", r50 + 0.01))
                    elif m == 2 and "cum2d_ret_50" in row:
                        r50 = float(row.get("cum2d_ret_50", 0.0))
                        r10 = float(row.get("cum2d_ret_10", r50 - 0.01))
                        r90 = float(row.get("cum2d_ret_90", r50 + 0.01))

                r25 = row.get(f"ret_{m}d_25")
                r75 = row.get(f"ret_{m}d_75")
                r05 = row.get(f"ret_{m}d_05")
                r95 = row.get(f"ret_{m}d_95")

                for q_i, q_val in enumerate(q_round):
                    if q_val == 0.10:
                        pred_mat[h_i, q_i] = r10
                    elif q_val == 0.50:
                        pred_mat[h_i, q_i] = r50
                    elif q_val == 0.90:
                        pred_mat[h_i, q_i] = r90
                    elif q_val == 0.25 and r25 is not None:
                        pred_mat[h_i, q_i] = float(r25)
                    elif q_val == 0.75 and r75 is not None:
                        pred_mat[h_i, q_i] = float(r75)
                    elif q_val == 0.05 and r05 is not None:
                        pred_mat[h_i, q_i] = float(r05)
                    elif q_val == 0.95 and r95 is not None:
                        pred_mat[h_i, q_i] = float(r95)
                    else:
                        pred_mat[h_i, q_i] = _interp_quantile(q_val, r10, r50, r90)

        cone, summary = build_cone_and_summary(
            anchor_date=anchor_date,
            anchor_close=anchor_close,
            milestones=milestones,
            quantiles=quantiles,
            pred_cum_returns=pred_mat,
        )
        return {
            "ticker": str(row["ticker"]).upper().strip(),
            "name": str(row.get("name") or row["ticker"]),
            "sector": str(row.get("sector") or "Equities"),
            "universe": str(row.get("universe") or "custom"),
            "is_index": bool(row.get("is_index", False)),
            "anchor_date": anchor_date,
            "anchor_close": anchor_close,
            "screen_category": screen_category,
            "model_type": model_type,
            "lookback": lookback,
            "milestones": milestones,
            "quantiles": quantiles,
            "cone": cone,
            "summary": summary,
        }

    def run_market_screening(
        self,
        universe: str = "all",
        top_n: int = 10,
        min_price: float = 5.0,
        interval: str = "1d",
        batch_size: int = 128,
        refresh_data: bool = False,
        save_all_screened: bool = False,
        model_run: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes ML_stock MarketScreener and persists the screening call and hit predictions."""
        t0 = time.time()
        screener, used_path, used_run_id = self._resolve_screener_and_run_id(model_run)
        u_clean = universe.lower().strip() or "all"
        top_n = max(1, min(100, int(top_n)))
        min_price = max(0.0, float(min_price))

        tickers = screener.resolve_tickers(u_clean)
        if not tickers:
            raise ValueError(f"No tickers resolved for universe '{u_clean}'.")

        if refresh_data:
            self.loader.fetch_multiple(
                tickers=tickers, interval=interval, force_refresh=False
            )

        df_results = screener.screen(
            tickers=tickers,
            interval=interval,
            batch_size=int(batch_size),
            min_price=min_price,
        )

        if df_results.empty:
            raise ValueError(
                f"Screening produced no valid records for universe '{u_clean}'."
            )

        stocks_df = df_results[~df_results["is_index"]].copy()
        idx_df = df_results[df_results["is_index"]].copy()

        top_bull_df = stocks_df.sort_values(by="cum2d_ret_50", ascending=False).head(top_n)
        top_bear_df = stocks_df.sort_values(by="cum2d_ret_50", ascending=True).head(top_n)
        top_vol_df = stocks_df.sort_values(by="volatility_spread", ascending=False).head(top_n)

        benchmark_order = [
            "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^FTSE",
            "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^KS11", "^KS200", "^VIX",
        ]
        if not idx_df.empty:
            idx_df = idx_df.assign(
                _order=idx_df["ticker"].apply(
                    lambda s: benchmark_order.index(s) if s in benchmark_order else 99
                )
            ).sort_values(by="_order").drop(columns=["_order"])

        def clean_records(sub_df: pd.DataFrame) -> List[Dict[str, Any]]:
            if sub_df.empty:
                return []
            recs = sub_df.to_dict(orient="records")
            cleaned_recs: List[Dict[str, Any]] = []
            for r in recs:
                c_r: Dict[str, Any] = {}
                for k, v in r.items():
                    if isinstance(v, (np.floating, float)):
                        c_r[k] = round(float(v), 6) if not np.isnan(v) else 0.0
                    elif isinstance(v, (np.integer, int)):
                        c_r[k] = int(v)
                    elif isinstance(v, (np.bool_, bool)):
                        c_r[k] = bool(v)
                    else:
                        c_r[k] = v
                cleaned_recs.append(c_r)
            return cleaned_recs

        bull_list = clean_records(top_bull_df)
        bear_list = clean_records(top_bear_df)
        vol_list = clean_records(top_vol_df)
        idx_list = clean_records(idx_df)

        milestones = [int(m) for m in screener.hp.get("milestones", [1, 2])]
        quantiles = [float(q) for q in screener.quantiles]
        model_type = (
            "market_query"
            if isinstance(screener.model, MarketQueryPatchTST)
            else "patchtst"
        )
        lookback = int(screener.lookback)

        # Build prediction payloads for each unique hit (preserving primary hit category tags)
        hit_map: Dict[str, Dict[str, Any]] = {}
        category_tags: Dict[str, List[str]] = {}

        def register_hits(rows: List[Dict[str, Any]], cat_label: str) -> None:
            for r in rows:
                sym = str(r["ticker"]).upper().strip()
                category_tags.setdefault(sym, [])
                if cat_label not in category_tags[sym]:
                    category_tags[sym].append(cat_label)
                if sym not in hit_map:
                    hit_map[sym] = r

        register_hits(bull_list, "bullish")
        register_hits(bear_list, "bearish")
        register_hits(vol_list, "volatility")
        register_hits(idx_list, "index")

        if save_all_screened:
            register_hits(clean_records(stocks_df), "screened")

        hit_predictions: List[Dict[str, Any]] = []
        for sym, r in hit_map.items():
            cat_str = ",".join(category_tags.get(sym, ["screen"]))
            hit_predictions.append(
                self._screen_row_to_prediction_payload(
                    row=r,
                    milestones=milestones,
                    quantiles=quantiles,
                    screen_category=cat_str,
                    model_type=model_type,
                    lookback=lookback,
                )
            )

        duration = round(time.time() - t0, 3)
        results_payload = {
            "top_bullish": bull_list,
            "top_bearish": bear_list,
            "top_volatility": vol_list,
            "indexes": idx_list,
            "milestones": milestones,
            "quantiles": quantiles,
        }

        saved_run = self.db.save_screening_run(
            universe=u_clean,
            top_n=top_n,
            min_price=min_price,
            interval=interval,
            total_candidates=len(tickers),
            total_screened=len(df_results),
            checkpoint_path=str(used_path),
            duration_sec=duration,
            results_payload=results_payload,
            hit_predictions=hit_predictions,
            run_id=used_run_id,
        )
        return saved_run
