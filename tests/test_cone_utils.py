"""Comprehensive automated tests for cone_utils.py.

Tests Issue 2 and Issue 8 fixes:
1. Safe handling of legacy 3-quantile predictions in enrich_prediction_with_realized without KeyError.
2. Safe handling of None/NaN quantile values without TypeError.
3. Quantile crossing / inversion rearrangement and non-negative spreads in build_cone_and_summary.
4. Proper synthesis of 7 quantiles from 3-quantile model outputs.
5. Index collision guarding in resolve_quantile_indices for custom quantiles.
6. Milestone indexing safety when milestones do not start with 1 or have arbitrary orders.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List
import unittest

# Ensure Stock_app root is in sys.path
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd

from cone_utils import (
    _check_in_band,
    build_cone_and_summary,
    enrich_prediction_with_realized,
    resolve_quantile_indices,
    to_yyyy_mm_dd,
)


class TestResolveQuantileIndices(unittest.TestCase):
    """Tests for resolve_quantile_indices and index collision protection."""

    def test_standard_7_quantiles(self) -> None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        q_idx = resolve_quantile_indices(quantiles)
        expected = {
            "q05": 0,
            "q10": 1,
            "q25": 2,
            "q50": 3,
            "q75": 4,
            "q90": 5,
            "q95": 6,
        }
        self.assertEqual(q_idx, expected)

    def test_standard_3_quantiles(self) -> None:
        quantiles = [0.10, 0.50, 0.90]
        q_idx = resolve_quantile_indices(quantiles)
        self.assertEqual(q_idx["q10"], 0)
        self.assertEqual(q_idx["q50"], 1)
        self.assertEqual(q_idx["q90"], 2)
        self.assertIsNone(q_idx["q05"])
        self.assertIsNone(q_idx["q25"])
        self.assertIsNone(q_idx["q75"])
        self.assertIsNone(q_idx["q95"])

    def test_guard_against_collisions_custom_quantiles(self) -> None:
        # Case 1: Custom quantiles [0.05, 0.50, 0.95]
        # In legacy code, q10 would fallback to 0 (colliding with q05) and q90 to 2 (colliding with q95).
        q_idx = resolve_quantile_indices([0.05, 0.50, 0.95])
        assigned_indices = [idx for idx in q_idx.values() if idx is not None]
        # Every assigned index must be distinct
        self.assertEqual(len(assigned_indices), len(set(assigned_indices)))
        self.assertEqual(q_idx["q05"], 0)
        self.assertEqual(q_idx["q50"], 1)
        self.assertEqual(q_idx["q95"], 2)
        self.assertIsNone(q_idx["q10"])
        self.assertIsNone(q_idx["q90"])

        # Case 2: Custom quantiles [0.25, 0.75]
        q_idx2 = resolve_quantile_indices([0.25, 0.75])
        assigned_indices2 = [idx for idx in q_idx2.values() if idx is not None]
        self.assertEqual(len(assigned_indices2), len(set(assigned_indices2)))
        self.assertEqual(q_idx2["q25"], 0)
        self.assertEqual(q_idx2["q75"], 1)

        # Case 3: Single quantile [0.50]
        q_idx3 = resolve_quantile_indices([0.50])
        assigned_indices3 = [idx for idx in q_idx3.values() if idx is not None]
        self.assertEqual(len(assigned_indices3), len(set(assigned_indices3)))
        self.assertEqual(q_idx3["q50"], 0)
        self.assertIsNone(q_idx3["q10"])
        self.assertIsNone(q_idx3["q90"])

        # Case 4: Empty quantiles
        q_idx_empty = resolve_quantile_indices([])
        self.assertTrue(all(v is None for v in q_idx_empty.values()))


class TestEnrichPredictionWithRealized(unittest.TestCase):
    """Tests for enrich_prediction_with_realized focusing on KeyError and TypeError fixes."""

    def setUp(self) -> None:
        self.anchor_date = "2026-01-05"
        self.anchor_close = 100.0
        # 10 business days starting from 2026-01-01
        dates = pd.bdate_range("2026-01-01", periods=10)
        # Anchor is at index 2 (2026-01-05)
        closes = [98.0, 99.0, 100.0, 101.5, 103.0, 104.5, 106.0, 107.0, 108.0, 109.0]
        self.df = pd.DataFrame(
            {
                "Open": [c - 0.5 for c in closes],
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
                "Close": closes,
                "Volume": [1000] * 10,
            },
            index=dates,
        )

    def test_legacy_3_quantile_prediction_no_keyerror(self) -> None:
        """Legacy 3-quantile predictions only have q10, q50, q90.

        Ensure no KeyError is raised when accessing missing q25, q75, q05, q95,
        and band flags evaluate safely (in_50_band is None, in_80_band is bool).
        """
        legacy_pred: Dict[str, Any] = {
            "anchor_date": self.anchor_date,
            "anchor_close": self.anchor_close,
            "cone": {
                "milestones": [
                    {
                        "horizon_days": 1,
                        "prices": {"q10": 98.0, "q50": 101.0, "q90": 104.0},
                        "returns": {"q10": -0.02, "q50": 0.01, "q90": 0.04},
                    },
                    {
                        "horizon_days": 2,
                        "prices": {"q10": 97.0, "q50": 102.0, "q90": 107.0},
                        "returns": {"q10": -0.03, "q50": 0.02, "q90": 0.07},
                    },
                ]
            },
        }

        enriched = enrich_prediction_with_realized(legacy_pred, self.df)
        self.assertTrue(enriched["realized"]["has_data"])
        self.assertGreater(enriched["realized"]["elapsed_trading_days"], 0)

        milestones = enriched["cone"]["milestones"]
        self.assertEqual(len(milestones), 2)

        # Day 1 milestone
        m1 = milestones[0]
        self.assertTrue(m1["is_realized"])
        self.assertEqual(m1["actual_close"], 101.5)
        # in_50_band and in_90_band must be None because q25/q75 and q05/q95 bounds are missing
        self.assertIsNone(m1["in_50_band"])
        self.assertIsNone(m1["in_90_band"])
        # in_80_band should be evaluated (98.0 <= 101.5 <= 104.0 -> True)
        self.assertIs(m1["in_80_band"], True)
        self.assertIs(m1["direction_correct"], True)

        # Day 2 milestone (actual_close is 103.0)
        m2 = milestones[1]
        self.assertTrue(m2["is_realized"])
        self.assertEqual(m2["actual_close"], 103.0)
        self.assertIsNone(m2["in_50_band"])
        self.assertIsNone(m2["in_90_band"])
        self.assertIs(m2["in_80_band"], True)

    def test_prediction_with_none_values_no_typeerror(self) -> None:
        """Explicit None values in returns or prices must not raise TypeError."""
        none_pred: Dict[str, Any] = {
            "anchor_date": self.anchor_date,
            "anchor_close": self.anchor_close,
            "cone": {
                "milestones": [
                    {
                        "horizon_days": 1,
                        "prices": {
                            "q05": None,
                            "q10": 98.0,
                            "q25": None,
                            "q50": None,
                            "q75": 105.0,
                            "q90": 107.0,
                            "q95": None,
                        },
                        "returns": {
                            "q05": None,
                            "q10": -0.02,
                            "q25": None,
                            "q50": None,
                            "q75": 0.05,
                            "q90": 0.07,
                            "q95": None,
                        },
                    }
                ]
            },
        }

        enriched = enrich_prediction_with_realized(none_pred, self.df)
        m1 = enriched["cone"]["milestones"][0]
        # in_50_band had q25=None -> must safely be None without TypeError '<='
        self.assertIsNone(m1["in_50_band"])
        # in_90_band had q05=None, q95=None -> must safely be None
        self.assertIsNone(m1["in_90_band"])
        # in_80_band has both bounds present (98.0 and 107.0) -> True
        self.assertIs(m1["in_80_band"], True)

        # Daily cone interpolation must succeed without TypeError float(None)
        daily_cone = enriched["cone"]["daily_cone"]
        self.assertGreater(len(daily_cone), 0)
        for pt in daily_cone:
            self.assertIn("prices", pt)
            self.assertIn("returns", pt)
            for k in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]:
                self.assertIsInstance(pt["returns"][k], float)
                self.assertIsInstance(pt["prices"][k], float)

    def test_enrich_with_none_or_nan_actual_close(self) -> None:
        """When actual close is non-finite or unrealized, band flags must safely be None."""
        df_inf = self.df.copy()
        # Set index 3 Close to infinity (non-finite)
        df_inf.loc[df_inf.index[3], "Close"] = float("inf")

        pred = {
            "anchor_date": self.anchor_date,
            "anchor_close": self.anchor_close,
            "cone": {
                "milestones": [
                    {
                        "horizon_days": 1,
                        "prices": {
                            "q05": 97.0,
                            "q10": 98.0,
                            "q25": 99.0,
                            "q50": 100.0,
                            "q75": 102.0,
                            "q90": 104.0,
                            "q95": 105.0,
                        },
                        "returns": {
                            "q05": -0.03,
                            "q10": -0.02,
                            "q25": -0.01,
                            "q50": 0.0,
                            "q75": 0.02,
                            "q90": 0.04,
                            "q95": 0.05,
                        },
                    },
                    {
                        "horizon_days": 50,  # Far future horizon, unrealized
                        "prices": {"q10": 95.0, "q50": 105.0, "q90": 115.0},
                        "returns": {"q10": -0.05, "q50": 0.05, "q90": 0.15},
                    },
                ]
            },
        }

        enriched = enrich_prediction_with_realized(pred, df_inf)
        m1 = enriched["cone"]["milestones"][0]
        # Step 1 was non-finite inf -> actual_close safely None, bands safely None
        self.assertIsNone(m1["actual_close"])
        self.assertIsNone(m1["in_50_band"])
        self.assertIsNone(m1["in_80_band"])
        self.assertIsNone(m1["in_90_band"])

        # Step 50 is unrealized -> actual_close None, bands None
        m2 = enriched["cone"]["milestones"][1]
        self.assertFalse(m2["is_realized"])
        self.assertIsNone(m2["actual_close"])
        self.assertIsNone(m2["in_50_band"])
        self.assertIsNone(m2["in_80_band"])
        self.assertIsNone(m2["in_90_band"])

    def test_check_in_band_helper(self) -> None:
        """Verify _check_in_band returns None when any bound or act is None/NaN."""
        self.assertIsNone(_check_in_band(None, 100.0, 110.0))
        self.assertIsNone(_check_in_band(90.0, None, 110.0))
        self.assertIsNone(_check_in_band(90.0, 100.0, None))
        self.assertIsNone(_check_in_band(None, None, None))
        self.assertIsNone(_check_in_band(float("nan"), 100.0, 110.0))
        self.assertIsNone(_check_in_band(90.0, float("nan"), 110.0))
        self.assertIsNone(_check_in_band(90.0, 100.0, float("nan")))
        self.assertIsNone(_check_in_band(float("inf"), 100.0, 110.0))
        self.assertTrue(_check_in_band(90.0, 100.0, 110.0))
        self.assertFalse(_check_in_band(90.0, 120.0, 110.0))


class TestBuildConeAndSummary(unittest.TestCase):
    """Tests for build_cone_and_summary: inversion handling, 3-quantile synthesis, and milestone indexing."""

    def setUp(self) -> None:
        self.anchor_date = "2026-01-05"
        self.anchor_close = 100.0

    def test_quantile_crossing_inversion_handled_7_quantiles(self) -> None:
        """Inverted/crossed returns must be sorted monotonically to ensure non-negative spreads."""
        milestones = [1, 2]
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        # Intentionally completely inverted returns (descending instead of ascending)
        pred_rets = np.array(
            [
                [0.08, 0.05, 0.02, 0.00, -0.01, -0.04, -0.07],
                [0.12, 0.08, 0.04, 0.01, -0.02, -0.06, -0.10],
            ],
            dtype=np.float64,
        )

        cone, summary = build_cone_and_summary(
            anchor_date=self.anchor_date,
            anchor_close=self.anchor_close,
            milestones=milestones,
            quantiles=quantiles,
            pred_cum_returns=pred_rets,
        )

        for entry in cone["milestones"]:
            r = entry["returns"]
            p = entry["prices"]
            # Verify strict non-decreasing order
            q_order = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
            for i in range(len(q_order) - 1):
                self.assertLessEqual(r[q_order[i]], r[q_order[i + 1]])
                self.assertLessEqual(p[q_order[i]], p[q_order[i + 1]])

            # Spreads must all be non-negative
            self.assertGreaterEqual(entry["spread_50"], 0.0)
            self.assertGreaterEqual(entry["spread_80"], 0.0)
            self.assertGreaterEqual(entry["spread_90"], 0.0)
            self.assertLessEqual(entry["spread_50"], entry["spread_80"])
            self.assertLessEqual(entry["spread_80"], entry["spread_90"])

        # Summary check
        self.assertGreaterEqual(summary["volatility_spread_2d"], 0.0)
        self.assertGreater(summary["skew_2d"], 0.0)

    def test_3_quantile_checkpoint_synthesis(self) -> None:
        """3-quantile predictions [q10, q50, q90] must synthesize proper non-negative 7-quantile bands."""
        milestones = [1, 2, 5, 20]
        quantiles = [0.10, 0.50, 0.90]
        # Standard monotonic 3-quantile returns
        pred_rets = np.array(
            [
                [-0.02, 0.005, 0.03],
                [-0.03, 0.010, 0.05],
                [-0.05, 0.020, 0.08],
                [-0.08, 0.030, 0.12],
            ],
            dtype=np.float64,
        )

        cone, summary = build_cone_and_summary(
            anchor_date=self.anchor_date,
            anchor_close=self.anchor_close,
            milestones=milestones,
            quantiles=quantiles,
            pred_cum_returns=pred_rets,
        )

        self.assertEqual(len(cone["milestones"]), 4)
        for entry in cone["milestones"]:
            r = entry["returns"]
            p = entry["prices"]
            q_order = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
            # All 7 quantiles must exist
            for q_k in q_order:
                self.assertIn(q_k, r)
                self.assertIn(q_k, p)

            # Strict monotonicity and proper spacing
            for i in range(len(q_order) - 1):
                self.assertLessEqual(r[q_order[i]], r[q_order[i + 1]])
                self.assertLessEqual(p[q_order[i]], p[q_order[i + 1]])

            self.assertGreaterEqual(entry["spread_50"], 0.0)
            self.assertGreaterEqual(entry["spread_80"], 0.0)
            self.assertGreaterEqual(entry["spread_90"], 0.0)
            self.assertLessEqual(entry["spread_50"], entry["spread_80"])
            self.assertLessEqual(entry["spread_80"], entry["spread_90"])

        # JSON serialization without NaN
        dumped_cone = json.dumps(cone, allow_nan=False)
        dumped_summary = json.dumps(summary, allow_nan=False)
        self.assertNotIn("NaN", dumped_cone)
        self.assertNotIn("NaN", dumped_summary)

    def test_milestone_indexing_safety(self) -> None:
        """When milestones do not start with 1, do not blindly assign milestone 0 to day 1."""
        # Case A: Milestones [5, 10, 20] (no day 1, no day 2)
        milestones = [5, 10, 20]
        quantiles = [0.10, 0.50, 0.90]
        pred_rets = np.array(
            [
                [-0.05, 0.02, 0.08],
                [-0.07, 0.03, 0.10],
                [-0.10, 0.05, 0.15],
            ],
            dtype=np.float64,
        )

        cone, summary = build_cone_and_summary(
            anchor_date=self.anchor_date,
            anchor_close=self.anchor_close,
            milestones=milestones,
            quantiles=quantiles,
            pred_cum_returns=pred_rets,
        )

        # day1 and cum2d must be None because they were not forecast
        self.assertIsNone(summary["day1_ret_50"])
        self.assertIsNone(summary["day1_price_50"])
        self.assertIsNone(summary["cum2d_ret_50"])
        self.assertIsNone(summary["price_2d_50"])
        self.assertIsNone(summary["volatility_spread_2d"])
        # cum5d and cum20d must be correctly populated
        self.assertEqual(summary["cum5d_ret_50"], 0.02)
        self.assertEqual(summary["cum20d_ret_50"], 0.05)

        # Case B: Milestones [2, 1, 5] (non-ascending order, 1 is not index 0)
        milestones_perm = [2, 1, 5]
        pred_rets_perm = np.array(
            [
                [-0.03, 0.015, 0.05],  # 2d return
                [-0.01, 0.008, 0.03],  # 1d return
                [-0.06, 0.025, 0.09],  # 5d return
            ],
            dtype=np.float64,
        )

        cone_b, summary_b = build_cone_and_summary(
            anchor_date=self.anchor_date,
            anchor_close=self.anchor_close,
            milestones=milestones_perm,
            quantiles=quantiles,
            pred_cum_returns=pred_rets_perm,
        )

        # day1_ret_50 must be from 1d (0.008), NOT from milestone 0 (0.015)
        self.assertAlmostEqual(summary_b["day1_ret_50"], 0.008, places=4)
        # cum2d_ret_50 must be from 2d (0.015)
        self.assertAlmostEqual(summary_b["cum2d_ret_50"], 0.015, places=4)


if __name__ == "__main__":
    unittest.main()
