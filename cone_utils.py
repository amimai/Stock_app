"""Probability cone construction and realized price evaluation utilities.

Formats raw PatchTST quantile forecasts into structured probability cones and
enriches stored predictions with post-prediction actual market trajectories.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def to_yyyy_mm_dd(val: Any) -> str:
    """Standardizes any datetime, timestamp, or string to strict YYYY-MM-DD format."""
    if val is None or val == "":
        return ""
    try:
        ts = pd.to_datetime(val)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        s = str(val).strip()
        return s[:10]


def resolve_quantile_indices(quantiles: List[float]) -> Dict[str, Optional[int]]:
    """Maps standard quantile keys (q05..q95) to their index in the quantiles list.
    Guards against index collisions when custom quantiles are passed.
    """
    if not quantiles:
        return {
            "q05": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
        }

    q_round = [round(float(q), 2) for q in quantiles]
    targets = {
        "q05": 0.05,
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
        "q95": 0.95,
    }

    result: Dict[str, Optional[int]] = {k: None for k in targets}
    used_indices: set[int] = set()

    # Pass 1: Exact matches
    for k, target in targets.items():
        if target in q_round:
            idx = q_round.index(target)
            if idx not in used_indices:
                result[k] = idx
                used_indices.add(idx)

    # Pass 2: Fallbacks for core quantiles (q10, q90, q50) avoiding collisions
    n = len(quantiles)
    if n == 1:
        if result["q50"] is None and 0 not in used_indices:
            result["q50"] = 0
            used_indices.add(0)
    elif n >= 2:
        if result["q10"] is None and 0 not in used_indices:
            result["q10"] = 0
            used_indices.add(0)
        if result["q90"] is None and (n - 1) not in used_indices:
            result["q90"] = n - 1
            used_indices.add(n - 1)
        if result["q50"] is None:
            mid_idx = n // 2
            if mid_idx not in used_indices:
                result["q50"] = mid_idx
                used_indices.add(mid_idx)

    return result


def _check_in_band(lower: Optional[float], act: Optional[float], upper: Optional[float]) -> Optional[bool]:
    """Returns True if lower <= act <= upper, False if outside, or None if any input is missing/invalid."""
    if lower is None or upper is None or act is None:
        return None
    try:
        if not (pd.notna(lower) and pd.notna(upper) and pd.notna(act)):
            return None
        l_f = float(lower)
        u_f = float(upper)
        a_f = float(act)
        if not (math.isfinite(l_f) and math.isfinite(u_f) and math.isfinite(a_f)):
            return None
        return bool(l_f <= a_f <= u_f)
    except (ValueError, TypeError):
        return None


def build_cone_and_summary(
    anchor_date: str,
    anchor_close: float,
    milestones: List[int],
    quantiles: List[float],
    pred_cum_returns: np.ndarray,
    sample_std: Optional[np.ndarray] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Builds the structured cone and summary dicts from cumulative log-return predictions.

    Args:
        anchor_date: YYYY-MM-DD date string of the last input bar.
        anchor_close: Closing price at anchor_date.
        milestones: List of horizon steps in trading days (e.g. [1, 2, 5, 20]).
        quantiles: List of quantile levels (e.g. [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).
        pred_cum_returns: Array of shape [len(milestones), len(quantiles)] with cumulative log returns.
        sample_std: Optional array of shape [len(milestones), len(quantiles)] with ensemble std.
    """
    anchor_date_clean = to_yyyy_mm_dd(anchor_date)
    q_idx = resolve_quantile_indices(quantiles)
    std_arr = sample_std if sample_std is not None else np.zeros_like(pred_cum_returns)

    milestone_entries: List[Dict[str, Any]] = []
    summary_by_horizon: Dict[str, Any] = {}

    for h_i, m in enumerate(milestones):
        rets = pred_cum_returns[h_i]
        stds = std_arr[h_i] if h_i < len(std_arr) else np.zeros_like(rets)

        q_ret_map: Dict[str, float] = {}
        q_price_map: Dict[str, float] = {}
        q_std_map: Dict[str, float] = {}

        for q_name, idx in q_idx.items():
            if idx is not None and 0 <= idx < len(rets):
                val_ret = float(rets[idx]) if (pd.notna(rets[idx]) and math.isfinite(float(rets[idx]))) else 0.0
                q_ret_map[q_name] = round(val_ret, 6)
                val_std = float(stds[idx]) if (idx < len(stds) and pd.notna(stds[idx]) and math.isfinite(float(stds[idx]))) else 0.0
                q_std_map[q_name] = round(val_std, 6)

        # Ensure median q50 exists
        if "q50" not in q_ret_map:
            if q_ret_map:
                q_ret_map["q50"] = round(float(np.median(list(q_ret_map.values()))), 6)
            else:
                q_ret_map["q50"] = 0.0

        # Ensure q10 and q90 exist
        if "q10" not in q_ret_map:
            q_ret_map["q10"] = q_ret_map.get("q05", q_ret_map.get("q25", q_ret_map["q50"]))
        if "q90" not in q_ret_map:
            q_ret_map["q90"] = q_ret_map.get("q95", q_ret_map.get("q75", q_ret_map["q50"]))

        # Ensure monotonic order among core quantiles (q10 <= q50 <= q90)
        r10 = q_ret_map["q10"]
        r50 = q_ret_map["q50"]
        r90 = q_ret_map["q90"]
        r10, r50, r90 = sorted([r10, r50, r90])
        q_ret_map["q10"] = r10
        q_ret_map["q50"] = r50
        q_ret_map["q90"] = r90

        # Synthesize missing intermediate/tail quantiles ensuring proper spacing and non-negativity
        if "q25" not in q_ret_map:
            q_ret_map["q25"] = round(0.5 * (r10 + r50), 6)
        if "q75" not in q_ret_map:
            q_ret_map["q75"] = round(0.5 * (r50 + r90), 6)
        if "q05" not in q_ret_map:
            q_ret_map["q05"] = round(r10 - 0.25 * max(0.0, r50 - r10), 6)
        if "q95" not in q_ret_map:
            q_ret_map["q95"] = round(r90 + 0.25 * max(0.0, r90 - r50), 6)

        # Ensure all 7 quantiles are strictly non-decreasing (rearrangement / monotonic sorting)
        q_order = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
        vals = [q_ret_map[k] for k in q_order]
        if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            vals.sort()
            for k, v in zip(q_order, vals):
                q_ret_map[k] = round(v, 6)

        # Compute price map from sorted returns
        for k in q_order:
            q_price_map[k] = round(float(anchor_close * math.exp(q_ret_map[k])), 4)

        # Synthesize missing standard deviations if needed
        if "q25" not in q_std_map:
            q_std_map["q25"] = round(0.5 * (q_std_map.get("q10", 0.0) + q_std_map.get("q50", 0.0)), 6)
        if "q75" not in q_std_map:
            q_std_map["q75"] = round(0.5 * (q_std_map.get("q50", 0.0) + q_std_map.get("q90", 0.0)), 6)
        if "q05" not in q_std_map:
            q_std_map["q05"] = q_std_map.get("q10", 0.0)
        if "q95" not in q_std_map:
            q_std_map["q95"] = q_std_map.get("q90", 0.0)

        r10 = q_ret_map["q10"]
        r50 = q_ret_map["q50"]
        r90 = q_ret_map["q90"]
        spread_80 = max(0.0, round(r90 - r10, 6))
        spread_50 = max(0.0, round(q_ret_map["q75"] - q_ret_map["q25"], 6))
        spread_90 = max(0.0, round(q_ret_map["q95"] - q_ret_map["q05"], 6))
        down_diff = max(1e-6, r50 - r10)
        up_diff = max(1e-6, r90 - r50)
        skew = round(up_diff / down_diff, 4)

        entry = {
            "horizon_days": int(m),
            "returns": q_ret_map,
            "prices": q_price_map,
            "stds": q_std_map,
            "spread_80": spread_80,
            "spread_50": spread_50,
            "spread_90": spread_90,
            "skew": skew,
        }
        milestone_entries.append(entry)
        summary_by_horizon[f"{m}d"] = {
            "ret_50": r50,
            "ret_10": r10,
            "ret_90": r90,
            "ret_25": q_ret_map["q25"],
            "ret_75": q_ret_map["q75"],
            "ret_05": q_ret_map["q05"],
            "ret_95": q_ret_map["q95"],
            "price_50": q_price_map["q50"],
            "price_10": q_price_map["q10"],
            "price_90": q_price_map["q90"],
            "price_25": q_price_map["q25"],
            "price_75": q_price_map["q75"],
            "price_05": q_price_map["q05"],
            "price_95": q_price_map["q95"],
            "spread_80": spread_80,
            "skew": skew,
        }

    cone = {
        "anchor_date": anchor_date_clean,
        "anchor_close": round(float(anchor_close), 4),
        "milestones": milestone_entries,
    }

    d1 = summary_by_horizon.get("1d")
    d2 = summary_by_horizon.get("2d")
    d5 = summary_by_horizon.get("5d")
    d20 = summary_by_horizon.get("20d")

    summary = {
        "day1_ret_50": d1.get("ret_50", 0.0) if d1 else None,
        "day1_price_50": d1.get("price_50", anchor_close) if d1 else None,
        "cum2d_ret_10": d2.get("ret_10", 0.0) if d2 else None,
        "cum2d_ret_25": d2.get("ret_25", 0.0) if d2 else None,
        "cum2d_ret_50": d2.get("ret_50", 0.0) if d2 else None,
        "cum2d_ret_75": d2.get("ret_75", 0.0) if d2 else None,
        "cum2d_ret_90": d2.get("ret_90", 0.0) if d2 else None,
        "price_2d_10": d2.get("price_10", anchor_close) if d2 else None,
        "price_2d_25": d2.get("price_25", anchor_close) if d2 else None,
        "price_2d_50": d2.get("price_50", anchor_close) if d2 else None,
        "price_2d_75": d2.get("price_75", anchor_close) if d2 else None,
        "price_2d_90": d2.get("price_90", anchor_close) if d2 else None,
        "cum5d_ret_50": d5.get("ret_50") if d5 else None,
        "price_5d_50": d5.get("price_50") if d5 else None,
        "cum20d_ret_50": d20.get("ret_50") if d20 else None,
        "price_20d_50": d20.get("price_50") if d20 else None,
        "volatility_spread_2d": d2.get("spread_80", 0.0) if d2 else None,
        "skew_2d": d2.get("skew", 1.0) if d2 else None,
        "horizons": summary_by_horizon,
    }
    return cone, summary


def enrich_prediction_with_realized(
    pred: Dict[str, Any],
    df: pd.DataFrame,
    context_before: int = 35,
    context_after_extra: int = 10,
) -> Dict[str, Any]:
    """Enriches a stored prediction with daily interpolated cone points and realized price performance."""
    enriched = dict(pred)
    cone = dict(pred.get("cone", {}))
    milestone_list = cone.get("milestones", [])
    anchor_date_str = to_yyyy_mm_dd(pred.get("anchor_date"))
    raw_anchor_close = pred.get("anchor_close", cone.get("anchor_close", 0.0))
    anchor_close = float(raw_anchor_close) if raw_anchor_close is not None else 0.0

    if df is None or df.empty or not milestone_list:
        enriched["realized"] = {"has_data": False, "elapsed_trading_days": 0}
        return enriched

    close_col = "Close" if "Close" in df.columns else df.columns[0]
    df = df.dropna(subset=[close_col])
    if df.empty:
        enriched["realized"] = {"has_data": False, "elapsed_trading_days": 0}
        return enriched

    # Map DataFrame dates to YYYY-MM-DD strings
    date_strs = [to_yyyy_mm_dd(ts) for ts in df.index]
    open_col = "Open" if "Open" in df.columns else close_col
    high_col = "High" if "High" in df.columns else close_col
    low_col = "Low" if "Low" in df.columns else close_col

    # Locate anchor index (exact match or latest bar <= anchor_date)
    anchor_idx: Optional[int] = None
    for idx, d_str in enumerate(date_strs):
        if d_str == anchor_date_str:
            anchor_idx = idx
            break
        if d_str < anchor_date_str:
            anchor_idx = idx
        else:
            break

    if anchor_idx is None:
        enriched["realized"] = {"has_data": False, "elapsed_trading_days": 0}
        return enriched

    total_bars = len(df)
    elapsed_days = max(0, (total_bars - 1) - anchor_idx)
    max_horizon = max(int(m["horizon_days"]) for m in milestone_list)

    # Generate projected business days beyond the last date in df if needed
    last_df_date = pd.to_datetime(date_strs[-1])
    needed_future_days = max(0, (anchor_idx + max_horizon) - (total_bars - 1))
    future_bdates: List[str] = []
    if needed_future_days > 0:
        bdr = pd.bdate_range(start=last_df_date + pd.Timedelta(days=1), periods=needed_future_days + 5)
        future_bdates = [to_yyyy_mm_dd(d) for d in bdr]

    def get_trading_date_for_step(step: int) -> tuple[str, bool, Optional[int]]:
        target_idx = anchor_idx + step
        if target_idx < total_bars:
            return date_strs[target_idx], True, target_idx
        future_offset = target_idx - total_bars
        if future_offset < len(future_bdates):
            return future_bdates[future_offset], False, None
        fallback_d = last_df_date + pd.Timedelta(days=int((future_offset + 1) * 1.45))
        return to_yyyy_mm_dd(fallback_d), False, None

    # Build keyframe nodes for interpolation: day 0 -> each milestone
    q_keys = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    keyframes_days = [0] + [int(m["horizon_days"]) for m in milestone_list]
    keyframes_rets: Dict[str, List[float]] = {k: [0.0] for k in q_keys}
    milestone_day_set = set(int(m["horizon_days"]) for m in milestone_list)

    for m in milestone_list:
        rets_m = m.get("returns", {})
        if not isinstance(rets_m, dict):
            rets_m = {}

        q50_val = rets_m.get("q50")
        fallback_val = 0.0
        if q50_val is not None:
            try:
                f_q50 = float(q50_val)
                if math.isfinite(f_q50):
                    fallback_val = f_q50
            except (ValueError, TypeError):
                fallback_val = 0.0

        for k in q_keys:
            val = rets_m.get(k)
            ret_f = fallback_val
            if val is not None:
                try:
                    f_val = float(val)
                    if math.isfinite(f_val):
                        ret_f = f_val
                except (ValueError, TypeError):
                    ret_f = fallback_val
            keyframes_rets[k].append(ret_f)

    # Interpolate daily cone steps 0..max_horizon
    daily_cone: List[Dict[str, Any]] = []
    for step in range(0, max_horizon + 1):
        d_str, is_realized, df_idx = get_trading_date_for_step(step)
        step_rets: Dict[str, float] = {}
        step_prices: Dict[str, float] = {}
        for k in q_keys:
            interp_r = float(np.interp(step, keyframes_days, keyframes_rets[k]))
            step_rets[k] = round(interp_r, 6)
            step_prices[k] = round(float(anchor_close * np.exp(interp_r)), 4)

        actual_close = None
        actual_high = None
        actual_low = None
        actual_open = None
        actual_log_ret = None
        if is_realized and df_idx is not None:
            c_raw = df[close_col].iloc[df_idx]
            actual_close = round(float(c_raw), 4)

            o_raw = df[open_col].iloc[df_idx]
            o_val = float(o_raw) if (pd.notna(o_raw) and math.isfinite(float(o_raw))) else actual_close
            actual_open = round(float(o_val), 4)

            h_raw = df[high_col].iloc[df_idx]
            h_val = float(h_raw) if (pd.notna(h_raw) and math.isfinite(float(h_raw))) else max(o_val, actual_close)
            actual_high = round(float(h_val), 4)

            l_raw = df[low_col].iloc[df_idx]
            l_val = float(l_raw) if (pd.notna(l_raw) and math.isfinite(float(l_raw))) else min(o_val, actual_close)
            actual_low = round(float(l_val), 4)

            if anchor_close > 0 and actual_close > 0:
                actual_log_ret = round(float(np.log(actual_close / anchor_close)), 6)

        daily_cone.append(
            {
                "step": step,
                "date": d_str,
                "is_milestone": step in milestone_day_set,
                "is_realized": is_realized,
                "prices": step_prices,
                "returns": step_rets,
                "actual_close": actual_close,
                "actual_open": actual_open,
                "actual_high": actual_high,
                "actual_low": actual_low,
                "actual_return": actual_log_ret,
            }
        )

    # Evaluate milestone outcomes
    milestone_outcomes: List[Dict[str, Any]] = []
    for m in milestone_list:
        h_days = int(m.get("horizon_days", 0))
        d_str, is_realized, df_idx = get_trading_date_for_step(h_days)
        p_map = m.get("prices") or {}
        r_map = m.get("returns") or {}
        outcome: Dict[str, Any] = {
            **m,
            "target_date": d_str,
            "is_realized": is_realized,
            "actual_close": None,
            "actual_return": None,
            "error_pct": None,
            "in_50_band": None,
            "in_80_band": None,
            "in_90_band": None,
            "direction_correct": None,
        }
        if is_realized and df_idx is not None:
            c_raw = df[close_col].iloc[df_idx]
            act_c: Optional[float] = None
            if pd.notna(c_raw):
                try:
                    f_c = float(c_raw)
                    if math.isfinite(f_c):
                        act_c = f_c
                except (ValueError, TypeError):
                    act_c = None

            act_r: Optional[float] = None
            if act_c is not None and anchor_close > 0 and act_c > 0:
                act_r = float(np.log(act_c / anchor_close))

            p_q50 = p_map.get("q50")
            pred_c50 = float(p_q50) if (p_q50 is not None and pd.notna(p_q50)) else float(anchor_close)
            r_q50 = r_map.get("q50")
            pred_r50 = float(r_q50) if (r_q50 is not None and pd.notna(r_q50)) else 0.0

            error_pct = None
            if act_c is not None and anchor_close > 0:
                error_pct = round((act_c - pred_c50) / anchor_close, 6)

            direction_correct = None
            if act_r is not None and pred_r50 is not None:
                direction_correct = bool((act_r >= 0 and pred_r50 >= 0) or (act_r < 0 and pred_r50 < 0))

            outcome.update(
                {
                    "actual_close": round(act_c, 4) if act_c is not None else None,
                    "actual_return": round(act_r, 6) if act_r is not None else None,
                    "error_pct": error_pct,
                    "in_50_band": _check_in_band(p_map.get("q25"), act_c, p_map.get("q75")),
                    "in_80_band": _check_in_band(p_map.get("q10"), act_c, p_map.get("q90")),
                    "in_90_band": _check_in_band(p_map.get("q05"), act_c, p_map.get("q95")),
                    "direction_correct": direction_correct,
                }
            )
        milestone_outcomes.append(outcome)

    # Context bars around the prediction anchor date for focused inspection
    start_ctx = max(0, anchor_idx - context_before)
    end_ctx = min(total_bars, anchor_idx + max_horizon + context_after_extra + 1)
    context_bars: List[Dict[str, Any]] = []
    for i in range(start_ctx, end_ctx):
        c_raw = df[close_col].iloc[i]
        c_val = float(c_raw)

        o_raw = df[open_col].iloc[i]
        o_val = float(o_raw) if (pd.notna(o_raw) and math.isfinite(float(o_raw))) else c_val

        h_raw = df[high_col].iloc[i]
        h_val = float(h_raw) if (pd.notna(h_raw) and math.isfinite(float(h_raw))) else max(o_val, c_val)

        l_raw = df[low_col].iloc[i]
        l_val = float(l_raw) if (pd.notna(l_raw) and math.isfinite(float(l_raw))) else min(o_val, c_val)

        context_bars.append(
            {
                "date": date_strs[i],
                "open": round(float(o_val), 4),
                "high": round(float(h_val), 4),
                "low": round(float(l_val), 4),
                "close": round(float(c_val), 4),
                "rel_step": int(i - anchor_idx),
            }
        )

    c_last = df[close_col].iloc[-1]
    latest_close = float(c_last) if (pd.notna(c_last) and math.isfinite(float(c_last))) else float(anchor_close)
    latest_ret = float(np.log(latest_close / anchor_close)) if anchor_close > 0 and latest_close > 0 else 0.0

    cone["milestones"] = milestone_outcomes
    cone["daily_cone"] = daily_cone
    enriched["cone"] = cone
    enriched["context_bars"] = context_bars
    enriched["realized"] = {
        "has_data": elapsed_days > 0,
        "elapsed_trading_days": int(elapsed_days),
        "max_horizon_days": int(max_horizon),
        "latest_date": date_strs[-1],
        "latest_close": round(latest_close, 4),
        "latest_return_since_anchor": round(latest_ret, 6),
    }
    return enriched
