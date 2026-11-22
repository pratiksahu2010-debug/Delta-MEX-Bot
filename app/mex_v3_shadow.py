"""MEX V3 adaptive squeeze detector, initially shadow-only.

This module deliberately does not create live setups or alerts. It evaluates
completed candles and returns a directional candidate plus detailed evidence
for walk-forward measurement alongside the production MEX V2 detector.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .candle_quality import directional_candle_quality
from .indicators import adx, atr, vwap
from .momentum import _finite, _frame, _linear_regression_endpoint

BB_LENGTH = 20
BB_STDDEV = 2.0
KC_LENGTH = 20
KC_ATR_MULTIPLE = 1.5
ADX_LENGTH = 14
DEFAULT_MIN_BARS = 8
DEFAULT_DEPTH_PCT = 0.20
DEFAULT_DEPTH_LOOKBACK = 200
DEFAULT_MIN_RVOL = 1.50


def _regression_slope(values):
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        return np.nan
    x = np.arange(len(array), dtype=float)
    slope, _ = np.polyfit(x, array, 1)
    return float(slope)


def _reject(diagnostics, stage, reason, values=None):
    if diagnostics is not None:
        diagnostics.clear()
        clean = {}
        for key, value in (values or {}).items():
            if isinstance(value, (int, float, np.number)):
                number = float(value)
                clean[key] = round(number, 8) if math.isfinite(number) else None
            else:
                clean[key] = value
        diagnostics.update({"stage": stage, "reason": reason, "values": clean})
    return None


def _episode_columns(on, high, low, depth_rank, depth_pct):
    """Track complete squeeze episodes, including their full high/low range."""
    size = len(on)
    run = np.zeros(size, dtype=int)
    episode_high = np.full(size, np.nan)
    episode_low = np.full(size, np.nan)
    deep_count = np.zeros(size, dtype=int)
    release_run = np.zeros(size, dtype=int)
    release_high = np.full(size, np.nan)
    release_low = np.full(size, np.nan)
    release_deep_count = np.zeros(size, dtype=int)

    current_run = 0
    current_high = -np.inf
    current_low = np.inf
    current_deep = 0
    for i in range(size):
        if bool(on[i]):
            if current_run == 0:
                current_high = float(high[i])
                current_low = float(low[i])
                current_deep = 0
            else:
                current_high = max(current_high, float(high[i]))
                current_low = min(current_low, float(low[i]))
            current_run += 1
            rank = _finite(depth_rank[i])
            if rank is not None and rank <= depth_pct:
                current_deep += 1
            run[i] = current_run
            episode_high[i] = current_high
            episode_low[i] = current_low
            deep_count[i] = current_deep
        else:
            if i > 0 and bool(on[i - 1]):
                release_run[i] = current_run
                release_high[i] = current_high
                release_low[i] = current_low
                release_deep_count[i] = current_deep
            current_run = 0
            current_high = -np.inf
            current_low = np.inf
            current_deep = 0

    return {
        "squeeze_run": run,
        "squeeze_episode_high": episode_high,
        "squeeze_episode_low": episode_low,
        "squeeze_deep_count": deep_count,
        "release_run": release_run,
        "release_range_high": release_high,
        "release_range_low": release_low,
        "release_deep_count": release_deep_count,
    }


def _indicators(frame, depth_lookback=DEFAULT_DEPTH_LOOKBACK, depth_pct=DEFAULT_DEPTH_PCT):
    f = frame.copy()
    f["atr20"] = atr(f, KC_LENGTH)
    f["adx"], f["plus_di"], f["minus_di"] = adx(f, ADX_LENGTH)

    # Preserve the production MEX BB-inside-KC boundary condition exactly:
    # BB is SMA-centred; KC remains EMA-centred as in MEX V2.
    f["bb_mid"] = f.close.rolling(BB_LENGTH, min_periods=BB_LENGTH).mean()
    f["bb_std"] = f.close.rolling(BB_LENGTH, min_periods=BB_LENGTH).std(ddof=0)
    f["bb_upper"] = f.bb_mid + BB_STDDEV * f.bb_std
    f["bb_lower"] = f.bb_mid - BB_STDDEV * f.bb_std
    f["bb_width"] = f.bb_upper - f.bb_lower

    f["kc_mid"] = f.close.ewm(
        span=KC_LENGTH, adjust=False, min_periods=KC_LENGTH
    ).mean()
    f["kc_upper"] = f.kc_mid + KC_ATR_MULTIPLE * f.atr20
    f["kc_lower"] = f.kc_mid - KC_ATR_MULTIPLE * f.atr20
    f["kc_width"] = f.kc_upper - f.kc_lower

    # Continuous width ratio is a depth measurement. Because the existing
    # BB and KC midlines differ (SMA vs EMA), ratio < 1 is NOT substituted for
    # the exact boundary containment test below.
    f["squeeze_ratio"] = (
        (BB_STDDEV * f.bb_std) /
        (KC_ATR_MULTIPLE * f.atr20).replace(0, np.nan)
    )
    f["squeeze_on"] = (
        (f.bb_upper < f.kc_upper) & (f.bb_lower > f.kc_lower)
    )
    f["squeeze_release"] = (
        f.squeeze_on.shift(1, fill_value=False) & ~f.squeeze_on
    )

    # Adaptive per-symbol percentile. The current bar is included in its own
    # rolling rank; the full 200-observation window is required.
    f["depth_rank"] = f.squeeze_ratio.rolling(
        depth_lookback, min_periods=depth_lookback
    ).rank(pct=True)

    highest = f.high.rolling(BB_LENGTH, min_periods=BB_LENGTH).max()
    lowest = f.low.rolling(BB_LENGTH, min_periods=BB_LENGTH).min()
    donchian_mid = (highest + lowest) / 2.0
    momentum_base = (donchian_mid + f.bb_mid) / 2.0
    momentum_raw = f.close - momentum_base
    f["ttm_momentum"] = momentum_raw.rolling(
        BB_LENGTH, min_periods=BB_LENGTH
    ).apply(_linear_regression_endpoint, raw=True)
    f["ttm_momentum_slope"] = momentum_raw.rolling(
        BB_LENGTH, min_periods=BB_LENGTH
    ).apply(_regression_slope, raw=True)

    f["volume_average"] = f.volume.shift(1).rolling(
        20, min_periods=20
    ).mean()
    f["rvol"] = f.volume / f.volume_average.replace(0, np.nan)

    f["vwap"] = np.nan
    f["vwap_session_volume"] = 0.0
    session_date = f.timestamp.iloc[-1].date()
    session_mask = f.timestamp.dt.date == session_date
    session = f.loc[session_mask]
    if not session.empty:
        f.loc[session_mask, "vwap"] = vwap(session)
        f.loc[session_mask, "vwap_session_volume"] = (
            session.volume.fillna(0).cumsum()
        )

    episodes = _episode_columns(
        f.squeeze_on.fillna(False).to_numpy(dtype=bool),
        f.high.to_numpy(dtype=float),
        f.low.to_numpy(dtype=float),
        f.depth_rank.to_numpy(dtype=float),
        depth_pct,
    )
    for key, values in episodes.items():
        f[key] = values
    return f


def evaluate_mex_v3_shadow(
    frame,
    *,
    adx_min=19.0,
    min_rvol=DEFAULT_MIN_RVOL,
    min_bars=DEFAULT_MIN_BARS,
    depth_pct=DEFAULT_DEPTH_PCT,
    depth_lookback=DEFAULT_DEPTH_LOOKBACK,
    min_deep_fraction=0.50,
    diagnostics=None,
):
    """Return a shadow-only MEX V3 candidate; never arms or sends a signal."""
    if min_bars < 2 or not 0 < depth_pct < 1 or depth_lookback < 50:
        return _reject(diagnostics, "configuration", "invalid_shadow_parameters")
    if not 0 < min_deep_fraction <= 1:
        return _reject(diagnostics, "configuration", "invalid_deep_fraction")

    f = _frame(frame, diagnostics)
    if f is None:
        return None
    if len(f) < max(220, depth_lookback + BB_LENGTH):
        return _reject(
            diagnostics, "history", "insufficient_depth_history",
            {"bars": len(f), "required_bars": max(220, depth_lookback + BB_LENGTH)},
        )

    latest_open = f.timestamp.iloc[-1].to_pydatetime()
    if datetime.now(timezone.utc) < latest_open + timedelta(minutes=5):
        return _reject(
            diagnostics, "candle", "latest_candle_not_closed",
            {"candle_time": latest_open.isoformat()},
        )

    x = _indicators(f, depth_lookback=depth_lookback, depth_pct=depth_pct)
    c = x.iloc[-1]
    required = {
        "atr20": c.atr20, "adx": c.adx, "plus_di": c.plus_di,
        "minus_di": c.minus_di, "ttm_momentum": c.ttm_momentum,
        "ttm_momentum_slope": c.ttm_momentum_slope, "rvol": c.rvol,
        "vwap": c.vwap, "depth_rank": c.depth_rank,
    }
    invalid = [key for key, value in required.items() if _finite(value) is None]
    if invalid:
        return _reject(
            diagnostics, "indicators", "shadow_indicator_warmup_or_invalid",
            {"invalid_indicators": invalid, "bars": len(x),
             "squeeze_ratio": c.squeeze_ratio, "depth_rank": c.depth_rank},
        )
    if float(c.atr20) <= 0:
        return _reject(
            diagnostics, "indicators", "shadow_atr_nonpositive",
            {"atr20": c.atr20},
        )

    if not bool(c.squeeze_release):
        return _reject(
            diagnostics, "squeeze", "shadow_squeeze_not_released",
            {"squeeze_on": bool(c.squeeze_on),
             "squeeze_previous": bool(x.squeeze_on.iloc[-2]),
             "squeeze_ratio": c.squeeze_ratio,
             "depth_rank": c.depth_rank,
             "squeeze_run": int(c.squeeze_run)},
        )

    run = int(c.release_run)
    range_high = _finite(c.release_range_high)
    range_low = _finite(c.release_range_low)
    deep_count = int(c.release_deep_count)
    required_deep = int(math.ceil(run * min_deep_fraction))
    values = {
        "squeeze_ratio": c.squeeze_ratio,
        "depth_rank": c.depth_rank,
        "squeeze_duration_bars": run,
        "squeeze_duration_minutes": run * 5,
        "deep_bars": deep_count,
        "required_deep_bars": required_deep,
        "depth_pct": depth_pct,
        "depth_lookback": depth_lookback,
        "squeeze_range_high": range_high,
        "squeeze_range_low": range_low,
        "rvol": c.rvol,
        "min_rvol": min_rvol,
        "adx": c.adx,
        "adx_min": adx_min,
        "ttm_momentum": c.ttm_momentum,
        "ttm_momentum_slope": c.ttm_momentum_slope,
    }
    if run < min_bars:
        return _reject(diagnostics, "duration", "shadow_squeeze_duration_too_short", values)
    if deep_count < required_deep:
        return _reject(diagnostics, "depth", "shadow_squeeze_depth_insufficient", values)
    if float(c.rvol) < float(min_rvol):
        return _reject(diagnostics, "rvol", "shadow_rvol_below_minimum", values)
    if float(c.adx) < float(adx_min):
        return _reject(diagnostics, "adx", "shadow_adx_below_minimum", values)

    long_ok = (
        float(c.ttm_momentum) > 0
        and float(c.ttm_momentum_slope) > 0
        and float(c.plus_di) > float(c.minus_di)
        and float(c.close) > float(c.vwap)
        and range_high is not None
        and float(c.close) > range_high
    )
    short_ok = (
        float(c.ttm_momentum) < 0
        and float(c.ttm_momentum_slope) < 0
        and float(c.minus_di) > float(c.plus_di)
        and float(c.close) < float(c.vwap)
        and range_low is not None
        and float(c.close) < range_low
    )
    if not (long_ok or short_ok):
        values.update({
            "close": c.close, "vwap": c.vwap,
            "plus_di": c.plus_di, "minus_di": c.minus_di,
            "breakout_long": bool(range_high is not None and float(c.close) > range_high),
            "breakout_short": bool(range_low is not None and float(c.close) < range_low),
        })
        return _reject(diagnostics, "confirmation", "shadow_direction_or_breakout_not_confirmed", values)

    side = "LONG" if long_ok else "SHORT"
    quality = directional_candle_quality(
        c.open, c.high, c.low, c.close, side,
        min_body_ratio=0.30, min_close_location=0.70,
        max_opposing_wick_ratio=0.30,
    )
    values.update({
        "side": side, "close": c.close, "vwap": c.vwap,
        "plus_di": c.plus_di, "minus_di": c.minus_di,
        "candle_quality": quality,
    })
    if not quality["passed"]:
        return _reject(diagnostics, "candle_quality", "shadow_candle_quality_failed", values)

    # Match the live MEX structural stop policy for comparable forward outcomes.
    reference = float(c.close)
    atr_value = float(c.atr20)
    recent = x.iloc[-5:]
    stop = (
        float(recent.low.min()) - 0.15 * atr_value if side == "LONG"
        else float(recent.high.max()) + 0.15 * atr_value
    )
    risk = reference - stop if side == "LONG" else stop - reference
    risk_atr = risk / atr_value if atr_value > 0 else float("inf")
    if not math.isfinite(risk) or risk <= 0 or risk_atr > 3.0:
        values.update({"entry_reference": reference, "stop_loss": stop,
                       "risk_distance_atr": risk_atr})
        return _reject(diagnostics, "risk", "shadow_structural_risk_out_of_bounds", values)
    target = (
        reference + 2.0 * risk if side == "LONG"
        else reference - 2.0 * risk
    )
    trigger = float(c.high) if side == "LONG" else float(c.low)
    values.update({
        "entry_reference": reference,
        "trigger_price": trigger,
        "stop_loss": stop,
        "take_profit_1": target,
        "atr": atr_value,
        "risk_distance_atr": risk_atr,
        "entry_model": "release_candle_close_shadow_without_quote_confirmation",
    })

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "stage": "accepted",
            "reason": "mex_v3_shadow_candidate",
            "values": values,
        })
    return {
        "strategy": "MEX_V3_SHADOW", "side": side, "values": values,
        "entry_reference": reference, "trigger_price": trigger,
        "stop_loss": stop, "take_profit_1": target, "atr": atr_value,
    }
