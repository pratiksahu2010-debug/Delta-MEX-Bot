"""MEX V2: completed-candle TTM Squeeze momentum explosion detector.

The detector uses only four decision indicators: TTM Squeeze, RVOL,
ADX/+DI/-DI and UTC-session VWAP. ATR is retained only for structural risk
sizing. It returns a setup intent; SignalEngine handles live quote confirmation,
persistence, deduplication and Telegram delivery.
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .candle_quality import directional_candle_quality
from .indicators import adx, atr, vwap
from .risk import LIVE_TARGET_R_MULTIPLE


BB_LENGTH = 20
BB_STDDEV = 2.0
KC_LENGTH = 20
KC_ATR_MULTIPLE = 1.5
RVOL_LENGTH = 20
ADX_LENGTH = 14
ADX_MIN_DEFAULT = 19.0
BREAKOUT_LOOKBACK = 8


def _reject(diagnostics, stage: str, reason: str, values=None):
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"stage": stage, "reason": reason})
        if values:
            clean = {}
            for key, value in values.items():
                if isinstance(value, (int, float, np.number)):
                    number = float(value)
                    clean[key] = round(number, 6) if math.isfinite(number) else None
                else:
                    clean[key] = value
            diagnostics["values"] = clean
    return None


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _frame(frame, diagnostics=None):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns) or len(frame) < 60:
        return _reject(
            diagnostics, "history", "insufficient_or_missing_ohlcv",
            {"bars": len(frame) if isinstance(frame, pd.DataFrame) else 0},
        )
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        f[column] = pd.to_numeric(f[column], errors="coerce")
    columns = ["open", "high", "low", "close", "volume"]
    if f[["timestamp", *columns]].isna().any().any():
        return _reject(diagnostics, "data_validation", "invalid_timestamp_or_ohlcv")
    if not np.isfinite(f[columns].to_numpy(dtype=float)).all():
        return _reject(diagnostics, "data_validation", "non_finite_ohlcv")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f.timestamp.duplicated().any() or (f.volume < 0).any():
        return _reject(diagnostics, "data_validation", "duplicate_timestamps_or_negative_volume")
    if (f.high < f[["open", "close", "low"]].max(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    if (f.low > f[["open", "close", "high"]].min(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    gaps = f.timestamp.diff().dropna().dt.total_seconds()
    if not gaps.empty and (gaps != 300).any():
        return _reject(diagnostics, "data_validation", "non_contiguous_5m_candles")
    return f


def _linear_regression_endpoint(values):
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        return np.nan
    x = np.arange(len(array), dtype=float)
    slope, intercept = np.polyfit(x, array, 1)
    return float(intercept + slope * (len(array) - 1))


def _indicators(frame):
    f = frame.copy()
    f["atr"] = atr(f, 14)
    f["adx"], f["plus_di"], f["minus_di"] = adx(f, ADX_LENGTH)

    # Bollinger Bands: SMA(20) +/- 2 population standard deviations.
    f["bb_mid"] = f.close.rolling(BB_LENGTH, min_periods=BB_LENGTH).mean()
    f["bb_std"] = f.close.rolling(BB_LENGTH, min_periods=BB_LENGTH).std(ddof=0)
    f["bb_upper"] = f.bb_mid + BB_STDDEV * f.bb_std
    f["bb_lower"] = f.bb_mid - BB_STDDEV * f.bb_std

    # Keltner Channels: EMA(20) +/- 1.5 * ATR(20).
    f["kc_mid"] = f.close.ewm(span=KC_LENGTH, adjust=False, min_periods=KC_LENGTH).mean()
    f["kc_atr"] = atr(f, KC_LENGTH)
    f["kc_upper"] = f.kc_mid + KC_ATR_MULTIPLE * f.kc_atr
    f["kc_lower"] = f.kc_mid - KC_ATR_MULTIPLE * f.kc_atr
    f["squeeze_on"] = (f.bb_upper < f.kc_upper) & (f.bb_lower > f.kc_lower)
    f["squeeze_release"] = f.squeeze_on.shift(1, fill_value=False) & ~f.squeeze_on

    # TTM momentum histogram: linear-regression endpoint of close minus the
    # midpoint of the Donchian midpoint and 20-period SMA.
    highest = f.high.rolling(BB_LENGTH, min_periods=BB_LENGTH).max()
    lowest = f.low.rolling(BB_LENGTH, min_periods=BB_LENGTH).min()
    momentum_base = ((highest + lowest) / 2.0 + f.close.rolling(
        BB_LENGTH, min_periods=BB_LENGTH
    ).mean()) / 2.0
    momentum_raw = f.close - momentum_base
    f["ttm_momentum"] = momentum_raw.rolling(
        BB_LENGTH, min_periods=BB_LENGTH
    ).apply(_linear_regression_endpoint, raw=True)

    # RVOL compares the completed candle to the preceding 20 completed candles.
    f["volume_average"] = f.volume.shift(1).rolling(
        RVOL_LENGTH, min_periods=RVOL_LENGTH
    ).mean()
    f["rvol"] = f.volume / f.volume_average.replace(0, np.nan)

    # Crypto VWAP resets at 00:00 UTC. A full session needs up to 288 5m bars;
    # the caller requests 320 so the current UTC session is represented fully.
    f["vwap"] = np.nan
    session_date = f.timestamp.iloc[-1].date()
    session_mask = f.timestamp.dt.date == session_date
    session = f.loc[session_mask]
    if not session.empty:
        f.loc[session_mask, "vwap"] = vwap(session)
    f["vwap_session_bars"] = 0
    f.loc[session_mask, "vwap_session_bars"] = np.arange(1, len(session) + 1)
    return f


def _risk(frame, side, reference, atr_value):
    recent = frame.iloc[-5:]
    buffer = 0.15 * atr_value
    stop = (
        float(recent.low.min()) - buffer if side == "LONG"
        else float(recent.high.max()) + buffer
    )
    risk = reference - stop if side == "LONG" else stop - reference
    risk_atr = risk / atr_value if atr_value > 0 else float("inf")
    if not math.isfinite(risk) or risk <= 0 or risk_atr > 3.0:
        return None
    target = (
        reference + LIVE_TARGET_R_MULTIPLE * risk if side == "LONG"
        else reference - LIVE_TARGET_R_MULTIPLE * risk
    )
    return {"stop_loss": stop, "take_profit_1": target, "risk_distance_atr": risk_atr}


def evaluate_mex(frame, *, adx_min=ADX_MIN_DEFAULT, min_rvol=1.50,
                 min_atr_expansion=None, diagnostics=None):
    """Return a MEX V2 setup only on a completed TTM squeeze-release breakout.

    min_atr_expansion is accepted for backward compatibility but intentionally
    unused: ATR expansion is not a fifth signal filter in the agreed design.
    ATR is used only to normalize the structural stop.
    """
    f = _frame(frame, diagnostics)
    if f is None:
        return None
    # A forming live WebSocket candle must never arm a setup. The stream may
    # still invoke this detector, but only a fully closed 5m bar is actionable.
    latest_open = f.timestamp.iloc[-1].to_pydatetime()
    if datetime.now(timezone.utc) < latest_open + timedelta(minutes=5):
        return _reject(diagnostics, "candle", "latest_candle_not_closed",
                       {"candle_time": latest_open.isoformat()})
    f = _indicators(f)
    c, p = f.iloc[-1], f.iloc[-2]

    required_values = (
        c.atr, c.adx, c.plus_di, c.minus_di, c.rvol, c.vwap,
        c.ttm_momentum, p.ttm_momentum, c.vwap_session_bars,
    )
    if any(_finite(value) is None for value in required_values) or float(c.atr) <= 0:
        return _reject(
            diagnostics, "indicators", "indicator_warmup_or_invalid",
            {"atr": c.atr, "adx": c.adx, "rvol": c.rvol,
             "vwap": c.vwap, "ttm_momentum": c.ttm_momentum},
        )

    if not bool(c.squeeze_release):
        return _reject(
            diagnostics, "squeeze", "ttm_squeeze_not_released",
            {"squeeze_previous": bool(p.squeeze_on),
             "squeeze_current": bool(c.squeeze_on)},
        )

    if float(c.rvol) < float(min_rvol):
        return _reject(
            diagnostics, "rvol", "relative_volume_below_minimum",
            {"rvol": c.rvol, "min_rvol": min_rvol},
        )

    if float(c.adx) < float(adx_min):
        return _reject(
            diagnostics, "adx", "adx_below_minimum",
            {"adx": c.adx, "adx_min": adx_min},
        )

    momentum_rising = float(c.ttm_momentum) > float(p.ttm_momentum)
    momentum_falling = float(c.ttm_momentum) < float(p.ttm_momentum)
    if (float(c.ttm_momentum) > 0 and momentum_rising
            and float(c.plus_di) > float(c.minus_di)
            and float(c.close) > float(c.vwap)):
        side = "LONG"
    elif (float(c.ttm_momentum) < 0 and momentum_falling
            and float(c.minus_di) > float(c.plus_di)
            and float(c.close) < float(c.vwap)):
        side = "SHORT"
    else:
        return _reject(
            diagnostics, "direction", "four_indicator_direction_not_aligned",
            {"ttm_momentum": c.ttm_momentum, "momentum_previous": p.ttm_momentum,
             "plus_di": c.plus_di, "minus_di": c.minus_di,
             "close": c.close, "vwap": c.vwap},
        )

    prior = f.iloc[-(BREAKOUT_LOOKBACK + 1):-1]
    prior_high = float(prior.high.max())
    prior_low = float(prior.low.min())
    if side == "LONG" and float(c.close) <= prior_high:
        return _reject(
            diagnostics, "structure", "breakout_close_not_above_range",
            {"close": c.close, "range_high": prior_high},
        )
    if side == "SHORT" and float(c.close) >= prior_low:
        return _reject(
            diagnostics, "structure", "breakdown_close_not_below_range",
            {"close": c.close, "range_low": prior_low},
        )

    quality = directional_candle_quality(
        c.open, c.high, c.low, c.close, side,
        min_body_ratio=0.30, min_close_location=0.70,
        max_opposing_wick_ratio=0.30,
    )
    if not quality["passed"]:
        return _reject(
            diagnostics, "candle_quality", "directional_candle_quality_failed",
            {"quality": quality},
        )

    reference = float(c.close)
    trigger = float(c.high) if side == "LONG" else float(c.low)
    risk = _risk(f, side, reference, float(c.atr))
    if risk is None:
        return _reject(
            diagnostics, "risk", "structural_risk_out_of_bounds",
            {"atr": c.atr, "reference": reference},
        )

    indicators = {
        "mex_version": "MEX_V2_TTM_SQUEEZE",
        "ttm_squeeze_release": True,
        "ttm_momentum": round(float(c.ttm_momentum), 8),
        "ttm_momentum_previous": round(float(p.ttm_momentum), 8),
        "rvol": round(float(c.rvol), 4),
        "adx": round(float(c.adx), 2),
        "adx_min": float(adx_min),
        "plus_di": round(float(c.plus_di), 2),
        "minus_di": round(float(c.minus_di), 2),
        "vwap": round(float(c.vwap), 8),
        "vwap_session_bars": int(c.vwap_session_bars),
        "range_high": prior_high,
        "range_low": prior_low,
        "candle_quality": quality,
    }
    reason = (
        f"MEX V2 TTM SQUEEZE RELEASE; RVOL={float(c.rvol):.2f}, "
        f"ADX={float(c.adx):.1f}, DI={'+' if side == 'LONG' else '-'} "
        f"direction, price {'above' if side == 'LONG' else 'below'} VWAP"
    )
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "stage": "accepted", "reason": "mex_v2_setup_accepted",
            "values": indicators,
        })
    return {
        "strategy": "MEX", "side": side,
        "entry_reference": reference, "trigger_price": trigger,
        "stop_loss": risk["stop_loss"],
        "take_profit_1": risk["take_profit_1"],
        "atr": float(c.atr),
        "risk_distance_atr": round(risk["risk_distance_atr"], 4),
        "reason": reason, "candle_time": c.timestamp,
        "indicators": indicators,
    }
