"""Independent completed-5m ORB and MEX setup detectors.

Both detectors return setup intents only. SignalEngine performs fresh L1
quote confirmation, extension/risk validation, durable deduplication and
Telegram delivery. No forming candle or order API is used here.
"""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import numpy as np
import pandas as pd
from .risk import LIVE_TARGET_R_MULTIPLE
from .indicators import adx, atr, ema, rsi, vwap
from .candle_quality import directional_candle_quality

def _reject(diagnostics, stage: str, reason: str, values=None):
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"stage": stage, "reason": reason})
        if values:
            clean = {}
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    number = float(value)
                    clean[key] = round(number, 6) if math.isfinite(number) else None
                else:
                    clean[key] = value
            diagnostics["values"] = clean
    return None


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _frame(frame, diagnostics=None):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns) or len(frame) < 60:
        return _reject(diagnostics, "history", "insufficient_or_missing_ohlcv", {"bars": len(frame) if isinstance(frame, pd.DataFrame) else 0})
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    if f[["timestamp", *numeric_columns]].isna().any().any():
        return _reject(diagnostics, "data_validation", "invalid_timestamp_or_ohlcv")
    if not np.isfinite(f[numeric_columns].to_numpy(dtype=float)).all():
        return _reject(diagnostics, "data_validation", "non_finite_ohlcv")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f.timestamp.duplicated().any() or (f.volume < 0).any():
        return _reject(diagnostics, "data_validation", "duplicate_timestamps_or_negative_volume")
    if (f.high < f[["open", "close", "low"]].max(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    if (f.low > f[["open", "close", "high"]].min(axis=1)).any():
        return _reject(diagnostics, "data_validation", "invalid_ohlc_structure")
    return f


def _indicators(f):
    f = f.copy()
    f["ema9"] = ema(f.close, 9)
    f["ema21"] = ema(f.close, 21)
    f["ema20"] = ema(f.close, 20)
    f["ema50"] = ema(f.close, 50)
    f["rsi"] = rsi(f.close, 14)
    f["atr"] = atr(f, 14)
    f["adx"], f["plus_di"], f["minus_di"] = adx(f, 14)
    f["vwap"] = np.nan
    session_date = f.timestamp.iloc[-1].date()
    mask = f.timestamp.dt.date == session_date
    f.loc[mask, "vwap"] = vwap(f.loc[mask])
    f["volume_base"] = f.volume.shift(1).rolling(20, min_periods=15).mean()
    f["rvol"] = f.volume / f.volume_base.replace(0, np.nan)
    f["atr_base"] = f.atr.shift(1).rolling(20, min_periods=15).mean()
    f["atr_expansion"] = f.atr / f.atr_base.replace(0, np.nan)
    f["roc3"] = f.close.pct_change(3) * 100.0
    fast = f.close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = f.close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_hist"] = macd_line - signal
    f["bb_mid"] = f.close.rolling(20, min_periods=20).mean()
    f["bb_std"] = f.close.rolling(20, min_periods=20).std(ddof=0)
    f["bb_width"] = (4 * f.bb_std) / f.bb_mid.replace(0, np.nan)
    return f


def _risk(f, side, reference, atr_value):
    recent = f.iloc[-5:]
    buffer = 0.15 * atr_value
    stop = float(recent.low.min()) - buffer if side == "LONG" else float(recent.high.max()) + buffer
    risk = (reference - stop) if side == "LONG" else (stop - reference)
    risk_atr = risk / atr_value if atr_value > 0 else float("inf")
    if not math.isfinite(risk) or risk <= 0 or risk_atr > 3.0:
        return None
    target = reference + LIVE_TARGET_R_MULTIPLE * risk if side == "LONG" else reference - LIVE_TARGET_R_MULTIPLE * risk
    return {"stop_loss": stop, "take_profit_1": target, "risk_distance_atr": risk_atr}


def _base_result(f, side, strategy, reason, trigger_price, reference, atr_value, extra, diagnostics=None):
    risk = _risk(f, side, reference, atr_value)
    if risk is None:
        return _reject(diagnostics, "risk", "structural_risk_out_of_bounds", {"atr": atr_value, "reference": reference})
    candle = f.iloc[-1]
    return {
        "strategy": strategy, "side": side,
        "entry_reference": float(reference), "trigger_price": float(trigger_price),
        "stop_loss": risk["stop_loss"], "take_profit_1": risk["take_profit_1"],
        "atr": float(atr_value), "risk_distance_atr": round(risk["risk_distance_atr"], 4),
        "reason": reason, "candle_time": candle.timestamp,
        "indicators": extra,
    }


def evaluate_mex(frame, *, adx_min=None, min_rvol=1.50,
                 min_atr_expansion=1.15, diagnostics=None):
    """Detect one genuine 5m momentum explosion; no squeeze or continuation paths.

    Hard gates focus on participation, volatility expansion, directional
    displacement and candle quality. EMA/DI/RSI/VWAP are a small context score,
    not a unanimous checklist, so one lagging indicator cannot veto an impulse.
    """
    f = _frame(frame, diagnostics)
    if f is None:
        return None
    f = _indicators(f)
    c, p = f.iloc[-1], f.iloc[-2]
    prior_atr = _finite(p.atr)
    vals = [c.atr, c.ema9, c.ema21, c.rsi, c.plus_di, c.minus_di,
            c.rvol, c.atr_expansion, c.roc3, prior_atr]
    if any(_finite(x) is None for x in vals) or float(c.atr) <= 0 or prior_atr <= 0:
        return _reject(
            diagnostics, "indicators", "indicator_warmup_or_invalid",
            {"atr": c.atr, "prior_atr": prior_atr, "rvol": c.rvol,
             "atr_expansion": c.atr_expansion},
        )

    if float(c.rvol) < min_rvol or float(c.atr_expansion) < min_atr_expansion:
        failed = [
            name for name, passed in (
                ("rvol", float(c.rvol) >= min_rvol),
                ("atr_expansion", float(c.atr_expansion) >= min_atr_expansion),
            ) if not passed
        ]
        return _reject(
            diagnostics, "participation_and_expansion", "volume_or_atr_expansion_below_minimum",
            {"failed_filters": failed, "rvol": c.rvol, "min_rvol": min_rvol,
             "atr_expansion": c.atr_expansion, "min_atr_expansion": min_atr_expansion},
        )

    # Direction must be present in both the impulse candle and its recent
    # three-candle displacement. This avoids classifying a large wick as trend.
    if c.close > c.open and c.close > p.close and float(c.roc3) > 0:
        side = "LONG"
        sign = 1.0
    elif c.close < c.open and c.close < p.close and float(c.roc3) < 0:
        side = "SHORT"
        sign = -1.0
    else:
        return _reject(
            diagnostics, "direction", "no_aligned_directional_impulse",
            {"open": c.open, "close": c.close, "previous_close": p.close, "roc3": c.roc3},
        )

    # Normalize the current candle against ATR known before this candle.
    candle_range_atr = (float(c.high) - float(c.low)) / prior_atr
    candle_body_atr = abs(float(c.close) - float(c.open)) / prior_atr
    impulse_atr = sign * (float(c.close) - float(p.close)) / prior_atr
    roc_atr = sign * (float(c.close) - float(f.close.iloc[-4])) / prior_atr
    if (candle_range_atr < 0.90 or candle_body_atr < 0.45
            or impulse_atr < 0.55 or roc_atr < 0.65):
        return _reject(
            diagnostics, "impulse_strength", "candle_or_price_acceleration_too_weak",
            {"range_atr": candle_range_atr, "body_atr": candle_body_atr,
             "impulse_atr": impulse_atr, "roc3_atr": roc_atr},
        )

    # A real explosion must take out a recent local extreme, not just print a
    # high-volume candle in the middle of a range.
    prior = f.iloc[-9:-1]
    structure_break = (
        float(c.close) > float(prior.high.max()) if side == "LONG"
        else float(c.close) < float(prior.low.min())
    )
    if not structure_break:
        return _reject(
            diagnostics, "structure", "recent_structure_not_broken",
            {"side": side, "close": c.close,
             "prior_8_high": float(prior.high.max()),
             "prior_8_low": float(prior.low.min())},
        )

    quality = directional_candle_quality(
        c.open, c.high, c.low, c.close, side,
        min_body_ratio=0.30, min_close_location=0.70,
        max_opposing_wick_ratio=0.30,
    )
    if not quality["passed"]:
        return _reject(diagnostics, "candle_quality", "directional_candle_quality_failed", {"quality": quality})

    # Context is scored, not stacked as four mandatory filters. At least two
    # of EMA alignment, DI pressure, RSI momentum and VWAP position must agree.
    context_checks = {
        "ema_alignment": bool(c.ema9 > c.ema21) if side == "LONG" else bool(c.ema9 < c.ema21),
        "di_pressure": bool(c.plus_di > c.minus_di) if side == "LONG" else bool(c.minus_di > c.plus_di),
        "rsi_momentum": bool(c.rsi >= 50) if side == "LONG" else bool(c.rsi <= 50),
        "vwap_position": bool(c.close >= c.vwap) if side == "LONG" else bool(c.close <= c.vwap),
    }
    context_score = sum(context_checks.values())
    if context_score < 2:
        return _reject(
            diagnostics, "directional_context", "insufficient_directional_context",
            {"context_score": context_score, "required": 2, "checks": context_checks},
        )

    extension = abs(float(c.close) - float(c.ema9)) / float(c.atr)
    if extension > 1.50:
        return _reject(
            diagnostics, "entry_quality", "ema_extension_too_large",
            {"extension_atr": extension, "max_extension_atr": 1.50},
        )

    reference = float(c.close)
    trigger = float(c.high) if side == "LONG" else float(c.low)
    reason = (
        f"MEX MOMENTUM_EXPLOSION; RVOL={float(c.rvol):.2f}, "
        f"ATR expansion={float(c.atr_expansion):.2f}, "
        f"impulse={impulse_atr:.2f} ATR, context={context_score}/4"
    )
    return _base_result(
        f, side, "MEX", reason, trigger, reference, float(c.atr),
        {
            "mex_path": "MOMENTUM_EXPLOSION",
            "rvol": round(float(c.rvol), 4),
            "atr_expansion": round(float(c.atr_expansion), 4),
            "rsi": round(float(c.rsi), 2),
            "roc3": round(float(c.roc3), 4),
            "range_atr": round(candle_range_atr, 4),
            "body_atr": round(candle_body_atr, 4),
            "impulse_atr": round(impulse_atr, 4),
            "roc3_atr": round(roc_atr, 4),
            "structure_break": True,
            "context_score": context_score,
            "context_checks": context_checks,
            "extension_atr": round(extension, 4),
            "candle_quality": quality,
        }, diagnostics,
    )

