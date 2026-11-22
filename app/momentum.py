"""Minimal MEX: strict 5-minute closed-candle momentum explosion signals.

No forming-candle evaluation, squeeze/ADX/VWAP/DI filters, or delayed quote
confirmation. The scanner passes a contiguous history of completed candles.
"""
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pandas as pd


REQUIRED_BARS = 250
TIMEFRAME_SECONDS = 300
ATR_LENGTH = 14
RVOL_LENGTH = 20
DONCHIAN_LENGTH = 20
EMA_LENGTH = 21
MIN_TR_RATIO = 2.0
MIN_RVOL = 2.0
MIN_CLOSE_LOCATION_LONG = 0.75
MAX_CLOSE_LOCATION_SHORT = 0.25
MAX_EXTENSION_ATR = 2.5
MIN_ATR_TO_ROUND_TRIP_FEE = 3.0
MIN_RR_TO_TP2 = 1.5
MIN_STOP_ATR = 0.8


def _reject(diagnostics, stage, reason, values=None):
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"stage": stage, "reason": reason, "values": values or {}})
    return None


def _wilder_atr(frame: pd.DataFrame, period: int = ATR_LENGTH) -> pd.Series:
    """Wilder ATR with an SMA seed followed by Wilder's recursive smoothing."""
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    values = np.full(len(frame), np.nan, dtype=float)
    if len(frame) < period:
        return pd.Series(values, index=frame.index, name="atr")
    raw = tr.to_numpy(dtype=float)
    seed = float(np.mean(raw[:period]))
    values[period - 1] = seed
    for index in range(period, len(raw)):
        values[index] = ((values[index - 1] * (period - 1)) + raw[index]) / period
    return pd.Series(values, index=frame.index, name="atr")


def _prepare_frame(frame, now=None, diagnostics=None):
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        return _reject(diagnostics, "history", "missing_ohlcv_columns")
    if len(frame) < REQUIRED_BARS:
        return _reject(
            diagnostics, "history", "insufficient_closed_bars",
            {"bars_received": len(frame), "bars_required": REQUIRED_BARS},
        )
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    for name in ("open", "high", "low", "close", "volume"):
        f[name] = pd.to_numeric(f[name], errors="coerce")
    fields = ["open", "high", "low", "close", "volume"]
    if f[["timestamp", *fields]].isna().any().any():
        return _reject(diagnostics, "data", "invalid_timestamp_or_ohlcv")
    if not np.isfinite(f[fields].to_numpy(dtype=float)).all():
        return _reject(diagnostics, "data", "non_finite_ohlcv")
    f = f.sort_values("timestamp").reset_index(drop=True)
    if f["timestamp"].duplicated().any() or (f["volume"] < 0).any():
        return _reject(diagnostics, "data", "duplicate_timestamps_or_negative_volume")
    if (f["high"] < f[["open", "close", "low"]].max(axis=1)).any():
        return _reject(diagnostics, "data", "invalid_ohlc_structure")
    if (f["low"] > f[["open", "close", "high"]].min(axis=1)).any():
        return _reject(diagnostics, "data", "invalid_ohlc_structure")

    f = f.iloc[-REQUIRED_BARS:].reset_index(drop=True)
    if any(int(stamp.timestamp()) % TIMEFRAME_SECONDS != 0 for stamp in f["timestamp"]):
        return _reject(diagnostics, "data", "off_grid_5m_timestamp")
    gaps = f["timestamp"].diff().dropna().dt.total_seconds()
    if len(gaps) != REQUIRED_BARS - 1 or not (gaps == TIMEFRAME_SECONDS).all():
        return _reject(
            diagnostics, "data", "gap_in_required_250_bars",
            {"bars_checked": len(f), "expected_gap_seconds": TIMEFRAME_SECONDS,
             "gap_count": int((gaps != TIMEFRAME_SECONDS).sum())},
        )
    current = f.iloc[-1]["timestamp"].to_pydatetime()
    now = now or datetime.now(timezone.utc)
    if current + timedelta(seconds=TIMEFRAME_SECONDS) > now.astimezone(timezone.utc):
        return _reject(
            diagnostics, "candle", "latest_candle_not_closed",
            {"candle_open_time": current.isoformat(),
             "candle_close_time": (current + timedelta(seconds=TIMEFRAME_SECONDS)).isoformat()},
        )
    return f


def _indicators(frame):
    f = frame.copy()
    f["atr"] = _wilder_atr(f, ATR_LENGTH)
    previous_close = f["close"].shift(1)
    f["true_range"] = pd.concat(
        [(f["high"] - f["low"]), (f["high"] - previous_close).abs(),
         (f["low"] - previous_close).abs()], axis=1,
    ).max(axis=1)
    f["tr_ratio"] = f["true_range"] / f["atr"].shift(1).replace(0, np.nan)
    f["atr_pct"] = f["atr"] / f["close"].replace(0, np.nan)
    f["rvol_baseline"] = f["volume"].shift(1).rolling(
        RVOL_LENGTH, min_periods=RVOL_LENGTH
    ).mean()
    f["rvol"] = f["volume"] / f["rvol_baseline"].replace(0, np.nan)
    f["donchian_high_prior"] = f["high"].shift(1).rolling(
        DONCHIAN_LENGTH, min_periods=DONCHIAN_LENGTH
    ).max()
    f["donchian_low_prior"] = f["low"].shift(1).rolling(
        DONCHIAN_LENGTH, min_periods=DONCHIAN_LENGTH
    ).min()
    f["ema21"] = f["close"].ewm(
        span=EMA_LENGTH, adjust=False, min_periods=EMA_LENGTH
    ).mean()
    candle_range = (f["high"] - f["low"]).replace(0, np.nan)
    f["close_location"] = (f["close"] - f["low"]) / candle_range
    return f


def evaluate_mex(
    frame, *, quote_volume_24h_usd=None,
    min_quote_volume_usd=100_000.0,
    round_trip_taker_fee_pct=0.10,
    min_rvol=MIN_RVOL, min_tr_ratio=MIN_TR_RATIO,
    max_extension_atr=MAX_EXTENSION_ATR,
    min_cost_multiple=MIN_ATR_TO_ROUND_TRIP_FEE,
    min_rr_to_tp2=MIN_RR_TO_TP2,
    now=None, diagnostics=None,
):
    """Evaluate one completed 5m candle; return a signal or the first failed gate."""
    f = _prepare_frame(frame, now=now, diagnostics=diagnostics)
    if f is None:
        return None
    x = _indicators(f)
    c, p = x.iloc[-1], x.iloc[-2]
    required = (
        "atr", "tr_ratio", "atr_pct", "rvol", "donchian_high_prior",
        "donchian_low_prior", "ema21", "close_location",
    )
    invalid = [name for name in required if not _finite(c[name])]
    if invalid:
        return _reject(diagnostics, "indicators", "indicator_warmup_or_invalid",
                       {"invalid_indicators": invalid})

    entry = float(c["close"])
    close_location = float(c["close_location"])
    if entry > float(c["donchian_high_prior"]):
        if float(c["tr_ratio"]) < min_tr_ratio:
            return _reject(diagnostics, "trigger", "tr_ratio_below_minimum",
                           _values(c, {"min_tr_ratio": min_tr_ratio}))
        if float(c["rvol"]) < min_rvol:
            return _reject(diagnostics, "trigger", "rvol_below_minimum",
                           _values(c, {"min_rvol": min_rvol}))
        if close_location < MIN_CLOSE_LOCATION_LONG:
            return _reject(diagnostics, "trigger", "close_location_not_strong_long",
                           _values(c, {"min_close_location": MIN_CLOSE_LOCATION_LONG}))
        if entry <= float(c["ema21"]):
            return _reject(diagnostics, "trigger", "close_not_above_ema21",
                           _values(c, {}))
        side = "LONG"
    elif entry < float(c["donchian_low_prior"]):
        if float(c["tr_ratio"]) < min_tr_ratio:
            return _reject(diagnostics, "trigger", "tr_ratio_below_minimum",
                           _values(c, {"min_tr_ratio": min_tr_ratio}))
        if float(c["rvol"]) < min_rvol:
            return _reject(diagnostics, "trigger", "rvol_below_minimum",
                           _values(c, {"min_rvol": min_rvol}))
        if close_location > MAX_CLOSE_LOCATION_SHORT:
            return _reject(diagnostics, "trigger", "close_location_not_strong_short",
                           _values(c, {"max_close_location": MAX_CLOSE_LOCATION_SHORT}))
        if entry >= float(c["ema21"]):
            return _reject(diagnostics, "trigger", "close_not_below_ema21",
                           _values(c, {}))
        side = "SHORT"
    else:
        return _reject(diagnostics, "trigger", "no_donchian_breakout",
                       _values(c, {}))

    atr_value = float(c["atr"])
    atr_pct = float(c["atr_pct"])
    extension_atr = abs(entry - float(c["ema21"])) / atr_value
    if extension_atr > max_extension_atr:
        return _reject(diagnostics, "safety", "entry_extension_exceeded",
                       _values(c, {"extension_atr": extension_atr,
                                   "max_extension_atr": max_extension_atr}))

    fee_pct = float(round_trip_taker_fee_pct)
    if not math.isfinite(fee_pct) or fee_pct <= 0:
        return _reject(diagnostics, "safety", "round_trip_taker_fee_not_configured",
                       {"round_trip_taker_fee_pct": fee_pct})
    minimum_atr_pct = min_cost_multiple * fee_pct / 100.0
    if atr_pct < minimum_atr_pct:
        return _reject(diagnostics, "safety", "atr_below_cost_floor",
                       _values(c, {"minimum_atr_pct": minimum_atr_pct,
                                   "fee_cost_multiple": min_cost_multiple}))

    try:
        turnover = float(quote_volume_24h_usd)
    except (TypeError, ValueError, OverflowError):
        turnover = float("nan")
    if not math.isfinite(turnover) or turnover < min_quote_volume_usd:
        return _reject(diagnostics, "safety", "quote_volume_24h_below_minimum",
                       _values(c, {"quote_volume_24h_usd": turnover,
                                   "min_quote_volume_usd": min_quote_volume_usd}))

    stop_floor = 0.8 * atr_value
    stop = (
        min(float(c["low"]), entry - stop_floor) if side == "LONG"
        else max(float(c["high"]), entry + stop_floor)
    )
    risk = entry - stop if side == "LONG" else stop - entry
    if not math.isfinite(risk) or risk <= 0:
        return _reject(diagnostics, "risk", "invalid_stop_distance", _values(c, {}))
    tp1 = entry + risk if side == "LONG" else entry - risk
    tp2 = entry + 2.0 * risk if side == "LONG" else entry - 2.0 * risk
    gross_reward = tp2 - entry if side == "LONG" else entry - tp2
    rr = gross_reward / risk
    if not math.isfinite(rr) or rr < min_rr_to_tp2:
        return _reject(diagnostics, "safety", "tp2_risk_reward_below_minimum",
                       _values(c, {"risk_reward_tp2": rr,
                                   "min_risk_reward_tp2": min_rr_to_tp2}))

    values = _values(c, {
        "side": side, "entry": entry, "stop_loss": stop, "tp1": tp1, "tp2": tp2,
        "risk_reward_tp2_net": rr, "quote_volume_24h_usd": turnover,
        "round_trip_taker_fee_pct": fee_pct,
        "extension_atr": extension_atr,
        "min_quote_volume_usd": min_quote_volume_usd,
    })
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"stage": "accepted", "reason": "mex_candidate_accepted",
                            "values": values})
    return {
        "strategy": "MEX", "side": side, "entry_reference": entry,
        "trigger_price": entry, "stop_loss": stop,
        "take_profit_1": tp1, "take_profit_2": tp2,
        "atr": atr_value, "risk_distance": risk,
        "risk_reward": rr, "rvol": float(c["rvol"]),
        "tr_ratio": float(c["tr_ratio"]), "atr_pct": atr_pct,
        "close_location": close_location,
        "quote_volume_24h_usd": turnover,
        "candle_time": c["timestamp"].to_pydatetime(),
        "reason": f"MEX {side}: 5m range break, TR ratio {float(c['tr_ratio']):.2f}, RVOL {float(c['rvol']):.2f}",
        "indicators": values,
    }


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _values(candle, extra):
    keys = ("close", "high", "low", "atr", "atr_pct", "tr_ratio", "rvol",
            "close_location", "donchian_high_prior", "donchian_low_prior", "ema21")
    result = {key: _round_or_none(candle[key]) for key in keys}
    result.update(extra)
    return result


def _round_or_none(value):
    try:
        number = float(value)
        return round(number, 8) if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _risk(frame, side, reference, atr_value):
    """Legacy compatibility helper; the minimal MEX path uses its own 0.8 ATR stop."""
    recent = frame.iloc[-5:]
    buffer = 0.15 * atr_value
    stop = (
        float(recent.low.min()) - buffer if side == "LONG"
        else float(recent.high.max()) + buffer
    )
    risk = reference - stop if side == "LONG" else stop - reference
    if not math.isfinite(risk) or risk <= 0:
        return None
    target = reference + 2.0 * risk if side == "LONG" else reference - 2.0 * risk
    return {"stop_loss": stop, "take_profit_1": target, "risk_distance_atr": risk / atr_value}
