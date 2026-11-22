from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.momentum import evaluate_mex, _indicators, _wilder_atr


def make_frame(side="LONG", count=250):
    now = datetime.now(timezone.utc)
    last_open = datetime.fromtimestamp(int(now.timestamp() // 300) * 300 - 300, timezone.utc)
    start = last_open - timedelta(minutes=5 * (count - 1))
    rows = []
    for i in range(count):
        stamp = start + timedelta(minutes=5 * i)
        rows.append({
            "timestamp": stamp, "open": 100.0, "high": 100.15,
            "low": 99.85, "close": 100.0, "volume": 100.0,
        })
    if side == "LONG":
        rows[-1].update(open=100.05, high=100.60, low=100.00, close=100.50, volume=300.0)
    elif side == "SHORT":
        rows[-1].update(open=99.95, high=100.00, low=99.40, close=99.50, volume=300.0)
    return pd.DataFrame(rows), last_open + timedelta(minutes=5, seconds=1)


def evaluate(frame, now, **kwargs):
    return evaluate_mex(
        frame, now=now,
        round_trip_taker_fee_pct=0.10, diagnostics=kwargs.pop("diagnostics", {}),
        **kwargs,
    )


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_minimal_mex_accepts_strict_closed_candle_breakout(side):
    frame, now = make_frame(side)
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)

    assert result is not None, diagnostics
    assert result["side"] == side
    assert result["strategy"] == "MEX"
    assert result["entry_reference"] == pytest.approx(frame.iloc[-1].close)
    assert result["risk_reward"] == pytest.approx(2.0)
    assert diagnostics["reason"] == "mex_candidate_accepted"
    assert result["rvol"] == pytest.approx(3.0)
    assert result["tr_ratio"] == pytest.approx(2.0)


def test_wilder_atr_uses_sma_seed_then_recursive_smoothing():
    frame, _ = make_frame("LONG")
    atr = _wilder_atr(frame, 14)
    assert atr.iloc[12] != atr.iloc[12]  # not enough bars before the seed
    assert atr.iloc[13] == pytest.approx(0.30)
    assert atr.iloc[-2] == pytest.approx(0.30)
    expected = (13 * 0.30 + 0.60) / 14
    assert atr.iloc[-1] == pytest.approx(expected)


def test_rvol_uses_only_previous_twenty_completed_bars():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-20:], "volume"] = 100.0
    frame.loc[frame.index[-1], "volume"] = 250.0
    result = evaluate(frame, now)
    assert result is not None
    assert result["rvol"] == pytest.approx(2.5)


def test_prior_donchian_excludes_the_trigger_candle():
    frame, now = make_frame("LONG")
    diagnostics = {}
    result = evaluate_mex(
        frame, now=now,
        round_trip_taker_fee_pct=0.10, diagnostics=diagnostics,
    )
    assert result is not None, diagnostics
    assert result["indicators"]["donchian_high_prior"] < frame.iloc[-1]["high"]


def test_rejects_fewer_than_250_bars():
    frame, now = make_frame("LONG", count=249)
    diagnostics = {}
    result = evaluate_mex(frame, now=now,
                          round_trip_taker_fee_pct=0.10, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "insufficient_closed_bars"


def test_rejects_any_gap_inside_required_250_bar_window():
    frame, now = make_frame("LONG")
    frame.loc[100:, "timestamp"] += timedelta(minutes=5)
    diagnostics = {}
    result = evaluate_mex(frame, now=now,
                          round_trip_taker_fee_pct=0.10, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "gap_in_required_250_bars"


def test_rejects_off_grid_5m_timestamps():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], "timestamp"] += timedelta(minutes=1)
    diagnostics = {}
    result = evaluate_mex(frame, now=now,
                          round_trip_taker_fee_pct=0.10, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "off_grid_5m_timestamp"


def test_rejects_forming_latest_bar():
    frame, _ = make_frame("LONG")
    diagnostics = {}
    result = evaluate_mex(
        frame, now=frame.iloc[-1].timestamp.to_pydatetime() + timedelta(minutes=4),
        round_trip_taker_fee_pct=0.10,
        diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["reason"] == "latest_candle_not_closed"


def test_rvol_is_first_trigger_gate_after_breakout():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], "volume"] = 149.0
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "rvol_below_minimum"


def test_zero_current_volume_fails_rvol_gate_not_indicator_warmup():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], "volume"] = 0.0
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "rvol_below_minimum"


def test_tr_ratio_is_first_trigger_gate_after_breakout():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], ["high", "low", "close"]] = [100.45, 100.0, 100.4]
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "tr_ratio_below_minimum"


def test_close_location_gate_rejects_weak_breakout_close():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [100.05, 100.6, 99.9, 100.35]
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "close_location_not_strong_long"


def test_extension_gate_rejects_chasing_extended_bar():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [100.5, 102.0, 100.4, 101.9]
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "entry_extension_exceeded"


def test_cost_gate_rejects_atr_below_three_times_round_trip_fee():
    frame, now = make_frame("LONG")
    frame.loc[:, ["high", "low"]] = [100.02, 99.98]
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [100.01, 100.12, 100.0, 100.10]
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "atr_below_cost_floor"


def test_fixed_universe_symbol_is_not_rejected_for_missing_turnover():
    frame, now = make_frame("LONG")
    diagnostics = {}
    result = evaluate_mex(frame, now=now, round_trip_taker_fee_pct=0.10, diagnostics=diagnostics)
    assert result is not None, diagnostics
    assert result["side"] == "LONG"
    assert "quote_volume_24h_usd" not in result
    assert "quote_volume_24h_usd" not in result["indicators"]


def test_stop_is_never_tighter_than_0_8_atr_and_targets_are_1r_2r():
    frame, now = make_frame("LONG")
    result = evaluate(frame, now)
    assert result is not None
    risk = result["entry_reference"] - result["stop_loss"]
    assert risk >= 0.8 * result["atr"]
    assert result["take_profit_1"] == pytest.approx(result["entry_reference"] + risk)
    assert result["take_profit_2"] == pytest.approx(result["entry_reference"] + 2 * risk)


def test_only_the_first_failed_gate_is_reported():
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], "volume"] = 100.0
    diagnostics = {}
    result = evaluate(frame, now, diagnostics=diagnostics)
    assert result is None
    assert diagnostics["reason"] == "rvol_below_minimum"
