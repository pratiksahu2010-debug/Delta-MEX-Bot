from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import app.momentum as momentum


def _closed_breakout_frame(side: str) -> pd.DataFrame:
    """A valid, closed 5m breakout with a narrow prior range and structural stop."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=5 * 59 + 10)
    base = 102.0 if side == "LONG" else 98.0
    rows = []
    for index in range(60):
        stamp = start + timedelta(minutes=5 * index)
        rows.append({
            "timestamp": stamp,
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base,
            "volume": 100.0,
        })

    if side == "LONG":
        rows[-1].update(open=102.1, high=103.0, low=102.05, close=102.8)
    else:
        rows[-1].update(open=97.9, high=97.95, low=96.8, close=97.0)
    return pd.DataFrame(rows)


def _install_indicators(monkeypatch, frame, side: str, *, adx_value=19.0, rvol=1.50):
    enriched = frame.copy()
    for column in (
        "atr", "adx", "plus_di", "minus_di", "squeeze_on",
        "squeeze_release", "ttm_momentum", "rvol", "vwap",
        "vwap_session_bars",
    ):
        enriched[column] = 0.0

    enriched["atr"] = 1.0
    enriched["adx"] = adx_value
    enriched["rvol"] = rvol
    enriched["squeeze_on"] = False
    enriched["squeeze_release"] = False
    enriched["vwap_session_bars"] = 12

    if side == "LONG":
        enriched.loc[enriched.index[-2], "ttm_momentum"] = 0.5
        enriched.loc[enriched.index[-1], "ttm_momentum"] = 1.0
        enriched["plus_di"] = 30.0
        enriched["minus_di"] = 10.0
        enriched["vwap"] = 102.4
    else:
        enriched.loc[enriched.index[-2], "ttm_momentum"] = -0.5
        enriched.loc[enriched.index[-1], "ttm_momentum"] = -1.0
        enriched["plus_di"] = 10.0
        enriched["minus_di"] = 30.0
        enriched["vwap"] = 97.5

    enriched.loc[enriched.index[-2], "squeeze_on"] = True
    enriched.loc[enriched.index[-1], "squeeze_release"] = True
    monkeypatch.setattr(momentum, "_indicators", lambda _frame: enriched.copy())


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_mex_accepts_complete_directional_setup_and_builds_two_r_target(
    monkeypatch, side
):
    frame = _closed_breakout_frame(side)
    _install_indicators(monkeypatch, frame, side)
    diagnostics = {}

    result = momentum.evaluate_mex(
        frame, adx_min=19.0, min_rvol=1.50, diagnostics=diagnostics
    )

    assert result is not None
    assert result["side"] == side
    assert result["strategy"] == "MEX"
    assert diagnostics["stage"] == "accepted"
    assert result["indicators"]["adx"] == 19.0
    assert result["indicators"]["rvol"] == 1.5
    assert result["trigger_price"] == (
        frame.iloc[-1]["high"] if side == "LONG" else frame.iloc[-1]["low"]
    )
    risk = (
        result["entry_reference"] - result["stop_loss"]
        if side == "LONG"
        else result["stop_loss"] - result["entry_reference"]
    )
    reward = (
        result["take_profit_1"] - result["entry_reference"]
        if side == "LONG"
        else result["entry_reference"] - result["take_profit_1"]
    )
    assert reward == pytest.approx(2.0 * risk)


@pytest.mark.parametrize(
    ("side", "adx_value", "rvol", "expected_reason"),
    [
        ("LONG", 18.99, 1.50, "adx_below_minimum"),
        ("SHORT", 19.0, 1.49, "relative_volume_below_minimum"),
    ],
)
def test_mex_rejects_values_below_adx_or_rvol_boundary(
    monkeypatch, side, adx_value, rvol, expected_reason
):
    frame = _closed_breakout_frame(side)
    _install_indicators(
        monkeypatch, frame, side, adx_value=adx_value, rvol=rvol
    )
    diagnostics = {}

    result = momentum.evaluate_mex(
        frame, adx_min=19.0, min_rvol=1.50, diagnostics=diagnostics
    )

    assert result is None
    assert diagnostics["reason"] == expected_reason


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_mex_rejects_opposing_di_alignment(monkeypatch, side):
    frame = _closed_breakout_frame(side)
    _install_indicators(monkeypatch, frame, side)
    enriched = momentum._indicators(frame)
    if side == "LONG":
        enriched.loc[enriched.index[-1], "plus_di"] = 10.0
        enriched.loc[enriched.index[-1], "minus_di"] = 30.0
    else:
        enriched.loc[enriched.index[-1], "plus_di"] = 30.0
        enriched.loc[enriched.index[-1], "minus_di"] = 10.0
    monkeypatch.setattr(momentum, "_indicators", lambda _frame: enriched.copy())
    diagnostics = {}

    result = momentum.evaluate_mex(frame, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "four_indicator_direction_not_aligned"
