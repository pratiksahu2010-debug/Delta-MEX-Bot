from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import app.mex_v3_shadow as shadow


def _base_frame(count=240):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=5 * (count - 1) + 10)
    rows = []
    for i in range(count):
        rows.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
            "volume": 100.0,
        })
    return pd.DataFrame(rows)


def _install_shadow_fixture(monkeypatch, *, side="LONG", run=8, deep=8, rvol=2.0,
                            adx=25.0, slope=0.1):
    frame = _base_frame()
    on = np.zeros(len(frame), dtype=bool)
    on[-(run + 1):-1] = True
    depth_rank = np.full(len(frame), 0.50, dtype=float)
    depth_rank[-(run + 1):-1] = 0.10
    if deep < run:
        depth_rank[-(run + 1):-1][:run - deep] = 0.50

    if side == "LONG":
        highs = np.linspace(110.0, 105.0, run)
        lows = np.linspace(99.0, 98.0, run)
        frame.loc[frame.index[-(run + 1):-1], "high"] = highs
        frame.loc[frame.index[-(run + 1):-1], "low"] = lows
        frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [
            110.1, 111.5, 109.9, 111.2
        ]
        plus_di, minus_di, vwap_value, momentum = 30.0, 10.0, 100.0, 1.0
    else:
        lows = np.linspace(97.0, 95.0, run)
        highs = np.linspace(101.0, 102.0, run)
        frame.loc[frame.index[-(run + 1):-1], "low"] = lows
        frame.loc[frame.index[-(run + 1):-1], "high"] = highs
        frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [
            89.9, 90.1, 88.5, 88.8
        ]
        plus_di, minus_di, vwap_value, momentum = 10.0, 30.0, 100.0, -1.0

    enriched = frame.copy()
    enriched["atr20"] = 1.0
    enriched["adx"] = adx
    enriched["plus_di"] = plus_di
    enriched["minus_di"] = minus_di
    enriched["bb_std"] = 0.1
    enriched["bb_mid"] = 100.0
    enriched["bb_upper"] = 100.2
    enriched["bb_lower"] = 99.8
    enriched["bb_width"] = 0.4
    enriched["kc_mid"] = 100.0
    enriched["kc_upper"] = 101.5
    enriched["kc_lower"] = 98.5
    enriched["kc_width"] = 3.0
    enriched["squeeze_ratio"] = 0.10
    enriched["squeeze_on"] = on
    enriched["squeeze_release"] = np.r_[False, on[:-1]] & ~on
    enriched["depth_rank"] = depth_rank
    enriched["ttm_momentum"] = momentum
    enriched["ttm_momentum_slope"] = slope if side == "LONG" else -abs(slope)
    enriched["volume_average"] = 100.0
    enriched["rvol"] = rvol
    enriched["vwap"] = vwap_value
    enriched["vwap_session_volume"] = 24000.0
    episodes = shadow._episode_columns(
        on, enriched.high.to_numpy(), enriched.low.to_numpy(),
        depth_rank, 0.20,
    )
    for key, values in episodes.items():
        enriched[key] = values
    monkeypatch.setattr(shadow, "_indicators", lambda _frame, **_kwargs: enriched.copy())
    return frame, enriched


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_mex_v3_shadow_accepts_only_directional_release_breakout(monkeypatch, side):
    frame, enriched = _install_shadow_fixture(monkeypatch, side=side)
    diagnostics = {}

    result = shadow.evaluate_mex_v3_shadow(frame, diagnostics=diagnostics)

    assert result is not None, diagnostics
    assert result["strategy"] == "MEX_V3_SHADOW"
    assert result["side"] == side
    assert diagnostics["reason"] == "mex_v3_shadow_candidate"
    assert diagnostics["values"]["squeeze_duration_bars"] == 8
    assert diagnostics["values"]["deep_bars"] == 8


def test_release_range_covers_entire_squeeze_episode_not_only_last_six_bars(monkeypatch):
    frame, enriched = _install_shadow_fixture(monkeypatch, side="LONG", run=8)
    # The first bar in the eight-bar compression has the episode's highest high.
    assert enriched["release_range_high"].iloc[-1] == pytest.approx(
        enriched["high"].iloc[-9]
    )
    assert enriched["release_range_high"].iloc[-1] > enriched["high"].iloc[-6:-1].max()


def test_mex_v3_shadow_rejects_short_squeeze_duration(monkeypatch):
    frame, _ = _install_shadow_fixture(monkeypatch, run=7)
    diagnostics = {}

    result = shadow.evaluate_mex_v3_shadow(frame, min_bars=8, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "shadow_squeeze_duration_too_short"


def test_mex_v3_shadow_requires_half_of_episode_to_be_deep(monkeypatch):
    frame, _ = _install_shadow_fixture(monkeypatch, run=8, deep=3)
    diagnostics = {}

    result = shadow.evaluate_mex_v3_shadow(
        frame, min_deep_fraction=0.50, diagnostics=diagnostics
    )

    assert result is None
    assert diagnostics["reason"] == "shadow_squeeze_depth_insufficient"


def test_mex_v3_shadow_keeps_rvol_floor_at_1_50(monkeypatch):
    frame, _ = _install_shadow_fixture(monkeypatch, rvol=1.49)
    diagnostics = {}

    result = shadow.evaluate_mex_v3_shadow(frame, min_rvol=1.50, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "shadow_rvol_below_minimum"


def test_mex_v3_shadow_requires_momentum_slope_alignment(monkeypatch):
    frame, _ = _install_shadow_fixture(monkeypatch, side="LONG", slope=-0.1)
    diagnostics = {}

    result = shadow.evaluate_mex_v3_shadow(frame, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "shadow_direction_or_breakout_not_confirmed"


def test_mex_v3_shadow_does_not_use_ratio_below_one_as_replacement_for_boundary_gate():
    frame = _base_frame()
    indicators = shadow._indicators(frame)
    expected = (
        (indicators["bb_upper"] < indicators["kc_upper"])
        & (indicators["bb_lower"] > indicators["kc_lower"])
    )
    pd.testing.assert_series_equal(
        indicators["squeeze_on"], expected, check_names=False
    )
