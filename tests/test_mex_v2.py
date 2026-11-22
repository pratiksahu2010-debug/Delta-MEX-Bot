from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.momentum import _indicators, evaluate_mex


def _frame(count=90, *, latest_age_minutes=10):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=5 * (count - 1) + latest_age_minutes)
    rows = []
    price = 100.0
    for i in range(count):
        # Tight, valid bars provide a deterministic indicator warm-up fixture.
        close = price + 0.015 * np.sin(i / 3.0)
        rows.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "open": close - 0.01,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": 100.0 + (i % 7),
        })
        price = close
    return pd.DataFrame(rows)


def test_mex_rejects_forming_candle_before_indicator_evaluation():
    frame = _frame(latest_age_minutes=1)
    diagnostics = {}

    result = evaluate_mex(frame, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "latest_candle_not_closed"


def test_ttm_squeeze_columns_and_release_state_are_calculated():
    frame = _frame()
    indicators = _indicators(frame)

    for column in (
        "bb_upper", "bb_lower", "bb_width", "kc_upper", "kc_lower",
        "kc_width", "squeeze_ratio", "squeeze_on", "squeeze_release",
        "ttm_momentum", "rvol", "volume_average", "adx", "plus_di",
        "minus_di", "vwap", "vwap_session_volume",
    ):
        assert column in indicators.columns

    expected_release = (
        indicators["squeeze_on"].shift(1, fill_value=False)
        & ~indicators["squeeze_on"]
    )
    pd.testing.assert_series_equal(
        indicators["squeeze_release"], expected_release,
        check_names=False,
    )


def test_mex_requires_squeeze_release():
    frame = _frame()
    diagnostics = {}

    result = evaluate_mex(frame, adx_min=19, min_rvol=1.5, diagnostics=diagnostics)

    assert result is None
    # The low-range fixture is deliberately not a breakout release.
    assert diagnostics["reason"] == "ttm_squeeze_not_released"



def test_zero_rvol_baseline_has_specific_diagnostic():
    frame = _frame()
    # The current candle trades, but the preceding 20 completed candles have
    # zero volume. This is not an ADX or squeeze failure.
    frame.loc[frame.index[-21:-1], "volume"] = 0.0
    diagnostics = {}

    result = evaluate_mex(frame, adx_min=19, min_rvol=1.5, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "rvol_zero_or_missing_volume_baseline"
    assert diagnostics["values"]["current_volume"] > 0
    assert diagnostics["values"]["volume_average_previous_20"] == 0.0


def test_squeeze_rejection_reports_compression_ratios():
    frame = _frame()
    diagnostics = {}

    result = evaluate_mex(frame, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["reason"] == "ttm_squeeze_not_released"
    assert "squeeze_ratio_previous" in diagnostics["values"]
    assert "squeeze_ratio_current" in diagnostics["values"]
