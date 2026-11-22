from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.momentum import evaluate_mex


@pytest.mark.parametrize(
    ("volume", "expected_class", "expected_reason"),
    [
        (0.0, "no_trade", "zero_range_trigger_candle"),
        (125.0, "traded_at_one_price", "zero_range_trigger_candle"),
    ],
)
def test_mex_classifies_zero_range_trigger_candle_without_accepting_it(
    volume, expected_class, expected_reason
):
    now = datetime.now(timezone.utc)
    latest_open = int(now.timestamp()) // 300 * 300 - 300
    timestamps = [
        datetime.fromtimestamp(latest_open - (249 - i) * 300, tz=timezone.utc)
        for i in range(250)
    ]
    rows = []
    for stamp in timestamps[:-1]:
        rows.append({
            "timestamp": stamp, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.5, "volume": 100.0,
        })
    rows.append({
        "timestamp": timestamps[-1], "open": 100.0, "high": 100.0,
        "low": 100.0, "close": 100.0, "volume": volume,
    })
    diagnostics = {}

    result = evaluate_mex(pd.DataFrame(rows), now=now, diagnostics=diagnostics)

    assert result is None
    assert diagnostics["stage"] == "data"
    assert diagnostics["reason"] == expected_reason
    assert diagnostics["values"]["flat_bar_class"] == expected_class
    assert diagnostics["values"]["volume"] == volume
