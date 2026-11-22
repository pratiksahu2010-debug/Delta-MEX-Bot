from datetime import datetime, timedelta, timezone

import pandas as pd

from app.momentum import evaluate_mex


def test_rvol_rejection_includes_current_volume_and_baseline():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(250):
        close = 100.0 + index * 0.01
        if index == 249:
            close += 0.20
        rows.append({
            "timestamp": start + timedelta(minutes=5 * index),
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": 10.0 if index == 249 else 100.0,
        })
    frame = pd.DataFrame(rows)
    diagnostics = {}
    now = rows[-1]["timestamp"] + timedelta(minutes=5, seconds=1)

    result = evaluate_mex(
        frame,
        min_tr_ratio=0.5,
        min_rvol=1.5,
        now=now,
        diagnostics=diagnostics,
    )

    assert result is None
    assert diagnostics["reason"] == "rvol_below_minimum"
    assert diagnostics["values"]["volume"] == 10.0
    assert diagnostics["values"]["rvol_baseline"] == 100.0
    assert diagnostics["values"]["rvol"] == 0.1
