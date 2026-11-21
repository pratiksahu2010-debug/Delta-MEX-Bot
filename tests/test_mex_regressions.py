import numpy as np
import pandas as pd

from app.momentum import evaluate_mex


def test_mex_squeeze_release_without_structure_break_is_rejected(monkeypatch):
    import app.momentum as momentum

    count = 80
    stamps = pd.date_range("2026-11-15T18:00:00Z", periods=count, freq="5min")
    frame = pd.DataFrame({
        "timestamp": stamps,
        "open": np.full(count, 100.0),
        "high": np.full(count, 101.7),
        "low": np.full(count, 99.5),
        "close": np.full(count, 100.2),
        "volume": np.full(count, 100.0),
    })
    frame.loc[79, ["open", "high", "low", "close", "volume"]] = [
        100.3, 100.7, 100.1, 100.5, 150.0
    ]

    def indicators(data):
        out = data.copy()
        values = {
            "atr": 2.0, "adx": 25.0, "plus_di": 30.0, "minus_di": 10.0,
            "vwap": 100.0, "vwap_session_bars": 80, "rvol": 1.5,
            "ttm_momentum": 0.5,
        }
        for key, value in values.items():
            out[key] = value
        out.loc[out.index[-2], "ttm_momentum"] = 0.4
        out["squeeze_on"] = False
        out.loc[out.index[-2], "squeeze_on"] = True
        out["squeeze_release"] = False
        out.loc[out.index[-1], "squeeze_release"] = True
        return out

    monkeypatch.setattr(momentum, "_indicators", indicators)
    diagnostics = {}
    assert evaluate_mex(frame, diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "breakout_close_not_above_range"
