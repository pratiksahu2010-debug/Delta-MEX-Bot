import numpy as np
import pandas as pd

from app.momentum import evaluate_mex


def test_mex_squeeze_expansion_without_structure_break_is_rejected(monkeypatch):
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
    # A green expansion candle, but still below the prior 10-bar high.
    frame.loc[79, ["open", "high", "low", "close", "volume"]] = [
        100.3, 100.7, 100.1, 100.5, 150.0
    ]

    def indicators(data):
        out = data.copy()
        values = {
            "ema9": 100.4, "ema21": 100.0, "rsi": 60.0, "atr": 2.0,
            "adx": 25.0, "plus_di": 30.0, "minus_di": 10.0,
            "vwap": 100.0, "rvol": 1.5, "atr_expansion": 1.1,
            "roc3": 0.2, "macd_hist": 0.2, "bb_width": 0.04,
        }
        for key, value in values.items():
            out[key] = value
        out.loc[out.index[-21:-2], "bb_width"] = 0.03
        out.loc[out.index[-2], "bb_width"] = 0.02
        return out

    monkeypatch.setattr(momentum, "_indicators", indicators)
    assert evaluate_mex(frame) is None
