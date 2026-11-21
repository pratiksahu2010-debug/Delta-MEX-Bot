import pandas as pd
import pytest

from app.replay import replay_dataframe, validate_replay_data


def sample_frame(count=60):
    stamps = pd.date_range("2026-11-10T00:00:00Z", periods=count, freq="5min")
    return pd.DataFrame({
        "symbol": ["BTCUSD"] * count,
        "timestamp": stamps,
        "open": [100.0] * count,
        "high": [100.2] * count,
        "low": [99.8] * count,
        "close": [100.0] * count,
        "volume": [100.0] * count,
    })


def test_replay_rejects_retired_normal_strategy():
    with pytest.raises(ValueError, match="Unsupported replay strategies"):
        replay_dataframe(sample_frame(), strategies=("NORMAL",))


def test_replay_validates_duplicate_timestamps():
    frame = sample_frame()
    frame.loc[59, "timestamp"] = frame.loc[58, "timestamp"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_replay_data(frame)


def test_replay_rejects_non_finite_and_invalid_ohlc():
    frame = sample_frame()
    frame.loc[59, "close"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_replay_data(frame)
    frame = sample_frame()
    frame.loc[59, "high"] = 98.0
    with pytest.raises(ValueError, match="invalid high"):
        validate_replay_data(frame)














def test_load_strategy_universes_rejects_overlap(tmp_path):
    from app.replay import load_strategy_universes

    momentum = [f"S{i:03d}USD" for i in range(60)]
    tpr_rb = [f"T{i:03d}USD" for i in range(59)] + [momentum[0]]
    (tmp_path / "momentum_symbols.txt").write_text("\n".join(momentum), encoding="utf-8")
    (tmp_path / "tpr_rb_symbols.txt").write_text("\n".join(tpr_rb), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_strategy_universes(tmp_path)
