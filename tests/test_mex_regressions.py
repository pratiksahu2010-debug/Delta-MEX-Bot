"""The old TTM-squeeze regression is superseded by the minimal MEX rules."""
from test_mex_v2 import make_frame, evaluate


def test_current_bar_is_excluded_from_prior_20_bar_donchian_range():
    frame, now = make_frame("LONG")
    result = evaluate(frame, now)
    assert result is not None
    assert result["indicators"]["donchian_high_prior"] < frame.iloc[-1]["high"]


def test_gap_in_250_bar_window_is_a_hard_rejection():
    frame, now = make_frame("LONG")
    frame.loc[150:, "timestamp"] += __import__("datetime").timedelta(minutes=5)
    from app.momentum import evaluate_mex
    diagnostics = {}
    assert evaluate_mex(frame, now=now, quote_volume_24h_usd=250000,
                        round_trip_taker_fee_pct=0.10,
                        diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "gap_in_required_250_bars"


def test_zero_range_latest_candle_is_reported_as_data_rejection():
    from app.momentum import evaluate_mex
    frame, now = make_frame("LONG")
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = 100.0
    diagnostics = {}
    assert evaluate_mex(frame, now=now, quote_volume_24h_usd=250000,
                        round_trip_taker_fee_pct=0.10,
                        diagnostics=diagnostics) is None
    assert diagnostics["stage"] == "data"
    assert diagnostics["reason"] == "zero_range_trigger_candle"


def test_non_positive_ohlc_is_rejected_as_market_data_corruption():
    from app.momentum import evaluate_mex
    frame, now = make_frame("LONG")
    frame.loc[frame.index[10], "low"] = 0.0
    diagnostics = {}
    assert evaluate_mex(frame, now=now, quote_volume_24h_usd=250000,
                        round_trip_taker_fee_pct=0.10,
                        diagnostics=diagnostics) is None
    assert diagnostics["stage"] == "data"
    assert diagnostics["reason"] == "non_positive_ohlc"
