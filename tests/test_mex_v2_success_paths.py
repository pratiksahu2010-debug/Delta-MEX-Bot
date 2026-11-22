"""Regression cases for the minimal MEX implementation."""
from test_mex_v2 import make_frame, evaluate


def test_success_path_keeps_trigger_close_as_entry_and_targets_1r_2r():
    frame, now = make_frame("LONG")
    result = evaluate(frame, now)
    assert result is not None
    assert result["entry_reference"] == frame.iloc[-1]["close"]
    risk = result["entry_reference"] - result["stop_loss"]
    assert result["take_profit_1"] == result["entry_reference"] + risk
    assert result["take_profit_2"] == result["entry_reference"] + 2 * risk


def test_short_success_path_is_mirrored():
    frame, now = make_frame("SHORT")
    result = evaluate(frame, now)
    assert result is not None
    assert result["side"] == "SHORT"
    risk = result["stop_loss"] - result["entry_reference"]
    assert result["take_profit_1"] == result["entry_reference"] - risk
    assert result["take_profit_2"] == result["entry_reference"] - 2 * risk
