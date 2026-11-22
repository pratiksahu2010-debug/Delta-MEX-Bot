from app.config import Settings


def test_balanced_mex_threshold_defaults():
    assert Settings.model_fields["mex_min_rvol"].default == 1.15
    assert Settings.model_fields["mex_min_tr_ratio"].default == 1.5
    assert Settings.model_fields["mex_min_breakout_clearance_atr"].default == 0.10
    assert Settings.model_fields["signal_only"].default is True
    assert Settings.model_fields["orders_enabled"].default is False
