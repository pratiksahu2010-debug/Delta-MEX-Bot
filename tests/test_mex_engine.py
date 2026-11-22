import asyncio
from datetime import datetime, timedelta, timezone

from app.models import Candle
from app.signal_engine import SignalEngine
class FakeNotifier:
    def __init__(self):
        self.alerts = []

    async def send_signal(self, alert, feedback_token=""):
        self.alerts.append(alert)
        return True


def _closed_history():
    now = datetime.now(timezone.utc)
    bucket = int(now.timestamp()) // 300 * 300
    last_open = datetime.fromtimestamp(bucket - 300, timezone.utc)
    start = last_open - timedelta(minutes=5 * 249)
    bars = [
        Candle(
            timestamp=start + timedelta(minutes=5 * i),
            open=100.0, high=100.15, low=99.85, close=100.0, volume=100.0,
        )
        for i in range(250)
    ]
    bars[-1] = Candle(
        timestamp=last_open, open=100.05, high=100.60,
        low=100.00, close=100.50, volume=300.0,
    )
    return bars


def _engine(tmp_path, log_only):
    from types import SimpleNamespace
    settings = SimpleNamespace(
        delta_base_url="https://api.india.delta.exchange",
        state_db_path=str(tmp_path / "mex-engine.sqlite3"),
        symbol_list=["BTCUSD"], momentum_symbol_list=["BTCUSD"],
        tpr_rb_symbol_list=[], bot_module="MEX", mex_enabled=True,
        mex_log_only=log_only, mex_min_rvol=1.5, mex_min_tr_ratio=2.0,
        mex_min_24h_quote_volume_usd=100000.0,
        mex_round_trip_taker_fee_pct=0.10,
        candle_poll_interval_seconds=0.0, max_candle_lag_seconds=600,
        quote_max_age_seconds=3.0, signal_max_age_bars=2,
        outcome_horizon_bars=12, outcome_fee_pct=0.0,
        outcome_slippage_pct=0.0, outcome_tp2_r_multiple=2.0,
        trigger_confirmation_seconds=0.5,
        trigger_confirmation_max_gap_seconds=1.5,
    )
    return SignalEngine(settings, FakeNotifier())


def test_mex_log_only_records_candidate_and_persists_state_without_telegram(tmp_path, monkeypatch):
    async def run():
        engine = _engine(tmp_path, True)
        bars = _closed_history()

        async def candles(*_args, **_kwargs):
            return bars

        async def tickers():
            return [{"symbol": "BTCUSD", "turnover_usd": 250000.0}]

        monkeypatch.setattr(engine.rest, "candles", candles)
        monkeypatch.setattr(engine.rest, "tickers", tickers)
        await engine.refresh_symbol("BTCUSD")

        assert engine.mex_runtime["candidates"] == 1
        assert engine.mex_runtime["log_only_candidates"] == 1
        assert engine.notifier.alerts == []
        assert engine.store.due_alerts() == []
        mex_state = engine.store.mex_state()
        assert mex_state["signals_last_hour"] == 1
        assert mex_state["states"][0]["state"] == "COOLDOWN"
        assert mex_state["states"][0]["last_breakout_level"] is not None
        record = engine.journal.latest_record()
        assert record["reason"] == "log_only_candidate_state_claimed"
        assert record["decision"] == "rejected"
        await engine.rest.close()

    asyncio.run(run())


def test_mex_live_signal_uses_trigger_close_without_waiting_for_quote(tmp_path, monkeypatch):
    async def run():
        engine = _engine(tmp_path, False)
        bars = _closed_history()

        async def candles(*_args, **_kwargs):
            return bars

        async def tickers():
            return [{"symbol": "BTCUSD", "turnover_usd": 250000.0}]

        monkeypatch.setattr(engine.rest, "candles", candles)
        monkeypatch.setattr(engine.rest, "tickers", tickers)
        await engine.refresh_symbol("BTCUSD")

        assert len(engine.notifier.alerts) == 1
        alert = engine.notifier.alerts[0]
        assert alert["entry"] == bars[-1].close
        assert alert["take_profit_1"] > alert["entry"]
        assert alert["take_profit_2"] > alert["take_profit_1"]
        assert not engine.momentum_pending
        state = engine.store.mex_state()["states"][0]
        assert state["state"] == "COOLDOWN"
        assert engine.store.outbox_health()["sent"] == 1
        await engine.rest.close()

    asyncio.run(run())
