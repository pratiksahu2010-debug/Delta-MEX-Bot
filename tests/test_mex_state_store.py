from datetime import datetime, timedelta, timezone

from app.state_store import SignalStore


def _payload(symbol, candle_time):
    return {
        "symbol": symbol, "side": "LONG", "strategy": "MEX",
        "setup_time": candle_time, "entry": 100.0, "stop_loss": 99.0,
        "take_profit_1": 101.0, "take_profit_2": 102.0,
    }


def test_mex_state_and_outbox_survive_store_restart(tmp_path):
    path = tmp_path / "mex.sqlite3"
    store = SignalStore(str(path))
    now = datetime.now(timezone.utc)
    candle = now - timedelta(minutes=5)
    candle_text = candle.isoformat()
    key = f"BTCUSD:MEX:LONG:{candle_text}"

    accepted, reason = store.claim_mex_signal(
        key, "BTCUSD", "LONG", candle_text, _payload("BTCUSD", candle_text), now=now
    )
    assert accepted and reason == "accepted"

    reopened = SignalStore(str(path))
    state = reopened.mex_state()
    assert state["signals_last_hour"] == 1
    assert state["states"][0]["state"] == "IDLE"
    assert state["states"][0]["last_side"] == "LONG"
    assert reopened.due_alerts()[0][0] == key


def test_mex_same_symbol_can_signal_on_next_closed_candle(tmp_path):
    store = SignalStore(str(tmp_path / "mex.sqlite3"))
    now = datetime(2026, 11, 22, 10, 0, tzinfo=timezone.utc)
    candle = now - timedelta(minutes=5)
    stamp = candle.isoformat()
    accepted, _ = store.claim_mex_signal(
        f"BTCUSD:{stamp}", "BTCUSD", "LONG", stamp, _payload("BTCUSD", stamp), now=now
    )
    assert accepted

    next_candle = candle + timedelta(minutes=5)
    next_stamp = next_candle.isoformat()
    accepted, reason = store.claim_mex_signal(
        f"BTCUSD:{next_stamp}", "BTCUSD", "LONG", next_stamp,
        _payload("BTCUSD", next_stamp), now=now + timedelta(minutes=5)
    )
    assert accepted and reason == "accepted"
    assert store.mex_state()["states"][0]["state"] == "IDLE"


def test_mex_no_per_symbol_cooldown_after_new_signal(tmp_path):
    store = SignalStore(str(tmp_path / "mex.sqlite3"))
    now = datetime(2026, 11, 22, 10, 0, tzinfo=timezone.utc)
    candle = now - timedelta(minutes=5)
    stamp = candle.isoformat()
    assert store.claim_mex_signal(
        f"BTCUSD:{stamp}", "BTCUSD", "LONG", stamp, _payload("BTCUSD", stamp), now=now
    )[0]

    next_candle = candle + timedelta(minutes=60)
    next_stamp = next_candle.isoformat()
    accepted, reason = store.claim_mex_signal(
        f"BTCUSD:{next_stamp}", "BTCUSD", "SHORT", next_stamp,
        _payload("BTCUSD", next_stamp), now=now + timedelta(minutes=60)
    )
    assert accepted, reason
    state = store.mex_state()["states"][0]
    assert state["state"] == "IDLE"
    assert state["last_side"] == "SHORT"


def test_mex_global_hourly_cap_is_five_across_all_symbols(tmp_path):
    store = SignalStore(str(tmp_path / "mex.sqlite3"))
    # Use the real current time because mex_state() reports its rolling-hour
    # count relative to the wall clock, not the injected claim timestamp.
    now = datetime.now(timezone.utc)
    for index in range(5):
        symbol = f"COIN{index}USD"
        stamp = (now - timedelta(minutes=5)).isoformat()
        accepted, reason = store.claim_mex_signal(
            f"{symbol}:{stamp}", symbol, "LONG", stamp,
            _payload(symbol, stamp), now=now
        )
        assert accepted, reason

    stamp = (now - timedelta(minutes=5)).isoformat()
    accepted, reason = store.claim_mex_signal(
        f"COIN5USD:{stamp}", "COIN5USD", "LONG", stamp,
        _payload("COIN5USD", stamp), now=now
    )
    assert not accepted
    assert reason == "global_hourly_signal_limit"
    assert store.mex_state()["signals_last_hour"] == 5


def test_mex_duplicate_signal_cannot_be_queued_twice(tmp_path):
    store = SignalStore(str(tmp_path / "mex.sqlite3"))
    now = datetime.now(timezone.utc)
    stamp = (now - timedelta(minutes=5)).isoformat()
    key = f"BTCUSD:{stamp}"
    assert store.claim_mex_signal(
        key, "BTCUSD", "LONG", stamp, _payload("BTCUSD", stamp), now=now
    )[0]
    # Same-candle retry is rejected by the durable symbol state before another outbox insert.
    accepted, reason = store.claim_mex_signal(
        key, "BTCUSD", "LONG", stamp, _payload("BTCUSD", stamp), now=now
    )
    assert not accepted
    assert reason == "duplicate_symbol_candle"
    assert len(store.due_alerts()) == 1


def test_log_only_suppresses_only_pending_legacy_mex_outbox_items(tmp_path):
    store = SignalStore(str(tmp_path / "mex.sqlite3"))
    now = datetime.now(timezone.utc).isoformat()
    mex = _payload("BTCUSD", now)
    orb = {**_payload("ETHUSD", now), "strategy": "ORB"}
    store.enqueue_alert("legacy-mex", mex)
    store.enqueue_alert("unrelated-orb", orb)

    assert store.suppress_pending_mex_alerts() == 1
    due = store.due_alerts()
    assert [item[0] for item in due] == ["unrelated-orb"]


def test_log_only_candidate_does_not_consume_live_hourly_alert_cap(tmp_path):
    store = SignalStore(str(tmp_path / "mex.sqlite3"))
    now = datetime.now(timezone.utc)
    stamp = (now - timedelta(minutes=5)).isoformat()
    accepted, reason = store.claim_mex_signal(
        "LOGONLY:MEX:LONG:" + stamp, "LOGONLY", "LONG", stamp,
        _payload("LOGONLY", stamp), now=now, enqueue_alert=False,
    )
    assert accepted, reason
    assert store.mex_state()["signals_last_hour"] == 0

    live_stamp = (now - timedelta(minutes=5)).isoformat()
    accepted, reason = store.claim_mex_signal(
        "LIVE:MEX:LONG:" + live_stamp, "LIVE", "LONG", live_stamp,
        _payload("LIVE", live_stamp), now=now, enqueue_alert=True,
    )
    assert accepted, reason
    assert store.mex_state()["signals_last_hour"] == 1
    assert [key for key, _ in store.due_alerts()] == ["LIVE:MEX:LONG:" + live_stamp]
