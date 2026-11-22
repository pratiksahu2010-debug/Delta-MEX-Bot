# Delta Exchange India — Minimal MEX Signal Bot

Crypto-only, signal-only Momentum Explosion scanner. Strategy timeframe: **5-minute candles**. No order-placement code is used by this strategy.

## Closed-candle and history rules

- Only completed REST 5-minute candles are evaluated. The latest candle must have fully closed; forming WebSocket candlesticks are not subscribed to or evaluated.
- At least **250 closed bars per symbol** are required.
- Every interval in the latest 250-bar window must be exactly 300 seconds. A missing, duplicate, or off-grid interval rejects the symbol for that scan.
- REST requests ask for 320 bars to leave room for filtering and the 250-bar requirement.

## Indicators and trigger

Only these indicators are used:

- **ATR(14), Wilder smoothing**, seeded with the first 14-bar true-range SMA.
- **ATR%** = ATR(14) / close.
- **RVOL** = current completed-bar volume / mean volume of the preceding 20 completed bars (current bar excluded).
- **Donchian(20)** = highest high / lowest low of the preceding 20 bars (current bar excluded).
- **EMA(21)**.

Bar-derived values:

- **TR ratio** = current bar true range / previous bar ATR(14).
- **Close location** = (close - low) / (high - low).
- **Breakout clearance** = distance from the close beyond the prior Donchian boundary, divided by ATR(14); minimum 0.10 ATR.
- **Body ratio** = absolute candle body / full candle range; recorded for setup-quality ranking, not a mandatory rejection gate.
- **Quality score** (0–100) ranks candidates that already passed every mandatory gate. It is informational only and never rescues a failed setup or blocks a qualified one.

LONG requires all of: close at least 0.10 ATR above prior Donchian high, TR ratio >= 1.5, RVOL >= 1.15, close location >= 0.75, and close above EMA21.

SHORT requires all of: close at least 0.10 ATR below prior Donchian low, TR ratio >= 1.5, RVOL >= 1.15, close location <= 0.25, and close below EMA21.

## Safety gates

- Extension: absolute distance from close to EMA21 <= 2.5 × ATR(14).
- Cost: ATR% >= 3 × configured round-trip taker fee.
- Universe: scan only the fixed configured 120-symbol list. There is no 24-hour turnover filter and no dynamic symbol substitution.
- TP2 gross reward/risk must be >= 1.5. TP2 is 2R in this strategy.

Defaults are conservative starting values, not a substitute for account-specific fee settings or walk-forward testing:

- `MEX_MIN_RVOL=1.15`
- `MEX_MIN_BREAKOUT_CLEARANCE_ATR=0.10`
- `MEX_MIN_TR_RATIO=1.5`
- `MEX_ROUND_TRIP_TAKER_FEE_PCT=0.10` (percentage points; replace with the actual round-trip taker fee for the account)
- `MEX_LOG_ONLY=true`

## Entry, stop and targets

- Entry is the trigger candle close.
- LONG stop = the lower of trigger-bar low and entry minus 0.8 × ATR.
- SHORT stop = the higher of trigger-bar high and entry plus 0.8 × ATR.
- TP1 = 1R; TP2 = 2R.

## State and anti-spam

- Per-symbol state is durably stored in PostgreSQL on Render (SQLite locally).
- There is no timer-based per-symbol cooldown. The same symbol can qualify again on a later completed 5-minute candle; duplicate alerts for the same symbol and candle remain blocked.
- The current production path is direct breakout only. Retest entries require their own persisted setup lifecycle and are intentionally not emitted as duplicate alerts yet.
- A global database-serialized cap allows at most five MEX signals in any rolling hour across all symbols.
- The signal claim, state transition, event record, and Telegram outbox insert are committed atomically. Restarting the process cannot repeat a claimed signal.

## Scanner polling and validation

- After a symbol has a fresh, contiguous 250-bar history, the scanner waits until the next 5-minute candle has closed before downloading history again. This avoids repeatedly downloading the same 320 bars every 30 seconds.
- Data gaps, stale history, warmup, forming candles, and REST errors retain the configured retry cadence so recovery is not hidden.
- 15-minute context remains informational and is refreshed separately.

## Telegram delivery and validation

- `MEX_LOG_ONLY=false` enables Telegram alerts for qualified MEX candidates. Set it to `true` only when intentionally running a no-alert validation period.
- Telegram delivery requires both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. The durable outbox retries temporary delivery failures and records confirmed sends.
- Log-only candidates are journaled for analysis but do not consume live Telegram alert capacity. They are not replayed as alerts after log-only mode is disabled.
- Keep `SIGNAL_ONLY=true` and `ORDERS_ENABLED=false` at all times. Telegram alerts do not place orders.
- Review candidate/rejection counts, fee assumptions, and chronological out-of-sample results with realistic fees and slippage. The replay script is not a P&L backtest and does not establish expectancy.

## Optional squeeze-release path

Not implemented/enabled in this minimal base version. Do not add it until the base module has passed the one-week log-only review and walk-forward testing.

## Environment

```env
BOT_MODULE=MEX
MEX_ENABLED=true
MEX_MIN_RVOL=1.15
MEX_MIN_TR_RATIO=1.5
MEX_LOG_ONLY=false
MEX_ROUND_TRIP_TAKER_FEE_PCT=0.10
MEX_LOG_ONLY=true
SIGNAL_ONLY=true
ORDERS_ENABLED=false
PYTHON_VERSION=3.11.9
```

Use a dedicated PostgreSQL `DATABASE_URL` for Render production. Do not share the MEX state database with another bot.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q
```
