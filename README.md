# Delta Exchange India Signal Bot — MEX V2

Crypto-only, signal-only Momentum Explosion detector for Delta Exchange India.

## MEX V2 decision model

The detector uses four decision indicators:

- **TTM Squeeze:** Bollinger Bands (20, 2 standard deviations) inside Keltner Channels (EMA 20, 1.5 × ATR 20); a setup requires a squeeze-on previous completed candle followed by a squeeze release.
- **TTM momentum:** 20-period linear-regression momentum histogram; LONG requires positive and rising momentum, SHORT requires negative and falling momentum.
- **RVOL:** current completed 5-minute candle volume divided by the average volume of the preceding 20 completed candles; minimum 1.50.
- **ADX + DI:** ADX(14) minimum 19; +DI must exceed -DI for LONG and -DI must exceed +DI for SHORT.
- **Session VWAP:** UTC-midnight anchored VWAP; LONG close above VWAP, SHORT close below VWAP.

The TTM momentum histogram is part of the TTM Squeeze indicator family, not an additional standalone indicator. ATR is retained only for structural stop-loss sizing and is not a fifth signal filter.

## Entry qualification

A setup also requires a completed 5-minute candle close beyond the preceding 8-candle high/low and a directional candle-quality check. The setup is armed at the breakout candle's high/low. A fresh L1 ask (LONG) or bid (SHORT) must cross and hold the trigger through the existing quote-confirmation lifecycle. Extended entries and structural stops above 3 ATR are rejected.

Stop-loss is placed beyond the recent 5-candle swing with a 0.15 ATR buffer. Telegram target is 2R where valid. Signals only; no order execution is implemented or enabled.

## Data correctness

- Forming WebSocket candles cannot create MEX setups. `MEX_LIVE_ENABLED=false` disables the forming-candle detector path; the detector also rejects any 5-minute candle that has not closed. Live L1 quotes remain enabled for breakout trigger confirmation.
- RVOL uses actual completed-candle volume; no elapsed-time volume projection is used in the MEX V2 calculation.
- 5-minute REST history is expanded to 320 bars so the current UTC day's VWAP session is available (up to 288 bars).
- The scanner's recent 60-bar continuity check, quote freshness, setup persistence, deduplication, Telegram outbox and PostgreSQL production-state requirements are retained.

## Environment

- \`BOT_MODULE=MEX\`
- \`ADX_MIN=19\`
- \`MEX_MIN_RVOL=1.50\`
- \`SIGNAL_ONLY=true\`
- \`ORDERS_ENABLED=false\`

\`MEX_MIN_ATR_EXPANSION\` remains accepted for backwards-compatible configuration but is no longer a MEX signal gate.

## Run and deploy

Python 3.11.9. Install \`requirements.txt\`, configure \`.env\`, then run:

\`\`\`bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
\`\`\`

For Render production, configure a dedicated PostgreSQL \`DATABASE_URL\`; do not share the MEX state database with another bot.

## Tests

Run:

\`\`\`bash
pytest -q tests/test_mex_v2.py
\`\`\`
