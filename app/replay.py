"""Offline replay of the minimal MEX detector on completed 5m candles.

This is a detector replay, not a P&L backtest. Historical quote turnover must
be supplied per candle to exercise the liquidity gate without look-ahead.
"""
import math
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .momentum import evaluate_mex


REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS = ("quote_volume_24h_usd",)
SUPPORTED_STRATEGIES = ("MEX",)
MAX_HISTORY_BARS = 250
DEFAULT_REPLAY_THRESHOLDS = {
    "mex_min_rvol": 2.0,
    "mex_min_tr_ratio": 2.0,
    "mex_min_quote_volume_usd": 100000.0,
    "mex_round_trip_taker_fee_pct": 0.10,
}


def validate_replay_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and strictly validate multi-symbol completed 5m OHLCV input."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Replay input must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Replay input is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Replay input contains no candle rows")
    if frame["symbol"].isna().any():
        raise ValueError("Replay input contains a missing symbol")
    optional = [name for name in OPTIONAL_COLUMNS if name in frame.columns]
    data = frame.loc[:, [*REQUIRED_COLUMNS, *optional]].copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    if data["symbol"].eq("").any():
        raise ValueError("Replay input contains an empty symbol")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Replay input contains an invalid timestamp")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "quote_volume_24h_usd" in data.columns:
        data["quote_volume_24h_usd"] = pd.to_numeric(
            data["quote_volume_24h_usd"], errors="coerce"
        )
    numeric = data[["open", "high", "low", "close", "volume"]]
    if numeric.isna().any().any():
        raise ValueError("Replay input contains missing OHLCV values")
    if not numeric.apply(
        lambda column: column.map(lambda value: math.isfinite(float(value)))
    ).all().all():
        raise ValueError("Replay input contains non-finite OHLCV values")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Replay input contains non-positive prices")
    if (data["volume"] < 0).any():
        raise ValueError("Replay input contains negative volume")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Replay input contains invalid high values")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Replay input contains invalid low values")
    if ((data["timestamp"].dt.minute % 5) != 0).any() or (
        (data["timestamp"].dt.second != 0) | (data["timestamp"].dt.microsecond != 0)
    ).any():
        raise ValueError("Replay timestamps must align to 5-minute candle opens")
    data = data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if data.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Replay input contains duplicate symbol/timestamp rows")
    return data


def _evaluate(strategy: str, frame: pd.DataFrame, thresholds: dict):
    if strategy != "MEX":
        raise ValueError(f"Unsupported replay strategy: {strategy}")
    latest = frame.iloc[-1]
    diagnostics = {}
    candle_close_time = latest["timestamp"].to_pydatetime() + timedelta(minutes=5, seconds=1)
    result = evaluate_mex(
        frame,
        quote_volume_24h_usd=latest.get("quote_volume_24h_usd"),
        min_quote_volume_usd=thresholds["mex_min_quote_volume_usd"],
        round_trip_taker_fee_pct=thresholds["mex_round_trip_taker_fee_pct"],
        min_rvol=thresholds["mex_min_rvol"],
        min_tr_ratio=thresholds["mex_min_tr_ratio"],
        now=candle_close_time,
        diagnostics=diagnostics,
    )
    return result, diagnostics


def load_strategy_universes(config_dir="config") -> dict[str, set[str]]:
    """Load and validate the same disjoint symbol universes used by live config."""
    root = Path(config_dir)

    def read_symbols(filename, expected_count=None):
        path = root / filename
        if not path.is_file():
            raise ValueError(f"Configured symbol universe file does not exist: {path}")
        values = []
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            symbol = line.split("#", 1)[0].strip().upper()
            if not symbol:
                continue
            if symbol in seen:
                raise ValueError(f"Duplicate symbol {symbol} in {path}")
            seen.add(symbol)
            values.append(symbol)
        if not values:
            raise ValueError(f"Configured symbol universe is empty: {path}")
        if expected_count is not None and len(values) != expected_count:
            raise ValueError(f"{path} must contain exactly {expected_count} unique symbols")
        return set(values)

    mex_core = read_symbols("momentum_symbols.txt", expected_count=60)
    tpr_rb = read_symbols("tpr_rb_symbols.txt", expected_count=60)
    if mex_core & tpr_rb:
        raise ValueError("MEX symbol seed lists overlap")
    return {"MEX": mex_core | tpr_rb}


def replay_dataframe(
    frame: pd.DataFrame,
    *,
    strategies=None,
    include_rejections: bool = False,
    threshold_overrides: dict | None = None,
    strategy_symbols: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Replay each eligible completed candle with a trailing 250-bar window."""
    data = validate_replay_data(frame)
    selected = tuple(str(value).strip().upper() for value in (
        SUPPORTED_STRATEGIES if strategies is None else strategies
    ))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select one or more unique replay strategies")
    unknown = sorted(set(selected) - set(SUPPORTED_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported replay strategies: {unknown}")

    thresholds = dict(DEFAULT_REPLAY_THRESHOLDS)
    if threshold_overrides is not None and not isinstance(threshold_overrides, dict):
        raise ValueError("Replay threshold overrides must be a JSON object")
    if threshold_overrides:
        unknown_thresholds = sorted(set(threshold_overrides) - set(thresholds))
        if unknown_thresholds:
            raise ValueError(f"Unsupported replay threshold keys: {unknown_thresholds}")
        thresholds.update({key: float(value) for key, value in threshold_overrides.items()})
    if not all(math.isfinite(value) and value > 0 for value in thresholds.values()):
        raise ValueError("Replay thresholds must be finite and positive")

    input_symbols = set(data["symbol"].unique())
    if strategy_symbols is None:
        eligible_by_strategy = {name: set(input_symbols) for name in selected}
    else:
        if not isinstance(strategy_symbols, dict) or set(selected) - set(strategy_symbols):
            raise ValueError("strategy_symbols must map every selected strategy to a symbol list")
        eligible_by_strategy = {
            name: {str(symbol).strip().upper() for symbol in strategy_symbols[name]}
            for name in selected
        }
    metrics = {
        name: {
            "eligible_symbols": len(input_symbols & eligible_by_strategy[name]),
            "evaluations": 0, "setups": 0, "errors": 0, "rejections": {},
        }
        for name in selected
    }
    events = []
    gap_count = 0
    segments = 0
    symbols = sorted(data["symbol"].unique().tolist())

    for symbol, group in data.groupby("symbol", sort=True):
        group = group.reset_index(drop=True)
        boundaries = [
            index for index in range(1, len(group))
            if (group.loc[index, "timestamp"] - group.loc[index - 1, "timestamp"]).total_seconds() != 300
        ]
        starts, ends = [0, *boundaries], [*boundaries, len(group)]
        gap_count += len(boundaries)
        for start, end in zip(starts, ends):
            segment = group.iloc[start:end].reset_index(drop=True)
            if segment.empty:
                continue
            segments += 1
            for end_index in range(MAX_HISTORY_BARS - 1, len(segment)):
                window = segment.iloc[end_index - MAX_HISTORY_BARS + 1:end_index + 1]
                candle = window.iloc[-1]
                ohlcv = window.drop(columns=["symbol"]).reset_index(drop=True)
                for strategy in selected:
                    if symbol not in eligible_by_strategy[strategy]:
                        continue
                    stat = metrics[strategy]
                    stat["evaluations"] += 1
                    try:
                        result, diagnostics = _evaluate(strategy, ohlcv, thresholds)
                    except Exception as exc:
                        stat["errors"] += 1
                        reason = f"evaluation_error:{type(exc).__name__}"
                        stat["rejections"][reason] = stat["rejections"].get(reason, 0) + 1
                        if include_rejections:
                            events.append({
                                "symbol": symbol, "timestamp": candle["timestamp"].isoformat(),
                                "strategy": strategy, "decision": "error", "side": "",
                                "entry_reference": None, "trigger_price": None,
                                "stop_loss": None, "take_profit_1": None, "take_profit_2": None,
                                "rvol": None, "tr_ratio": None, "risk_reward": None,
                                "quote_volume_24h_usd": None, "atr": None,
                                "reason": reason, "diagnostics": str(exc),
                            })
                        continue
                    if result:
                        stat["setups"] += 1
                        events.append({
                            "symbol": symbol, "timestamp": candle["timestamp"].isoformat(),
                            "strategy": strategy, "decision": "candidate",
                            "side": result.get("side", ""),
                            "entry_reference": result.get("entry_reference"),
                            "trigger_price": result.get("trigger_price"),
                            "stop_loss": result.get("stop_loss"),
                            "take_profit_1": result.get("take_profit_1"),
                            "take_profit_2": result.get("take_profit_2"),
                            "rvol": result.get("rvol"), "tr_ratio": result.get("tr_ratio"),
                            "risk_reward": result.get("risk_reward"),
                            "quote_volume_24h_usd": result.get("quote_volume_24h_usd"),
                            "atr": result.get("atr"), "reason": result.get("reason", ""),
                            "diagnostics": str(result.get("indicators", {})),
                        })
                    else:
                        reason = str(diagnostics.get("reason", "no_setup"))
                        stat["rejections"][reason] = stat["rejections"].get(reason, 0) + 1
                        if include_rejections:
                            events.append({
                                "symbol": symbol, "timestamp": candle["timestamp"].isoformat(),
                                "strategy": strategy, "decision": "rejected", "side": "",
                                "entry_reference": None, "trigger_price": None,
                                "stop_loss": None, "take_profit_1": None, "take_profit_2": None,
                                "rvol": None, "tr_ratio": None, "risk_reward": None,
                                "quote_volume_24h_usd": None, "atr": None,
                                "reason": reason, "diagnostics": str(diagnostics),
                            })

    for stat in metrics.values():
        stat["setup_rate"] = (
            round(stat["setups"] / stat["evaluations"], 6) if stat["evaluations"] else 0.0
        )
        stat["rejections"] = dict(sorted(stat["rejections"].items()))
    columns = [
        "symbol", "timestamp", "strategy", "decision", "side",
        "entry_reference", "trigger_price", "stop_loss", "take_profit_1", "take_profit_2",
        "rvol", "tr_ratio", "risk_reward", "quote_volume_24h_usd", "atr",
        "reason", "diagnostics",
    ]
    return pd.DataFrame(events, columns=columns), {
        "mode": "detector_replay_only",
        "input_candles": int(len(data)),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "contiguous_segments": segments,
        "gap_boundaries": gap_count,
        "strategies": metrics,
        "thresholds": thresholds,
        "limitations": [
            "Detector replay only: no forward exits, P&L, drawdown or expectancy are calculated.",
            "Historical quote_volume_24h_usd is required per candle to test the liquidity gate without look-ahead.",
            "No spread, slippage, funding, Telegram latency or execution fills are simulated.",
            "The replay does not apply the live per-symbol cooldown or global hourly signal cap.",
        ],
    }
