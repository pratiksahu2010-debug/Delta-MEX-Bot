# One-day shadow ML training workflow

This workflow is deliberately research-only. It does not change the live MEX/ORB detector, Telegram alerts, thresholds, or order settings.

## Daily operating cycle

1. Keep the scanner running normally for at least 24 hours so confirmed setups can receive completed 5-minute OHLC outcome labels.
2. Allow the outcome resolver to finish each setup's configured observation horizon. Unresolved, expired, and same-candle ambiguous outcomes are not used as clean training labels.
3. Trigger the MEX training cycle once daily:
   `POST /admin/ml-training/MEX`
4. Optionally train ORB separately:
   `POST /admin/ml-training/ORB`
5. Inspect `GET /admin/ml-training` or `GET /status` for sample counts, chronological split boundaries, feature coverage, LONG/SHORT test metrics, expectancy, and quality-gate results.

All admin requests require the `X-Journal-Key` header. Do not put the key in a URL or commit it to Git.

## Suggested daily scheduler call

Use a Render Cron Job or an external scheduler to issue the authenticated POST once per day. Keep the web service itself free of an always-running training loop; model fitting runs in a worker thread and is protected against overlapping training cycles.

Example:

```sh
curl --fail-with-body -X POST \
  -H "X-Journal-Key: $JOURNAL_ADMIN_KEY" \
  "$BOT_URL/admin/ml-training/MEX"
```

## Data and model safeguards

- Training features are point-in-time snapshots. Market outcomes and human feedback remain separate.
- Missing feature telemetry is represented explicitly by coverage ratio and missing-count features; zero is no longer the only indication of missingness. The feature set includes the active minimal MEX detector's TR ratio, RVOL, close location, Donchian breakout clearance, EMA21 extension, and quality score.
- Only completed, unambiguous OHLC-path outcomes with a finite net return are eligible labels.
- Training, validation, and test data are chronological with a label-horizon embargo.
- The minimum validation and test sets are 20 samples each; the test selection must contain at least 10 observations.
- The candidate is saved only when out-of-sample selected expectancy is positive, at least as good as the full-test baseline, coverage is nontrivial, and TP1 Brier score is no worse than the training-prevalence baseline.
- Failed gates produce `quality_rejected`; insufficient chronological data produces `insufficient_data`. Neither status replaces an existing model.
- A passing candidate is saved in shadow mode only. There is no automatic promotion and no effect on live signal decisions.

## Important interpretation

One day is a collection cadence, not a promise that the data will be sufficient. If the bot produces too few confirmed setups or the outcome horizon has not elapsed, wait for more completed observations rather than weakening sample, leakage, or quality gates. A single daily cycle is not evidence of profitability; continue collecting out-of-sample shadow results before considering any manual promotion.
