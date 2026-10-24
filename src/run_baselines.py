# Phase 1: run both baselines through the backtest and save the results.
#
# Run from the repo root:
#     python -m src.run_baselines
#
# Writes to results/:
#     predictions_baselines.csv   one row per item / day / fold / model
#     metrics_baselines.csv       MAE, RMSSE, bias overall and by segment
#     run_config_baselines.json   the exact config (incl. seed) used for this run

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest, summarize
from src.data import load_config, load_dataset
from src.models.price_last_period import PriceLastPeriod
from src.models.seasonal_naive import SeasonalNaive


def main() -> None:
    config = load_config()
    np.random.seed(config["seed"])
    df = load_dataset(config)
    print(f"Data: {df['item_id'].nunique()} items, {df['date'].min().date()} to {df['date'].max().date()}")

    # Same data, same folds for both models
    preds = pd.concat([
        run_backtest(SeasonalNaive, df, config),
        run_backtest(PriceLastPeriod, df, config),
    ])
    metrics = summarize(preds)

    out = Path("results")
    out.mkdir(exist_ok=True)
    preds.to_csv(out / "predictions_baselines.csv", index=False)
    metrics.to_csv(out / "metrics_baselines.csv", index=False)
    (out / "run_config_baselines.json").write_text(json.dumps(config, indent=2))

    print(metrics[metrics["segmentation"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
