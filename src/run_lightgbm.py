# Phase 2: run LightGBM through the same backtest as the baselines.
#
# Run from the repo root:
#     python -m src.run_lightgbm
#
# Writes to results/:
#     predictions_lightgbm.csv         one row per item / day / fold
#     metrics_lightgbm.csv             MAE, RMSSE, bias overall and by segment
#     feature_importance_lightgbm.csv  total gain per feature, for each fold
#     run_config_lightgbm.json         the exact config (incl. seed) used for this run

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest, summarize
from src.data import load_config, load_dataset
from src.models.lightgbm_model import LightGBMForecaster


def main() -> None:
    config = load_config()
    np.random.seed(config["seed"])
    df = load_dataset(config)
    print(f"Data: {df['item_id'].nunique()} items, {df['date'].min().date()} to {df['date'].max().date()}")

    # Keep a handle on each fold's model so its feature importance can be saved
    models = []

    def make_model() -> LightGBMForecaster:
        lgb_cfg = config["lightgbm"]
        model = LightGBMForecaster(lgb_cfg["params"], lgb_cfg["num_boost_round"], config["seed"])
        models.append(model)
        return model

    preds = run_backtest(make_model, df, config)
    metrics = summarize(preds)
    importance = pd.concat(
        [m.feature_importance().rename("gain").rename_axis("feature").reset_index().assign(fold=k)
         for k, m in enumerate(models)],
        ignore_index=True,
    )

    out = Path("results")
    out.mkdir(exist_ok=True)
    preds.to_csv(out / "predictions_lightgbm.csv", index=False)
    metrics.to_csv(out / "metrics_lightgbm.csv", index=False)
    importance.to_csv(out / "feature_importance_lightgbm.csv", index=False)
    (out / "run_config_lightgbm.json").write_text(json.dumps(config, indent=2))

    print(metrics[metrics["segmentation"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
