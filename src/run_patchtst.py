# Phase 2: train and run PatchTST through the same backtest.
#
# Run from the repo root:
#     python -m src.run_patchtst
#
# CPU training, roughly 5-10 minutes per fold.
#
# Writes to results/:
#     predictions_patchtst.csv     one row per item / day / fold
#     metrics_patchtst.csv         MAE, RMSSE, bias overall and by segment
#     run_config_patchtst.json     the exact config (incl. seed) used for this run

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.backtest import run_backtest, summarize
from src.data import load_config, load_dataset
from src.models.patchtst_model import PatchTSTForecaster


def main() -> None:
    config = load_config()
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    df = load_dataset(config)
    print(f"Data: {df['item_id'].nunique()} items, {df['date'].min().date()} to {df['date'].max().date()}")

    def make_model() -> PatchTSTForecaster:
        return PatchTSTForecaster(**config["patchtst"], seed=config["seed"])

    preds = run_backtest(make_model, df, config)
    metrics = summarize(preds)

    out = Path("results")
    out.mkdir(exist_ok=True)
    preds.to_csv(out / "predictions_patchtst.csv", index=False)
    metrics.to_csv(out / "metrics_patchtst.csv", index=False)
    (out / "run_config_patchtst.json").write_text(json.dumps(config, indent=2))

    print(metrics[metrics["segmentation"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
