# Phase 2: run Chronos-Bolt (zero-shot) through the same backtest.
# Two point-forecast variants: the median and the mean of the quantiles
# (see src/models/chronos_model.py for why both).
#
# Run from the repo root:
#     python -m src.run_chronos
#
# The first run downloads the pretrained weights (~200 MB) from Hugging Face.
#
# Writes to results/:
#     predictions_chronos.csv     one row per item / day / fold / variant
#     metrics_chronos.csv         MAE, RMSSE, bias overall and by segment
#     run_config_chronos.json     the exact config (incl. seed) used for this run

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.backtest import run_backtest, summarize
from src.data import load_config, load_dataset
from src.models.chronos_model import ChronosZeroShot


def main() -> None:
    config = load_config()
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    df = load_dataset(config)
    print(f"Data: {df['item_id'].nunique()} items, {df['date'].min().date()} to {df['date'].max().date()}")

    model_id = config["chronos"]["model_id"]
    preds = pd.concat([
        run_backtest(lambda: ChronosZeroShot(model_id, point="median"), df, config),
        run_backtest(lambda: ChronosZeroShot(model_id, point="quantile_mean"), df, config),
    ])
    metrics = summarize(preds)

    out = Path("results")
    out.mkdir(exist_ok=True)
    preds.to_csv(out / "predictions_chronos.csv", index=False)
    metrics.to_csv(out / "metrics_chronos.csv", index=False)
    (out / "run_config_chronos.json").write_text(json.dumps(config, indent=2))

    print(metrics[metrics["segmentation"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
