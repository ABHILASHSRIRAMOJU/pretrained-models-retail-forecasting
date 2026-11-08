# Model 4: Chronos-Bolt, zero-shot
#
# A pretrained time series model (Amazon, trained on a large mix of public
# series). "Zero-shot" means it is NOT trained on our data at all: for each
# item we hand it the past daily units and it forecasts the next 28 days.
#
# It is univariate: it only sees past `units`. No price, no calendar, no
# events. That is a real limitation for retail demand and is part of what this
# comparison is meant to show.
#
# Point forecast: Chronos-Bolt outputs quantiles (10%, 20%, ..., 90%), not one
# number. Two ways to turn that into a point forecast:
#   "median"         the 50% quantile (what the library returns as "mean").
#                    For items that sell 0 on most days the median is often 0,
#                    so this tends to under-forecast intermittent items.
#   "quantile_mean"  the average of the 9 quantiles, a simple approximation of
#                    the mean of the forecast distribution.
# The mean is the better target for RMSSE (a squared error), so both are run.

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from src.backtest import Forecaster

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Load the pretrained weights once and share them between folds and variants
_PIPELINES: dict[str, BaseChronosPipeline] = {}


def _load(model_id: str) -> BaseChronosPipeline:
    if model_id not in _PIPELINES:
        _PIPELINES[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32)
    return _PIPELINES[model_id]


class ChronosZeroShot(Forecaster):
    def __init__(self, model_id: str = "amazon/chronos-bolt-small", point: str = "quantile_mean",
                 batch_size: int = 64):
        if point not in ("median", "quantile_mean"):
            raise ValueError(f"unknown point forecast: {point}")
        self.model_id = model_id
        self.point = point
        self.batch_size = batch_size
        self.name = f"chronos_bolt_{point}"

    def fit(self, train_df: pd.DataFrame) -> "ChronosZeroShot":
        # Nothing to train. Just keep each item's past units, oldest first.
        hist = train_df.sort_values("date")
        self.cutoff_ = hist["date"].max()
        self.series_ = {item: d["units"].to_numpy(np.float32) for item, d in hist.groupby("item_id")}
        return self

    def predict(self, horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
        pipeline = _load(self.model_id)
        items = list(self.series_)
        forecasts = {}
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            quantiles, median = pipeline.predict_quantiles(
                [torch.tensor(self.series_[i]) for i in batch],
                prediction_length=horizon, quantile_levels=QUANTILES)
            point = median if self.point == "median" else quantiles.mean(dim=-1)
            forecasts.update(zip(batch, point.numpy()))

        # Row for (item, date) gets the forecast for step = days after the cutoff
        out = future_df[["item_id", "date"]].copy()
        step = (out["date"] - self.cutoff_).dt.days - 1
        out["pred"] = [forecasts[i][s] for i, s in zip(out["item_id"], step)]
        # Demand can't be negative
        out["pred"] = out["pred"].clip(lower=0)
        return out
