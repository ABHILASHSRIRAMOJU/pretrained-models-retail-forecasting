# Model 1: seasonal naive
# The forecast for each day is what the item sold on the same weekday in the
# last week of training. E.g. every future Monday = last Monday's sales.
# This is the simplest reasonable baseline; anything else should beat it.

from __future__ import annotations

import pandas as pd

from src.backtest import Forecaster


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaive":
        # Keep only the last 7 days and index them by (item, weekday)
        cutoff = train_df["date"].max()
        last_week = train_df[train_df["date"] > cutoff - pd.Timedelta(days=7)]
        self.by_weekday_ = (
            last_week.assign(dow=last_week["date"].dt.dayofweek)
            .set_index(["item_id", "dow"])["units"]
        )
        # Backup in case an item is missing a weekday in that last week:
        # use its average over the last 28 days instead
        self.fallback_ = train_df.groupby("item_id")["units"].apply(lambda s: s.tail(28).mean())
        return self

    def predict(self, horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
        # Look up each future row by its item and weekday
        out = future_df[["item_id", "date"]].copy()
        keys = pd.MultiIndex.from_arrays([out["item_id"], out["date"].dt.dayofweek])
        out["pred"] = self.by_weekday_.reindex(keys).to_numpy()
        out["pred"] = out["pred"].fillna(out["item_id"].map(self.fallback_)).fillna(0.0)
        return out
