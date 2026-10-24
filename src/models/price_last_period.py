# Model 2: price + last-period regression (one small model per item)
#
#     units_t = a + b * log(price_t) + c * last_period_mean_t
#
# last_period_mean_t = average units over days t-55 .. t-28.
#
# Why shift by 28 days? The forecast horizon is 28 days, so for any day we
# forecast, the window t-55 .. t-28 is always in the past. That means the model
# can predict all 28 days in one go, without feeding its own predictions back
# in, and without peeking at the test period.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.backtest import Forecaster

SHIFT = 28   # has to be >= the horizon, otherwise the feature would leak
WINDOW = 28


# Build the two input features: log price and the previous-period average
def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id", "date"]).copy()
    df["log_price"] = np.log(df["sell_price"])
    df["last_period_mean"] = df.groupby("item_id")["units"].transform(
        lambda s: s.shift(SHIFT).rolling(WINDOW, min_periods=7).mean()
    )
    return df


class PriceLastPeriod(Forecaster):
    name = "price_last_period"
    features = ["log_price", "last_period_mean"]

    def __init__(self, alpha: float = 1.0):
        # Ridge instead of plain linear regression: the small penalty keeps the
        # price coefficient sensible for items whose price hardly ever changes
        self.alpha = alpha

    def fit(self, train_df: pd.DataFrame) -> "PriceLastPeriod":
        # Keep the history; it's needed later to compute features for the future rows
        self.history_ = train_df[["item_id", "date", "units", "sell_price"]]
        feats = _add_features(self.history_).dropna(subset=self.features)
        self.models_ = {
            item: Ridge(alpha=self.alpha).fit(d[self.features], d["units"])
            for item, d in feats.groupby("item_id")
        }
        return self

    def predict(self, horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
        if horizon > SHIFT:
            raise ValueError(f"horizon {horizon} > feature shift {SHIFT}: would leak")

        # Stack history + future rows so the 28-day-shifted average can be
        # computed for the future dates (it only uses history values)
        future = future_df[["item_id", "date", "sell_price"]].assign(units=np.nan)
        both = _add_features(pd.concat([self.history_, future], ignore_index=True))
        fut = both[both["units"].isna()].copy()

        fut["pred"] = 0.0
        for item, d in fut.groupby("item_id"):
            model = self.models_.get(item)
            if model is None or d[self.features].isna().any(axis=None):
                # Not enough history to fit/apply the model -> last 28-day average
                fallback = self.history_.loc[self.history_["item_id"] == item, "units"].tail(28).mean()
                fut.loc[d.index, "pred"] = fallback
            else:
                fut.loc[d.index, "pred"] = model.predict(d[self.features])

        # Demand can't be negative
        fut["pred"] = fut["pred"].clip(lower=0).fillna(0.0)
        return fut[["item_id", "date", "pred"]]
