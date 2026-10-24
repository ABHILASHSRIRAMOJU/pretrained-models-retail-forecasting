# Model 3: LightGBM, one global model across all items
#
# Inputs (see src/features.py): sales lags and rolling stats shifted by >= 28
# days, calendar/event features, and price features (price, price vs the item's
# recent average, price vs its highest price so far, weekly price change).
#
# Objective: Tweedie. Daily retail sales are non-negative counts with lots of
# zeros, and Tweedie loss is built for exactly that shape (it was the standard
# choice in the M5 competition).

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest import Forecaster
from src.features import CATEGORICAL, FEATURES, SHIFT, build_features

HISTORY_COLS = ["item_id", "cat_id", "date", "units", "sell_price",
                "is_event", "event_type_1", "snap"]


# Fix the category levels so training and prediction encode them the same way
def _as_categories(df: pd.DataFrame, levels: dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = pd.Categorical(df[col], categories=levels[col])
    return df


class LightGBMForecaster(Forecaster):
    name = "lightgbm"

    def __init__(self, params: dict | None = None, num_boost_round: int = 500, seed: int = 42):
        self.params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            **(params or {}),
            "seed": seed,
            "deterministic": True,
        }
        self.num_boost_round = num_boost_round

    def fit(self, train_df: pd.DataFrame) -> "LightGBMForecaster":
        # Keep the history; it's needed later to compute features for the future rows
        self.history_ = train_df[HISTORY_COLS]
        feats = build_features(self.history_)
        # Skip each item's first SHIFT days: they have no sales history yet
        feats = feats.dropna(subset=[f"lag_{SHIFT}"])

        self.levels_ = {col: sorted(feats[col].dropna().unique()) for col in CATEGORICAL}
        feats = _as_categories(feats, self.levels_)
        data = lgb.Dataset(feats[FEATURES], label=feats["units"], categorical_feature=CATEGORICAL)
        self.model_ = lgb.train(self.params, data, num_boost_round=self.num_boost_round)
        return self

    def predict(self, horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
        if horizon > SHIFT:
            raise ValueError(f"horizon {horizon} > feature shift {SHIFT}: would leak")

        # Stack history + future rows so lag/rolling features can be computed
        # for the future dates (they only reach back into history)
        future = future_df[[c for c in HISTORY_COLS if c != "units"]].assign(units=np.nan)
        both = build_features(pd.concat([self.history_, future], ignore_index=True))
        fut = _as_categories(both[both["units"].isna()], self.levels_)

        # Demand can't be negative (Tweedie already guarantees this; clip to be safe)
        fut["pred"] = np.clip(self.model_.predict(fut[FEATURES]), 0, None)
        return fut[["item_id", "date", "pred"]]

    # Which features the trees relied on most (total gain), for the write-up
    def feature_importance(self) -> pd.Series:
        gain = self.model_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=FEATURES).sort_values(ascending=False)
