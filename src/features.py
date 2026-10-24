# features.py
# Feature table for LightGBM. One row per item per day.
#
# Every feature built from `units` is shifted back by at least SHIFT = 28 days
# (the forecast horizon). So for any day in the 28-day test window, the
# features only use sales from before the cutoff. That lets the model predict
# all 28 days at once ("direct" forecasting), without feeding its own
# predictions back in and without peeking at the test period.
#
# Price and calendar features are not shifted: they are known in advance
# (same assumption as the rest of the project: future prices are planned).

from __future__ import annotations

import numpy as np
import pandas as pd

SHIFT = 28                         # must be >= the horizon, otherwise units would leak
LAGS = [28, 35, 42, 49, 56]        # same weekday, 4 to 8 weeks back
ROLL_WINDOWS = [7, 28, 56]         # rolling means of the shifted series

SALES_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["roll_std_28", "roll_zero_share_28"]
)
CALENDAR_FEATURES = ["dow", "day_of_month", "month", "is_event", "event_type", "snap"]
PRICE_FEATURES = ["sell_price", "price_ratio_28", "price_ratio_max", "price_change_7"]
ID_FEATURES = ["item_id", "cat_id"]

FEATURES = ID_FEATURES + SALES_FEATURES + CALENDAR_FEATURES + PRICE_FEATURES
CATEGORICAL = ["item_id", "cat_id", "event_type"]


# Add all features to a long table (item_id, date, units, sell_price, ...).
# Rows whose `units` is NaN (the future) still get features, because every
# units-based feature only looks at least SHIFT days back.
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    g_units = df.groupby("item_id")["units"]

    # --- sales history (all shifted by >= 28 days) ---
    for k in LAGS:
        df[f"lag_{k}"] = g_units.shift(k)
    shifted = g_units.shift(SHIFT)
    g_shifted = shifted.groupby(df["item_id"])
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = g_shifted.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["roll_std_28"] = g_shifted.transform(lambda s: s.rolling(28, min_periods=2).std())
    df["roll_zero_share_28"] = (shifted == 0).astype(float).where(shifted.notna()) \
        .groupby(df["item_id"]).transform(lambda s: s.rolling(28, min_periods=1).mean())

    # --- calendar ---
    df["dow"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["event_type"] = df["event_type_1"].fillna("none")

    # --- price ---
    g_price = df.groupby("item_id")["sell_price"]
    # Price vs the item's average price over the last 28 days (< 1 = cheaper than usual)
    df["price_ratio_28"] = df["sell_price"] / g_price.transform(lambda s: s.rolling(28, min_periods=1).mean())
    # Price vs the highest price seen so far for the item (a rough "regular price")
    df["price_ratio_max"] = df["sell_price"] / g_price.cummax()
    # Relative price change vs one week earlier
    df["price_change_7"] = df["sell_price"] / g_price.shift(7) - 1.0
    df["price_change_7"] = df["price_change_7"].replace([np.inf, -np.inf], np.nan)

    return df
