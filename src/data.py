# data.py
# Loads the raw M5 files, picks a smaller subset of items, and reshapes the
# sales from wide (one column per day) to long (one row per item per day).
#
# Final columns:
#   item_id, cat_id, store_id, date, units, sell_price,
#   wday, month, year, event_name_1, event_type_1, snap, is_event

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Read all experiment settings from config.yaml
def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Pick one store and sample a fixed number of items from each category.
# Items that sold on very few days are removed first, otherwise the sample
# ends up full of series that are almost entirely zeros.
def select_items(sales_wide: pd.DataFrame, store_id: str, items_per_category: int,
                 min_nonzero_days: int, seed: int) -> pd.DataFrame:
    store = sales_wide[sales_wide["store_id"] == store_id]
    day_cols = [c for c in store.columns if c.startswith("d_")]
    nonzero_days = (store[day_cols] > 0).sum(axis=1)
    store = store[nonzero_days >= min_nonzero_days]

    # Fixed seed so the same items are picked every run
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in store.groupby("cat_id"):
        n = min(items_per_category, len(group))
        picks.append(group.iloc[rng.choice(len(group), size=n, replace=False)])
    return pd.concat(picks).sort_values("item_id").reset_index(drop=True)


# The sales file uses day ids (d_1, d_2, ...) and the calendar maps them to dates.
# The copy of the calendar I downloaded doesn't include the `d` column, so I
# rebuild it here. This works because the calendar has one row per day starting
# at d_1 = 2011-01-29. I check for gaps first so a broken file fails loudly.
def add_day_index(calendar: pd.DataFrame) -> pd.DataFrame:
    if "d" in calendar.columns:
        return calendar
    calendar = calendar.sort_values("date").reset_index(drop=True)
    gaps = pd.to_datetime(calendar["date"]).diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError("calendar dates are not consecutive; cannot rebuild `d`")
    return calendar.assign(d=[f"d_{i + 1}" for i in range(len(calendar))])


# Wide -> long, then attach calendar features and the weekly price
def to_long(sales_wide: pd.DataFrame, calendar: pd.DataFrame,
            prices: pd.DataFrame, store_id: str) -> pd.DataFrame:
    id_cols = ["item_id", "cat_id", "store_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long = sales_wide[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="units"
    )

    # SNAP days are state-specific, so keep only the column for this store's state
    cal = calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
                    "event_name_1", "event_type_1", f"snap_{store_id[:2]}"]]
    cal = cal.rename(columns={f"snap_{store_id[:2]}": "snap"})
    long = long.merge(cal, on="d", how="left")

    # Prices are weekly, so join on the Walmart week id
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # No price = the item wasn't on the shelf yet. I drop these rows instead
    # of treating them as real days with zero demand.
    long = long.dropna(subset=["sell_price"])

    long["date"] = pd.to_datetime(long["date"])
    long["is_event"] = long["event_name_1"].notna().astype(int)
    long = long.drop(columns=["d", "wm_yr_wk"])
    return long.sort_values(["item_id", "date"]).reset_index(drop=True)


# Full pipeline: raw CSVs -> subset in long format, saved to data/processed/
def build_dataset(config: dict) -> pd.DataFrame:
    cfg = config["data"]
    raw = Path(cfg["raw_dir"])
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    calendar = add_day_index(pd.read_csv(raw / "calendar.csv"))
    prices = pd.read_csv(raw / "sell_prices.csv")

    subset = select_items(sales, cfg["store_id"], cfg["items_per_category"],
                          cfg["min_nonzero_days"], config["seed"])
    prices = prices[prices["store_id"] == cfg["store_id"]]
    long = to_long(subset, calendar, prices, cfg["store_id"])

    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "m5_subset_long.csv", index=False)
    return long


# Use the saved subset if it exists, otherwise build it from the raw files
def load_dataset(config: dict) -> pd.DataFrame:
    path = Path(config["data"]["processed_dir"]) / "m5_subset_long.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return build_dataset(config)
