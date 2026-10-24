# backtest.py
# The evaluation setup that every model goes through:
#   - rolling-origin folds (always split by time, never randomly)
#   - one shared fit / predict interface for all models
#   - metrics: MAE, RMSSE, bias
#   - segment labels so results can be broken down by item type
#
# Because every model runs through the same code, the results are directly
# comparable: same folds, same rows, same metrics.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

# Every model subclasses this and implements two methods:
#
#   fit(train_df)
#       Learn from history only, i.e. rows up to and including the cutoff date.
#
#   predict(horizon, future_df)
#       Forecast the `horizon` days after the cutoff. `future_df` holds the rows
#       to forecast (item_id, date) plus anything known ahead of time (price,
#       calendar, events). It never contains `units`.
#       Must return a DataFrame with item_id, date, pred.
#
# Assumption: future prices are known in advance (planned prices), same as in M5.
class Forecaster:
    name: str = "base"

    def fit(self, train_df: pd.DataFrame) -> "Forecaster":
        raise NotImplementedError

    def predict(self, horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    fold: int
    cutoff: pd.Timestamp   # last date the model is allowed to see
    test_end: pd.Timestamp


# The last fold tests the final `horizon` days of the data. Each earlier fold
# moves the cutoff back by `step` days. Returned oldest fold first.
def make_folds(last_date: pd.Timestamp, horizon: int, n_folds: int, step: int) -> list[Fold]:
    folds = []
    for k in range(n_folds):
        cutoff = last_date - pd.Timedelta(days=horizon + step * (n_folds - 1 - k))
        folds.append(Fold(k, cutoff, cutoff + pd.Timedelta(days=horizon)))
    return folds


# train = everything up to the cutoff, test = the next `horizon` days
def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= fold.cutoff]
    test = df[(df["date"] > fold.cutoff) & (df["date"] <= fold.test_end)]
    return train, test


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

# Label each item by volume and price behaviour. These are computed from the
# training data only; otherwise the test period would leak into the labels.
#   intermittent   = more than X% of training days had zero sales
#   price_volatile = price coefficient of variation above a threshold
def item_segments(train: pd.DataFrame, zero_share: float, price_cv: float) -> pd.DataFrame:
    g = train.groupby("item_id")
    seg = pd.DataFrame({
        "zero_share": g["units"].apply(lambda s: (s == 0).mean()),
        "price_cv": g["sell_price"].std() / g["sell_price"].mean(),
        "median_price": g["sell_price"].median(),
    })
    seg["volume_seg"] = np.where(seg["zero_share"] > zero_share, "intermittent", "high_volume")
    seg["price_seg"] = np.where(seg["price_cv"].fillna(0) > price_cv, "price_volatile", "price_stable")
    return seg.reset_index()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# I'm not using MAPE / sMAPE because many items have zero-sales days, which
# makes percentage errors blow up or become undefined.

# Mean absolute error, in units
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


# Relative bias = sum(forecast - actual) / sum(actual)
# Positive means over-forecasting, negative means under-forecasting.
def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    total = y.sum()
    return float((yhat - y).sum() / total) if total > 0 else float("nan")


# RMSSE denominator for each item: the mean squared error of a one-step naive
# forecast (yesterday = today) over the training period. Same as M5.
def naive_scale(train: pd.DataFrame) -> pd.Series:
    def _scale(s: pd.Series) -> float:
        return float(np.mean(np.diff(s.to_numpy(float)) ** 2))
    return train.sort_values("date").groupby("item_id")["units"].apply(_scale)


# RMSSE is computed per item and then averaged.
# Below 1 = better than the naive forecast; above 1 = worse.
# Items that never changed in training (scale = 0) are skipped.
def rmsse(preds: pd.DataFrame) -> float:
    p = preds[preds["scale"] > 0]
    per_item = p.groupby("item_id").apply(
        lambda d: np.sqrt(np.mean((d["units"] - d["pred"]) ** 2) / d["scale"].iloc[0]),
        include_groups=False,
    )
    return float(per_item.mean())


# ---------------------------------------------------------------------------
# Running a model through all folds
# ---------------------------------------------------------------------------

# Columns removed from the test rows before the model sees them
FUTURE_COLS_HIDDEN = ["units"]


# For each fold: train a fresh model, predict the test window, and attach the
# actuals and segment labels. Returns one row per item per test day.
def run_backtest(make_model: Callable[[], Forecaster], df: pd.DataFrame, config: dict) -> pd.DataFrame:
    bt, sg = config["backtest"], config["segments"]
    folds = make_folds(df["date"].max(), bt["horizon"], bt["n_folds"], bt["step"])
    out = []
    for fold in folds:
        train, test = split(df, fold)
        model = make_model().fit(train)
        future = test.drop(columns=FUTURE_COLS_HIDDEN)
        pred = model.predict(bt["horizon"], future)

        rows = test.merge(pred, on=["item_id", "date"], how="left")
        if rows["pred"].isna().any():
            raise ValueError(f"{model.name}: missing predictions in fold {fold.fold}")

        seg = item_segments(train, sg["intermittent_zero_share"], sg["price_volatile_cv"])
        rows = rows.merge(seg, on="item_id", how="left")
        rows = rows.merge(naive_scale(train).rename("scale"), on="item_id", how="left")
        # M5 has no promotion flag, so I use a price more than 5% below the
        # item's training median as a rough stand-in for a promotion.
        rows["is_price_cut"] = (rows["sell_price"] < 0.95 * rows["median_price"]).astype(int)
        rows["fold"] = fold.fold
        rows["cutoff"] = fold.cutoff
        rows["model"] = model.name
        out.append(rows)
    return pd.concat(out, ignore_index=True)


# Metrics table: overall, then broken down by each segment, for every model
def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    def _metrics(d: pd.DataFrame) -> dict:
        return {"n_rows": len(d), "n_items": d["item_id"].nunique(),
                "MAE": mae(d["units"], d["pred"]), "RMSSE": rmsse(d),
                "bias": bias(d["units"], d["pred"])}

    segmentations = {
        "overall": lambda d: pd.Series("all", index=d.index),
        "volume": lambda d: d["volume_seg"],
        "price": lambda d: d["price_seg"],
        "event": lambda d: d["is_event"].map({1: "event_day", 0: "no_event"}),
        "price_cut": lambda d: d["is_price_cut"].map({1: "price_cut", 0: "regular_price"}),
    }
    rows = []
    for model, d in preds.groupby("model"):
        for seg_name, label_fn in segmentations.items():
            labels = label_fn(d)
            for label, part in d.groupby(labels):
                rows.append({"model": model, "segmentation": seg_name, "segment": label, **_metrics(part)})
    return pd.DataFrame(rows)
