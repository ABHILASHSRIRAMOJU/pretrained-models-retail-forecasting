# Tests for the backtest: no leakage between train/test, metrics match
# hand-calculated values, and both baselines behave as expected.

import numpy as np
import pandas as pd
import pytest

from src.backtest import Forecaster, bias, mae, make_folds, naive_scale, rmsse, run_backtest, split
from src.models.price_last_period import PriceLastPeriod
from src.models.seasonal_naive import SeasonalNaive

CONFIG = {
    "backtest": {"horizon": 28, "n_folds": 3, "step": 28},
    "segments": {"intermittent_zero_share": 0.5, "price_volatile_cv": 0.05},
}


def make_df(n_days=200, items=("A", "B")):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for i, item in enumerate(items):
        weekly = np.array([1, 2, 3, 4, 5, 6, 7]) * (i + 1)
        rows.append(pd.DataFrame({
            "item_id": item, "date": dates,
            "units": weekly[dates.dayofweek].astype(float),
            "sell_price": 2.0, "is_event": 0,
        }))
    return pd.concat(rows, ignore_index=True)


# ---- folds / leakage ------------------------------------------------------

def test_folds_are_ordered_and_end_at_last_date():
    last = pd.Timestamp("2020-12-31")
    folds = make_folds(last, horizon=28, n_folds=3, step=28)
    assert [f.fold for f in folds] == [0, 1, 2]
    assert folds[-1].test_end == last
    assert all(a.cutoff < b.cutoff for a, b in zip(folds, folds[1:]))


def test_split_has_no_leakage():
    df = make_df()
    for fold in make_folds(df["date"].max(), 28, 3, 28):
        train, test = split(df, fold)
        assert train["date"].max() <= fold.cutoff < test["date"].min()
        assert test["date"].nunique() == 28


# Fake model that records what the harness showed it,
# so I can check it never sees future dates or actual units
class SpyModel(Forecaster):
    name = "spy"
    seen = []

    def fit(self, train_df):
        self.max_train_date = train_df["date"].max()
        return self

    def predict(self, horizon, future_df):
        SpyModel.seen.append((self.max_train_date, future_df["date"].min(), "units" in future_df))
        return future_df[["item_id", "date"]].assign(pred=0.0)


def test_harness_hides_future_units():
    SpyModel.seen = []
    run_backtest(SpyModel, make_df(), CONFIG)
    assert len(SpyModel.seen) == 3
    for max_train, first_future, has_units in SpyModel.seen:
        assert max_train < first_future
        assert not has_units


# ---- metrics --------------------------------------------------------------

def test_mae_and_bias_by_hand():
    y, yhat = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])
    assert mae(y, yhat) == pytest.approx(1.0)          # (1 + 0 + 2) / 3
    assert bias(y, yhat) == pytest.approx(3.0 / 6.0)   # sum(err) / sum(y)


def test_rmsse_by_hand():
    train = pd.DataFrame({"item_id": "A", "date": pd.date_range("2020-01-01", periods=4),
                          "units": [0.0, 2.0, 0.0, 2.0]})
    scale = naive_scale(train)["A"]
    assert scale == pytest.approx(4.0)                 # diffs: 2,-2,2 -> mean sq 4
    preds = pd.DataFrame({"item_id": "A", "units": [1.0, 1.0], "pred": [3.0, 3.0], "scale": scale})
    assert rmsse(preds) == pytest.approx(1.0)          # sqrt(4 / 4)


# ---- models ---------------------------------------------------------------

def test_seasonal_naive_is_perfect_on_pure_weekly_pattern():
    preds = run_backtest(SeasonalNaive, make_df(), CONFIG)
    assert mae(preds["units"], preds["pred"]) == pytest.approx(0.0)


def test_price_last_period_returns_all_rows_nonnegative():
    df = make_df()
    preds = run_backtest(PriceLastPeriod, df, CONFIG)
    assert len(preds) == 2 * 28 * 3
    assert (preds["pred"] >= 0).all()
