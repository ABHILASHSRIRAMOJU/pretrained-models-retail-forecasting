# Tests for the LightGBM features and model: sales features never use units
# from inside the forecast window, and the model fits the shared interface.

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.features import SALES_FEATURES, build_features
from src.models.lightgbm_model import LightGBMForecaster

CONFIG = {
    "backtest": {"horizon": 28, "n_folds": 3, "step": 28},
    "segments": {"intermittent_zero_share": 0.5, "price_volatile_cv": 0.05},
}


def make_df(n_days=200, items=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for item in items:
        rows.append(pd.DataFrame({
            "item_id": item, "cat_id": "FOODS", "date": dates,
            "units": rng.poisson(3, n_days).astype(float),
            "sell_price": np.where(np.arange(n_days) % 50 < 10, 1.8, 2.0),
            "is_event": 0, "event_type_1": np.nan, "snap": 0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_sales_features_ignore_units_inside_the_horizon():
    df = make_df()
    cutoff = df["date"].max() - pd.Timedelta(days=28)
    future = df["date"] > cutoff

    # Scramble every unit value in the last 28 days; features there must not change
    scrambled = df.copy()
    scrambled.loc[future, "units"] = 999.0

    a = build_features(df).set_index(["item_id", "date"])
    b = build_features(scrambled).set_index(["item_id", "date"])
    idx = df.loc[future].set_index(["item_id", "date"]).index
    pd.testing.assert_frame_equal(a.loc[idx, SALES_FEATURES], b.loc[idx, SALES_FEATURES])


def test_price_features():
    feats = build_features(make_df(n_days=60, items=("A",)))
    # Days 0-9 at 1.8, days 10-49 at 2.0, days 50-59 back at 1.8
    assert np.isclose(feats.iloc[7]["price_change_7"], 0.0)              # 1.8 vs 1.8
    assert np.isclose(feats.iloc[13]["price_change_7"], 2.0 / 1.8 - 1)   # 2.0 vs 1.8
    assert np.isclose(feats.iloc[55]["price_ratio_max"], 1.8 / 2.0)      # below max price
    assert np.isclose(feats.iloc[55]["price_ratio_28"], 1.8 / ((22 * 2.0 + 6 * 1.8) / 28))


def test_lightgbm_returns_all_rows_nonnegative():
    preds = run_backtest(lambda: LightGBMForecaster(num_boost_round=20), make_df(), CONFIG)
    assert len(preds) == 2 * 28 * 3
    assert (preds["pred"] >= 0).all()
