# Tests for data.py using a tiny fake version of the M5 files

import pandas as pd

from src.data import select_items, to_long


def fake_m5():
    sales = pd.DataFrame({
        "id": ["X_1", "Y_1", "Z_1"], "item_id": ["X", "Y", "Z"],
        "cat_id": ["FOODS", "FOODS", "HOBBIES"], "store_id": ["CA_1", "CA_1", "TX_1"],
        "d_1": [0, 1, 5], "d_2": [3, 1, 5], "d_3": [4, 0, 5],
    })
    calendar = pd.DataFrame({
        "d": ["d_1", "d_2", "d_3"], "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
        "wm_yr_wk": [1, 1, 2], "wday": [1, 2, 3], "month": 1, "year": 2011,
        "event_name_1": [None, "SuperBowl", None], "event_type_1": [None, "Sporting", None],
        "snap_CA": [0, 0, 1], "snap_TX": 0, "snap_WI": 0,
    })
    # Item X has no price in week 1 -> not on shelf yet.
    prices = pd.DataFrame({"store_id": "CA_1", "item_id": ["X", "Y", "Y"],
                           "wm_yr_wk": [2, 1, 2], "sell_price": [1.0, 2.0, 2.5]})
    return sales, calendar, prices


def test_select_items_filters_store_and_is_reproducible():
    sales, _, _ = fake_m5()
    a = select_items(sales, "CA_1", items_per_category=5, min_nonzero_days=1, seed=0)
    b = select_items(sales, "CA_1", items_per_category=5, min_nonzero_days=1, seed=0)
    assert set(a["item_id"]) == {"X", "Y"}
    pd.testing.assert_frame_equal(a, b)


def test_to_long_joins_price_and_drops_off_shelf_days():
    sales, calendar, prices = fake_m5()
    long = to_long(sales[sales["store_id"] == "CA_1"], calendar, prices, "CA_1")
    assert len(long[long["item_id"] == "X"]) == 1        # only d_3 has a price
    y = long[long["item_id"] == "Y"].set_index("date")
    assert y["sell_price"].tolist() == [2.0, 2.0, 2.5]
    assert y["is_event"].tolist() == [0, 1, 0]
    assert y["snap"].tolist() == [0, 0, 1]


def test_add_day_index_rebuilds_missing_d_column():
    from src.data import add_day_index
    cal = pd.DataFrame({"date": ["2011-01-30", "2011-01-29", "2011-01-31"]})
    assert add_day_index(cal)["d"].tolist() == ["d_1", "d_2", "d_3"]
