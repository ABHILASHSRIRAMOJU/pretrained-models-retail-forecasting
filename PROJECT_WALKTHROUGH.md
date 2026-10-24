# Project walkthrough: what we have done so far

This is a plain-language walkthrough of the project, for explaining it to the team.
It covers **Phase 1** (data, backtest harness, two baselines). Later phases will be
added once their code has been run. Every number below comes from
`results/metrics_baselines.csv`, which `python -m src.run_baselines` generates.

---

## 1. The question

> Do pretrained time series / LLM-based models actually forecast retail demand better
> than simple baselines, and does price information help?

The idea comes from the paper *Rethinking the Role of LLMs in Time Series Forecasting*
(arXiv 2602.14744). The plan is to build fair, simple baselines first, and only then
compare them with ML and pretrained models on the same footing.

---

## 2. The data (`src/data.py`, `config.yaml`)

**Source:** M5 Forecasting (Walmart, public Kaggle data). Three files:

| File | What it holds |
|---|---|
| `sales_train_evaluation.csv` | Daily units sold per item and store, in **wide** format (one column per day: `d_1`, `d_2`, ...) |
| `calendar.csv` | Maps each day to a date, weekday, Walmart week, events, SNAP days |
| `sell_prices.csv` | **Weekly** price per item and store |

**The subset** (all set in `config.yaml`, nothing hard-coded):
- One store: `CA_1`
- 100 items from each category (FOODS, HOBBIES, HOUSEHOLD), so **300 items**
- Items that sold on fewer than 30 days in total are removed before sampling
- Sampling uses a fixed seed (42), so every run picks the same items

**Steps in the pipeline:**
1. Keep store `CA_1` and sample the items.
2. Reshape **wide to long**: one row per item per day.
3. Join calendar features (date, weekday, month, events, SNAP for California).
4. Join the weekly price using the Walmart week id.
5. **Drop days with no price.** No price means the item was not on the shelf yet,
   so those days are not real zero-demand days.
6. Save to `data/processed/m5_subset_long.csv`.

**Final table columns:**

| Column | Meaning |
|---|---|
| `item_id`, `cat_id`, `store_id` | Which item, category, store |
| `date` | The day |
| `units` | Units sold that day. **This is the target we forecast.** |
| `sell_price` | Price that week |
| `wday`, `month`, `year` | Calendar features |
| `event_name_1`, `event_type_1`, `is_event` | Holiday or event on that day (e.g. SuperBowl) |
| `snap` | 1 if it is a SNAP (food stamp) day in California |

A small data quirk: our copy of `calendar.csv` had no `d` column, so the code rebuilds
it from the dates, and first checks that there are no gaps in the dates.

---

## 3. How models are evaluated (`src/backtest.py`)

This is the most important part. Every model goes through exactly the same code, so
the results can be compared directly.

### Rolling-origin backtest (split by time, never randomly)

We pretend to stand at a past date (the **cutoff**), train on everything up to that
date, and forecast the next **28 days**. We do this 3 times:

| Fold | Train on data up to (cutoff) | Forecast (test) window |
|---|---|---|
| 0 | 2016-02-28 | 2016-02-29 to 2016-03-27 |
| 1 | 2016-03-27 | 2016-03-28 to 2016-04-24 |
| 2 | 2016-04-24 | 2016-04-25 to 2016-05-22 |

A fresh model is trained in each fold. It never sees data after its cutoff.

### One shared interface for all models

Every model has two methods:

- `fit(train_df)`: learn from the history up to the cutoff.
- `predict(horizon, future_df)`: forecast the 28 test days. `future_df` holds the rows
  to forecast with anything known in advance (price, calendar, events). **The harness
  removes the `units` column first**, so a model cannot see the answers.

Assumption: future prices are known in advance (planned prices), same as in M5.

### Metrics

| Metric | What it tells you | Good value |
|---|---|---|
| **MAE** | Average error in units per item per day | Lower is better |
| **RMSSE** | Error relative to a "yesterday = today" naive forecast, per item, then averaged | Below 1 means better than naive |
| **Bias** | sum(forecast − actual) / sum(actual) | Close to 0. Positive = over-forecasting, negative = under-forecasting |

We do **not** use MAPE or sMAPE: many items have zero-sales days, and percentage errors
blow up or become undefined at zero.

### Segments (where does each model win or lose?)

Results are also split by item type. Segment labels are computed from **training data
only**, so the test period cannot leak into them.

| Segmentation | Groups | Rule |
|---|---|---|
| Volume | `high_volume` / `intermittent` | Intermittent = more than 50% zero-sales days in training |
| Price | `price_stable` / `price_volatile` | Volatile = price coefficient of variation above 5% |
| Event | `event_day` / `no_event` | Holiday or event on that day |
| Price cut | `price_cut` / `regular_price` | Price more than 5% below the item's training median. M5 has no promotion flag, so this is a rough stand-in. |

### Tests (`tests/`)

10 tests, all passing (`pytest`). They check that:
- train and test never overlap, and the model never sees future units
- MAE, bias and RMSSE match hand-calculated values
- seasonal naive is exactly right on a pure weekly pattern
- the price model returns a prediction for every row and never predicts negative demand
- the data pipeline drops off-shelf days and joins prices and events correctly

---

## 4. The models so far (`src/models/`)

### Model 1: Seasonal naive (`seasonal_naive.py`)

Each future day = what the item sold on the **same weekday in the last week of
training**. Every future Monday gets last Monday's sales, and so on.
It uses no price. It is the "anything should beat this" baseline.

### Model 2: Price + last-period regression (`price_last_period.py`)

One small **Ridge regression per item**:

```
units_t = a + b * log(price_t) + c * last_period_mean_t
```

- `log(price_t)`: the price on the day being forecast (known in advance).
- `last_period_mean_t`: the item's average daily units over days t−55 to t−28.

**Why shift by 28 days?** The horizon is 28 days. Shifting the window back by 28 means
that for every day we forecast, the window is fully in the past. So the model predicts
all 28 days at once, without feeding its own predictions back in and without peeking.

**Why Ridge and not plain linear regression?** Ridge adds a small penalty on the
coefficients. Many items barely change price, and plain regression could give them a
huge, meaningless price coefficient. The penalty keeps it sensible.

**Fallback:** if an item does not have enough history, it predicts the item's last
28-day average. Predictions are clipped at 0 because demand cannot be negative.

---

## 5. What is in the results files

### `results/predictions_baselines.csv`

One row per **item × test day × fold × model**
(300 items × 28 days × 3 folds × 2 models = 50,400 rows).

| Column | Meaning |
|---|---|
| `units` | **Actual** units sold that day (the truth) |
| `pred` | **The model's forecast** of `units` for that item and day, made at the fold's cutoff |
| `model` | Which model made the forecast |
| `fold`, `cutoff` | Which backtest fold, and the last date the model was allowed to see |
| `scale` | The item's naive in-sample error, used as the RMSSE denominator |
| `zero_share`, `price_cv`, `median_price` | Item statistics from training, used to build segments |
| `volume_seg`, `price_seg`, `is_price_cut`, `is_event` | Segment labels |

Error for a row = `pred − units`. All metrics are built from these two columns.

### `results/metrics_baselines.csv`

MAE, RMSSE and bias per model, overall and by every segment.

### `results/run_config_baselines.json`

The exact config (including the seed) used for the run, for reproducibility.

---

## 6. Phase 1 results (300 items, 3 folds × 28 days)

**Overall:**

| Model | MAE | RMSSE | Bias |
|---|---|---|---|
| Seasonal naive | 1.262 | 0.963 | −4.1% |
| Price + last-period (Ridge) | **1.132** | **0.735** | **+0.2%** |

**By segment (RMSSE):**

| Segment | Seasonal naive | Price + last-period |
|---|---|---|
| High-volume (111 items) | 0.935 | 0.744 |
| Intermittent (192 items) | 0.979 | 0.729 |
| Price-stable (245 items) | 0.957 | 0.724 |
| Price-volatile (57 items) | 0.990 | 0.782 |
| Event days | 0.921 | 0.708 |
| Price-cut days (24 items) | 1.066 | 0.846 |

(Item counts per segment can add up to slightly more than 300, because the labels are
computed per fold and an item can change segment between folds.)

**What we read from this:**
1. The price + last-period model beats seasonal naive in **every** segment, on both MAE
   and RMSSE, and it is almost unbiased overall.
2. Seasonal naive under-forecasts (−4.1%), and most on price-volatile items (−9.6%).
   Copying a single week is noisy and misses price changes.
3. By RMSSE, both models do worst on **price-volatile** and **price-cut** items. That is where
   price information should matter most, so it is the key segment to watch when we
   compare LightGBM and the pretrained models.

This is the bar the ML and pretrained models must clear.

---

## 7. Limitations

- One store and a sample of 300 items. Results may not generalize to other stores.
- Future prices are treated as known.
- "Price cut" is a proxy (price more than 5% below the median), not a real promotion flag.
- Only 3 folds, all in spring 2016, so the test windows cover one season.
- Zero-shot pretrained models (next phase) will not see price.

---

## 8. What comes next

| Phase | Work | Uses price? |
|---|---|---|
| 2 | LightGBM with lag, rolling, calendar and price features | Yes |
| 2 | Chronos-Bolt, zero-shot (pretrained, no training on our data) | No, it only sees past `units` |
| 2 | PatchTST trained on our subset | No |
| 3 | Price-response curves and +5% / +10% / +15% price-hike scenarios | Yes |
| 4 | README findings and team slides | — |

All new models will plug into the same `fit` / `predict` interface and run through the
same folds, metrics and segments, so their numbers sit right next to the table above.
