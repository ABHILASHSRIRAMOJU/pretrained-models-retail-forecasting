# Do pretrained / LLM-based models improve retail demand forecasting?

A reproducible forecasting study on the public **M5 (Walmart)** dataset, asking:

> Do pretrained time series / LLM-based models actually improve demand forecasting
> over simple and structured baselines — and does price information help?

Inspired by *Rethinking the Role of LLMs in Time Series Forecasting*
([arXiv 2602.14744](https://arxiv.org/abs/2602.14744), [code](https://github.com/EIT-NLP/LLM4TSF)),
which argues pretrained LLM backbones help most under distribution shift. This repo
tests a small slice of that claim in a retail demand + price setting.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Data pipeline, backtest harness, baselines | ✅ done |
| 2 | LightGBM, Chronos (zero-shot), PatchTST | ✅ done |
| 3 | Price-response curves & price-hike scenarios | ⏳ |
| 4 | Findings & slides | ⏳ |

## Results so far

Overall, one store (`CA_1`), 300 items, 3 folds × 28 days. Lower MAE / RMSSE is better;
bias is relative (negative = under-forecast). Numbers come from `results/metrics_*.csv`.

| Model | Uses price? | MAE | RMSSE | Bias |
|---|---|---|---|---|
| Seasonal naive | No | 1.262 | 0.963 | −0.041 |
| Price + last-period ridge | Yes | 1.132 | 0.735 | +0.002 |
| LightGBM | Yes | 1.097 | **0.726** | −0.031 |
| Chronos-Bolt, median | No | **0.988** | 0.777 | −0.348 |
| Chronos-Bolt, mean of quantiles | No | 1.025 | 0.732 | −0.141 |
| PatchTST | No | 1.052 | 0.731 | −0.067 |

Early read: every model beats seasonal naive, but the gap between the simple price
baseline and the stronger models is small on RMSSE. Chronos' median gets the lowest
MAE by forecasting low on intermittent items, which shows up as a large negative bias.
Segment-level results are in the same CSV files.

## Method

- **Data:** one store (`CA_1`), ~100 items per category, configured in `config.yaml`.
- **Evaluation:** rolling-origin backtest, 3 folds × 28-day horizon. Each fold trains only on data before its cutoff (tested in `tests/test_backtest.py`).
- **Metrics:** MAE, RMSSE (scaled by in-sample naive error), and relative bias. MAPE/sMAPE are avoided because many series contain zeros.
- **Segments:** high-volume vs intermittent, price-stable vs price-volatile, event vs non-event days, price-cut vs regular price.

| # | Model | Uses price? |
|---|---|---|
| 1 | Seasonal naive (same weekday last week) | No |
| 2 | Price + last-period ridge regression (per item) | Yes |
| 3 | LightGBM with lag, calendar, and price features | Yes |
| 4 | Chronos-Bolt, zero-shot | No |
| 5 | PatchTST | No |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp          # macOS only, needed by LightGBM
```

The first Chronos run downloads the pretrained weights (~200 MB) from Hugging Face.

Download `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv` from
[Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/raw/`.

## Run

```bash
pytest                       # leakage, metric and feature tests
python -m src.run_baselines  # seasonal naive + price/last-period ridge
python -m src.run_lightgbm   # LightGBM (also writes feature importance)
python -m src.run_chronos    # Chronos-Bolt, zero-shot
python -m src.run_patchtst   # PatchTST (trains with early stopping)
```

Each script writes to `results/`: `predictions_<model>.csv`, `metrics_<model>.csv`
(overall and by segment), and `run_config_<model>.json` (the exact config and seed used).
Model settings live in `config.yaml`.

## Repo layout

```
config.yaml          subset, horizon, folds, seed
src/data.py          load M5, subset, wide -> long
src/features.py      lag, rolling, calendar and price features (LightGBM)
src/backtest.py      folds, shared model interface, metrics, segments
src/models/          one file per model
src/run_*.py         one entry point per model
tests/               no-leakage, metric and feature tests
results/             generated metrics and figures
```

## Limitations

- One store and a sampled subset of items; results may not generalize across stores.
- Future prices are treated as known (planned prices), as in M5.
- M5 has no promotion flag; "price cut" (>5% below the item's training median) is a proxy.
- Zero-shot models do not see price.

## Data & license

M5 data is provided by Walmart via Kaggle under the competition's terms and is not redistributed here.
