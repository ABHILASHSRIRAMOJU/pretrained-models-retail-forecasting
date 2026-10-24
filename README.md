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
| 1 | Data pipeline, backtest harness, baselines | 🚧 in progress |
| 2 | LightGBM, Chronos (zero-shot), PatchTST | ⏳ |
| 3 | Price-response curves & price-hike scenarios | ⏳ |
| 4 | Findings & slides | ⏳ |

Results will be added here only after they are produced by code in this repo.

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
```

Download `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv` from
[Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into `data/raw/`.

## Run

```bash
pytest                      # leakage + metric tests
python -m src.run_baselines # writes results/metrics_baselines.csv
```

## Repo layout

```
config.yaml          subset, horizon, folds, seed
src/data.py          load M5, subset, wide -> long
src/backtest.py      folds, shared model interface, metrics, segments
src/models/          one file per model
tests/               no-leakage and metric tests
results/             generated metrics and figures
```

## Limitations

- One store and a sampled subset of items; results may not generalize across stores.
- Future prices are treated as known (planned prices), as in M5.
- M5 has no promotion flag; "price cut" (>5% below the item's training median) is a proxy.
- Zero-shot models do not see price.

## Data & license

M5 data is provided by Walmart via Kaggle under the competition's terms and is not redistributed here.
