# CLAUDE.md — Demand Forecasting Showcase

## Project goal
Build a reproducible forecasting study on **public data** that I will present to my data science team. The study asks:

> Do pretrained time series / LLM-based models actually improve demand forecasting over simple and structured baselines, and does price information help?

It is inspired by the paper *"Rethinking the Role of LLMs in Time Series Forecasting"* (arXiv 2602.14744; code: github.com/EIT-NLP/LLM4TSF), which argues that pretrained LLM backbones help most under distribution shift and out-of-domain settings. I want to test a small slice of that claim in a retail demand + price setting, and connect it to my day-job experience with price-elasticity modeling.

**Hard rules**
- Use only public data (M5). No client or employer data, names, or figures anywhere in the repo.
- Never invent results. Every number in a table, chart, or slide must come from code that was actually run.
- Prefer simple, readable code over clever code. I need to explain every step to my team.

## Data
- **M5 Forecasting (Walmart)** from Kaggle: `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`.
- Download with the Kaggle CLI (`kaggle competitions download -c m5-forecasting-accuracy`) or ask me to place the files in `data/raw/`. Do not commit raw data (add to `.gitignore`).
- To keep everything CPU-friendly, work on a **subset**: one store (e.g. `CA_1`), all three categories sampled down to a few hundred items. Make the subset definition a config value, not hard-coded.
- Convert wide sales to long format: `item_id, store_id, date, units, sell_price, event/calendar features`.

## Evaluation protocol (do this first, everything depends on it)
- **Time-based rolling-origin backtest**, never random splits.
- Horizon: 28 days. At least 3 folds, each fold trains only on data before its cutoff.
- Metrics: MAE, RMSSE (scaled by naive in-sample error), and bias. Avoid sMAPE/MAPE because many series contain zeros. Optionally add WRMSSE as in the M5 competition.
- Report metrics overall **and by segment**: high-volume vs intermittent items, price-stable vs price-volatile items, with and without promotions/events.
- One shared harness: every model implements the same `fit(train_df)` / `predict(horizon)` interface so results are directly comparable.

## Models (build in this order)
1. **Seasonal naive** (same weekday last week).
2. **Price + last-period baseline**: predict units from current price and previous-period units (e.g. simple regression). This mirrors a baseline I used professionally.
3. **LightGBM** with lag features, rolling means, calendar/event features, and price features (price, price ratio vs item's rolling mean, price change).
4. **Chronos, zero-shot** (e.g. `amazon/chronos-bolt-small` via the `chronos-forecasting` package). Univariate, no price input.
5. **PatchTST** (Hugging Face `transformers` implementation or `neuralforecast`) trained on the subset.
6. **Stretch, only after 1–5 work:** mirror the paper's ablation with a small GPT-2 backbone on patched inputs: pretrained vs random-init vs no-LLM (plain MLP/Transformer). Keep it clearly labeled as a small-scale replication, not a reproduction of the paper's numbers.

## Price-response analysis (the bridge to my pricing work)
- For items with enough price variation, fit a **sigmoid-style price-response curve** (units as a function of price ratio vs a reference price) with parameter bounds and regularization to avoid unrealistic demand swings.
- Compare its **+5% / +10% / +15% price-hike scenario** predictions against LightGBM's predictions when the price feature is changed.
- Discuss honestly: ML counterfactuals on observational data are confounded (promotions, seasonality, stockouts), which is why structured elasticity models are often preferred for pricing decisions. Do not claim causal results.

## Repo structure
```
.
├── CLAUDE.md
├── README.md              # question, setup, how to run, headline findings
├── config.yaml            # subset definition, horizon, folds, seeds
├── data/raw/              # gitignored
├── data/processed/
├── src/
│   ├── data.py            # loading, subsetting, long format
│   ├── features.py
│   ├── backtest.py        # rolling-origin harness + metrics
│   ├── models/            # one file per model, shared interface
│   └── price_response.py  # sigmoid fitting + scenario analysis
├── notebooks/             # exploration and final figures only
├── results/               # metrics CSVs, figures (generated, reproducible)
└── slides/                # deck outline and speaker notes (markdown)
```

## Conventions
- Python 3.10+, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `matplotlib`. Pin versions in `requirements.txt`.
- Set random seeds; log them in `results/`.
- Type hints and short docstrings on public functions. Add small tests for the backtest splitter (no leakage) and the metrics.
- Save every experiment's metrics to `results/*.csv` so slides and README can be regenerated from files.
- Figures: clear titles, labeled axes, one message per chart.

## Working style
- Work in phases and stop for my review at the end of each: (1) data + harness + baselines, (2) LightGBM + Chronos + PatchTST, (3) price-response analysis, (4) README + slides.
- Before running anything expensive (model downloads, long training), tell me the expected runtime and ask.
- If a result looks surprising (e.g. a baseline beating every ML model), investigate for leakage or bugs first, then report it plainly. Simple models winning is a valid finding.
- Flag assumptions and limitations explicitly (subset choice, one store, intermittent demand, zero-shot models not seeing price).

## Deliverables
1. Working repo that reproduces all tables and figures from `config.yaml`.
2. `README.md` with the question, method, results table, and 3 key findings.
3. `slides/outline.md`: 10–12 slide outline with speaker notes and 3 discussion questions for the team, e.g. *"When would you trust a zero-shot model for a pricing decision?"* and *"What would you need to make price counterfactuals trustworthy?"*
