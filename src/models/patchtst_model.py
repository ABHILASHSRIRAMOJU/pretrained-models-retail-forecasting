# Model 5: PatchTST, trained from scratch on our subset
#
# A Transformer for time series (Nie et al., 2023). It cuts the recent history
# into short "patches" (here 16 days, sliding by 8) and treats each patch like a
# word in a sentence. One model is trained across all items, and each fold gets
# a freshly trained model, like every other model in the project.
#
# Like Chronos it is univariate: it only sees past `units`, no price or calendar.
# Unlike Chronos it is NOT pretrained: it learns only from our 300 items.
#
# Loss: mean squared error, so the model aims for the mean like LightGBM
# (Tweedie) does. The library default (MAE) would aim for the median, which is
# often 0 for intermittent items.
#
# Training length: up to `max_steps`, with early stopping. The last 28 days of
# each fold's TRAINING data are held out as a validation window; training stops
# once the validation loss hasn't improved for `patience` checks (one check
# every 100 steps). The test window is never used to pick the training length.
#
# Implementation: the `neuralforecast` package (Nixtla).

from __future__ import annotations

import logging

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import PatchTST

from src.backtest import Forecaster

# PyTorch Lightning prints a lot per training run; keep only warnings
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class PatchTSTForecaster(Forecaster):
    name = "patchtst"

    def __init__(self, input_size: int = 112, max_steps: int = 2000, patience: int = 3,
                 learning_rate: float = 1e-4, batch_size: int = 32, seed: int = 42):
        self.input_size = input_size   # days of history the model looks at
        self.max_steps = max_steps
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, train_df: pd.DataFrame) -> "PatchTSTForecaster":
        self.cutoff_ = train_df["date"].max()
        self.horizon_ = 28
        data = train_df.rename(columns={"item_id": "unique_id", "date": "ds", "units": "y"})
        model = PatchTST(
            h=self.horizon_,
            input_size=self.input_size,
            loss=MSE(),
            max_steps=self.max_steps,
            val_check_steps=100,
            early_stop_patience_steps=self.patience,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            # quiet, CPU-only training, no checkpoint files
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        self.nf_ = NeuralForecast(models=[model], freq="D")
        # val_size: hold out the last 28 training days for early stopping.
        # Forecasts still start right after the cutoff, using all training days.
        self.nf_.fit(data[["unique_id", "ds", "y"]], val_size=self.horizon_)
        return self

    def predict(self, horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
        if horizon != self.horizon_:
            raise ValueError(f"model was built for horizon {self.horizon_}, got {horizon}")
        # neuralforecast forecasts the `h` days right after each item's last training day
        fc = self.nf_.predict().rename(columns={"unique_id": "item_id", "ds": "date", "PatchTST": "pred"})
        out = future_df[["item_id", "date"]].merge(fc[["item_id", "date", "pred"]],
                                                    on=["item_id", "date"], how="left")
        # Demand can't be negative
        out["pred"] = out["pred"].clip(lower=0)
        return out
