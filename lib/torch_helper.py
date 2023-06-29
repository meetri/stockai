# https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
# https://www.sciencedirect.com/science/article/pii/S2666827022000378
# https://www.sciencedirect.com/science/article/pii/S2666827022000378

import pickle
import talib
import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import (Baseline, TemporalFusionTransformer,
                                 TimeSeriesDataSet)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters)

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from lib.scraper import StockScraper

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


class StockTrainer():

    def __init__(self, data: pd.DataFrame, **kwargs):
        self.data = data

        self.training = None
        self.validation = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.trainer = None
        self.tft = None

        # mps / cuda ...
        self.device = kwargs.get("device", "cpu")
        self.torch_device = torch.device(self.device)

        self.group_id = kwargs.get("group_id", "Ticker")
        self.group_ids = [self.group_id]

        self.target = kwargs.get("target", "Close")

        # we can be more precise later...
        self.static_categoricals = self.group_ids

        # time_varying_unknown_reals=[
        #    # "Imp Vol", "Put/Call Vol", "Options Vol", "Put/Call OI",
        #    "EFFR", "UMCSENT", "UNRATE", "Dollar Index", "VIX",  "Close",
        #    "Volume"
        # ],
        self.train_on = kwargs.get("train_on", ["Close", "Volume"])
        self.learning_rate = kwargs.get("learning_rate", 0.001)

        self.max_prediction_length = kwargs.get("max_prediction_length", 6)
        self.max_encoder_length = kwargs.get("max_encoder_length", 30)
        self.max_epoch = kwargs.get("max_epoch", 50)
        self.batch_size = kwargs.get("batch_size", 32)

        self.time_idx = kwargs.get("time_idx", "time_idx")

    def create_model(self):

        training_cutoff = (
            self.data["time_idx"].max() - self.max_prediction_length)

        self.training = TimeSeriesDataSet(
            data=self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx=self.time_idx,
            target="Close",
            group_ids=self.group_ids,
            weight=None,

            max_encoder_length=self.max_encoder_length,
            min_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,

            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_unknown_reals=self.train_on,

            time_varying_known_reals=[self.time_idx, "Date"],
            time_varying_unknown_categoricals=[],

            variable_groups={},
            lags={},
            static_categoricals=self.group_ids,

            constant_fill_strategy={},

            target_normalizer=GroupNormalizer(
               groups=self.group_ids, transformation="softplus"
            ),

            add_relative_time_idx=True,  # add as feature
            add_target_scales=True,  # add as feature
            add_encoder_length=True,  # add as feature
        )

        # create validation set (predict=True) which means to predict the
        # last max_prediction_length points in time for each series
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, self.data, predict=True, stop_randomization=True
        )
        # create dataloaders for model
        self.train_dataloader = self.training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        self.val_dataloader = self.validation.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )

    def baseline_predictions(self):
        baseline_predictions = Baseline().predict(
            self.val_dataloader, return_y=True)
        return MAE()(baseline_predictions.output, baseline_predictions.y)

    def best_tft(self):
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        return TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    def train(self, **kwargs):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=10,
            verbose=False,
            mode="min"
        )

        # log the learning rate
        lr_logger = LearningRateMonitor()

        # logging results to a tensorboard
        logger = TensorBoardLogger("lightning_logs")

        self.trainer = pl.Trainer(
            max_epochs=self.max_epoch,
            accelerator=self.device,
            enable_model_summary=True,
            gradient_clip_val=kwargs.get(
                "gradient_clip_val", 0.23671769794503256),
            # coment in for training, running validation every 30 batches
            # limit_train_batches=50,
            # comment in to check that networkor dataset has no serious bugs
            # fast_dev_run=True,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=kwargs.get("learning_rate", 0.001059429134884675),
            hidden_size=kwargs.get("hidden_size", 40),
            attention_head_size=kwargs.get("attention_head_size", 2),
            dropout=kwargs.get("dropout", 0.1691311061383082),
            hidden_continuous_size=kwargs.get("hidden_continuous_size", 12),
            loss=QuantileLoss(),
            log_interval=10,
            # “ranger”, “sgd”, “adam”,
            # “adamw” or class name of optimizer in
            optimizer=kwargs.get("optimizer", "Ranger")
            # reduce_on_plateau_patience=4,
        )

        self.trainer.fit(
            self.tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )

    def calculate_ideal_learning_rate(self):

        if not self.training:
            self.create_model()

        """
        Trial 0 finished with value: 4.715355396270752 and parameters:
        {
            'gradient_clip_val': 0.02475856678316859,
            'hidden_size': 105,
            'dropout': 0.13600878831196433,
            'hidden_continuous_size': 25,
            'attention_head_size': 1,
            'learning_rate': 0.001015009495936967
        }. Best is trial 0 with value: 4.715355396270752.
        """

        # configure network and trainer
        pl.seed_everything(42)
        trainer = pl.Trainer(
            accelerator=self.device,
            gradient_clip_val=0.02475856678316859,
        )

        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.003134745802918898,
            # most important hyperparameter apart from learning rate
            hidden_size=105,

            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,

            # between 0.1 and 0.3 are good values
            dropout=0.13600878831196433,

            # set to <= hidden_size
            hidden_continuous_size=25,
            loss=QuantileLoss(),

            # “ranger”, “sgd”, “adam”,
            # “adamw” or class name of optimizer in
            optimizer="Ranger"
            # reduce learning rate if no improvement in validation
            # loss after x epochs
            # reduce_on_plateau_patience=1000,
        )

        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        return Tuner(trainer).lr_find(
            tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )

    def optimize_hyperparameters(self):
        study = optimize_hyperparameters(
            self.train_dataloader,
            self.val_dataloader,
            model_path="optuna_test",
            n_trials=100,
            max_epochs=self.max_epoch,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )

        # save study results
        # - also we can resume tuning at a later point in time
        with open("test_study.pkl", "wb") as fout:
            pickle.dump(study, fout)

        return study


def get_stock_data(
    ticker, period="15y",
    column_filter=["High", "Low", "Close", "Volume", "Date"]
):
    macro_data = get_macroeconomic_data().dropna()
    sdata = get_ticker_history(
        ticker, period=period)[column_filter]
    sdata["Ticker"] = ticker.upper()
    vix = get_ticker_history(
        "^vix",
        period=period
    )[["Close", "Date"]].rename(columns={"Close": "VIX"})
    usdx = get_ticker_history(
        "dx-y.nyb",
        period=period
    )[["Close", "Date"]].rename(columns={"Close": "USDX"})
    stock_data = sdata.merge(macro_data, on="Date", how="inner")
    stock_data = stock_data.merge(vix, on="Date", how="inner")
    stock_data = stock_data.merge(usdx, on="Date", how="inner")
    stock_data["time_idx"] = range(stock_data.shape[0])

    rsi = talib.RSI(stock_data["Close"], 14)
    atr = talib.ATR(
        stock_data["High"], stock_data["Low"], stock_data["Close"], 14)
    macd = talib.MACD(stock_data["Close"])
    stock_data["ATR"] = atr.fillna(method="bfill")
    stock_data["RSI"] = rsi.fillna(method="bfill")
    stock_data["MACD"] = macd[2].fillna(method="bfill")
    stock_data.head()

    return stock_data


def get_macroeconomic_data():
    # https://fred.stlouisfed.org/series/UNRATE
    effr_data = pd.read_csv("macroeconomics_data/EFFR.csv")
    umcsent_data = pd.read_csv("macroeconomics_data/UMCSENT.csv")
    unrate_data = pd.read_csv("macroeconomics_data/UNRATE.csv")

    macro_data = effr_data.merge(umcsent_data, on="DATE", how="left")
    macro_data = macro_data.merge(unrate_data, on="DATE", how="left")
    macro_data["Date"] = macro_data["DATE"]
    macro_data = macro_data.drop(columns=["DATE"])

    macro_data["EFFR"] = (
        macro_data["EFFR"].replace(".", np.nan).fillna(method="bfill"))
    macro_data["UMCSENT"] = macro_data["UMCSENT"].fillna(method="ffill")
    macro_data["UNRATE"] = macro_data["UNRATE"].fillna(method="ffill")
    return macro_data


def get_ticker_history(ticker, interval="1d", period="10y"):
    scraper = StockScraper(ticker, interval=interval, period=period)
    scraper.from_yahoo()
    sdata = scraper.data
    sdata["Date"] = sdata.index.strftime("%Y-%m-%d")
    sdata = sdata.reset_index(drop=True)
    return sdata


def get_options_history(ticker):
    options_data = pd.read_csv(f"options_data/{ticker}.csv")
    for col in ["Imp Vol"]:
        options_data[col] = (
            options_data[col].str.rstrip('%').astype('float') / 100.0)

    return options_data
