import pandas as pd

from lib.torch_helper import (
    StockTrainer, get_stock_data
)

# Configuration

amd = get_stock_data(
    "amd", column_filter=["High", "Low", "Close", "Volume", "Date"])

spy = get_stock_data(
    "spy", column_filter=["High", "Low", "Close", "Volume", "Date"])

stock_data = pd.concat([amd, spy]).reset_index(drop=True)
stock_data["Date"] = pd.to_datetime(stock_data["Date"])

print(stock_data.head(5))

# tdata_size = round(stock_data.shape[0] * 0.2)
# stock_data = stock_data[-tdata_size:]

# print("Using {tdata_size} row for calculating optimizated hyperparameters")
trainer = StockTrainer(
    stock_data,
    target="Close",
    train_on=[
        # "Imp Vol", "Put/Call Vol", "Options Vol", "Put/Call OI", "EFFR",
        "UMCSENT", "UNRATE", "USDX", "VIX", "Close", "Volume",
        "ATR", "RSI", "MACD"
    ],
    batch_size=64,
    max_epoch=100,
    learning_rate=0.001059429134884675
)
trainer.create_model()

lr = trainer.calculate_ideal_learning_rate()
print(f"suggested learning rate: {lr.suggestion()}")

study = trainer.optimize_hyperparameters()
print(study.best_trial.params)
