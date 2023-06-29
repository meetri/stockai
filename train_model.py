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
    learning_rate=0.01563
)
trainer.create_model()

# configure network and trainer

# Trial 0 finished with value: 2.4022581577301025 and parameters
hyper_params = {
    "gradient_clip_val": 0.05130328983794089,
    "hidden_size": 97,
    "dropout": 0.20191453684086638,
    "hidden_continuous_size": 61,
    "attention_head_size": 2,
    "learning_rate": 0.015631929305749356
}

trainer.train(**hyper_params, optimizer="Ranger")

best_tft = trainer.best_tft()
raw_predictions = best_tft.predict(
    trainer.val_dataloader,
    mode="raw",
    return_x=True,
    fast_dev_run=False
)

for k in range(len(raw_predictions[0][0])):
    plt = best_tft.plot_prediction(
        raw_predictions.x,
        raw_predictions.output,
        idx=k, add_loss_to_title=True
    )
    plt.legend()
    plt.savefig(f"./train_images/predict-{k}.png", dpi=300)
