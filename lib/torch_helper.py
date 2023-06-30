# https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
# https://www.sciencedirect.com/science/article/pii/S2666827022000378
# https://www.sciencedirect.com/science/article/pii/S2666827022000378

import pywt
import talib

import numpy as np
import pandas as pd

from lib.scraper import StockScraper


def get_stock_data(
    ticker, period="15y", column_filter=None
):
    macro_data = get_macroeconomic_data().dropna()
    sdata = get_ticker_history(
        ticker, period=period)
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
    stock_data["MACD-0"] = macd[0].fillna(method="bfill")
    stock_data["MACD-1"] = macd[1].fillna(method="bfill")
    stock_data["MACD-2"] = macd[2].fillna(method="bfill")

    # make sure data has even number of rows
    if stock_data.shape[0] % 2 == 1:
        stock_data = stock_data[1:]

    smooth = pywt.swt(
        stock_data["Close"].values,
        wavelet='haar',   # sym12
        level=1,
        trim_approx=True,
        norm=True
    )
    stock_data["CloseWT"] = smooth[0]
    stock_data["CloseWT-noise"] = smooth[1]

    # numbers get wonky at the end,
    # replace last value with the next to last value
    stock_data["CloseWT"][-1:] = stock_data["CloseWT"][-2:-1]
    stock_data["CloseWT-noise"][-1:] = stock_data["CloseWT-noise"][-2:-1]

    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

    stock_data["time_index"] = stock_data["Date"].astype(int)
    stock_data["time_index"] = (
        stock_data["time_index"] / 1000000000 / 86400).astype(int)
    stock_data["time_index"] -= stock_data["time_index"].min()
    # stock_data.head()

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
        macro_data["EFFR"].replace(
            ".", np.nan).fillna(method="bfill")).astype(float)
    macro_data["UMCSENT"] = macro_data["UMCSENT"].fillna(method="ffill")
    macro_data["UNRATE"] = macro_data["UNRATE"].fillna(method="ffill")

    macro_data["EFFR"] = macro_data["EFFR"].astype(float)
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
