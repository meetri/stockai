from os import getenv
import polygon
import yfinance as yf
import requests_cache


class StockScraper():

    def __init__(
        self, ticker: str, start=None, end=None, interval=None, period=None
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.period = period

        if (
            start is None and end is None
            and interval is None and period is None
        ):
            self.interval = "1d"
            self.period = "10y"

        self.client = None
        self.data = None

    def from_yahoo(self):
        session = requests_cache.CachedSession('yfinance.cache')
        session.headers['User-agent'] = 'stock-scraper/1.0'
        self.client = yf.Ticker(self.ticker, session=session)
        self.data = self.client.history(
            period=self.period,
            interval=self.interval,
            auto_adjust=True,
            actions=True,
            start=self.start,
            end=self.end,
            prepost=False
        )

    def from_poly(self):
        polykey = getenv("POLYKEY")
        if polykey:
            self.client = polygon.StocksClient(polykey)
            self.data = self.client.get_aggregate_bars(
                symbol=self.ticker, from_date=self.start, to_date=self.end,
                adjusted=True, sort="asc", limit=50000, run_parallel=True,
                multiplier=1, timespan='day', full_range=True, warnings=False
            )

        else:
            raise Exception("polygon api token missing from env:POLYKEY")
