from dataclasses import dataclass
import yfinance as yf
from datetime import datetime
import pandas as pd

from utils.data_collector import settings


class DataCollector:
    """
    A class for collecting currencies data to train the models
    """
    def get_data(self, start_date: datetime, end_date: datetime, ticker: str):
        data: pd.DataFrame = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(f"../data/{ticker}.csv")
