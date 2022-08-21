import yfinance as yf
from datetime import datetime
import pandas as pd

from utils import settings as s


class DataCollector:
    """
    A class for collecting currencies data to train the models
    """
    def update(self):
        """Update the currencies data"""
        for currency in s.CURRENCIES:
            data = yf.download(currency)
            data.to_csv(f"utils/data/{currency}.csv")
