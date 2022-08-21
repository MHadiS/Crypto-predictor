import yfinance as yf
import pandas as pd

from utils import settings as s


class DataCollector:
    """
    A class for collecting currencies data to train the models
    """

    def update(self) -> None:
        """Update the currencies data"""
        for currency in s.CURRENCIES:
            data: pd.DataFrame = yf.download(currency)
            data.to_csv(f"utils/data/{currency}.csv")
