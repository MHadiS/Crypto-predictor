import yfinance as yf
from datetime import datetime
import pandas as pd

from utils.data_collector import settings as s


class DataCollector:
    """
    A class for collecting currencies data to train the models
    """
    def get_data(self, start_date: datetime, end_date: datetime, ticker: str) -> pd.DataFrame:
        """Get the currencies data with yfinance api

        Args:
            start_date (datetime): start date of data
            end_date (datetime): end data of data
            ticker (str): the symbol for the finance
        
        Return:
            pandas DataFrame: downloaded data by yfinance
        """
        data: pd.DataFrame = yf.download(ticker, start=start_date, end=end_date)
        return data
    
    def update(self):
        """Update the currencies data
        """
        today = datetime.today()
        start = today.replace(today.year - 1)
        for currency in s.CURRENCIES:
            data = self.get_data(start, today, currency)
            data.to_csv(f"../data/{currency}.csv")


