import yfinance as yf
from datetime import datetime
import pandas as pd

from utils.data_collector import settings


class DataCollector:
    """
    A class for collecting currencies data to train the models
    """
