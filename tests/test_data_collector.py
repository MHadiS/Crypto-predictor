import pytest as pt
from datetime import datetime

from utils import DataCollector, CURRENCIES


@pt.fixture
def data_collector():
    return DataCollector()


def test_get_data(data_collector):
    start = datetime(2021, 1, 1)
    end = datetime(2022, 1, 1)
    data = data_collector.get_data(start, end, "BTC-USD")
    assert list(data.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
