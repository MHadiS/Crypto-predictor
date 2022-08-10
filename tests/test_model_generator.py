import pytest as pt

from utils import ModelGenerator


@pt.fixture
def model_generator():
    return ModelGenerator()


def test_read_data(model_generator): 
    global X_train, X_test, X_val, y_train, y_test, y_val
    X_train, X_test, X_val, y_train, y_test, y_val = model_generator.read_data("BTC-USD")
