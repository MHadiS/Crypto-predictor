import pytest as pt
import pickle as pkl
import numpy as np
from os import remove

from utils import ModelGenerator


@pt.fixture
def model_generator():
    return ModelGenerator()


def test_read_data(model_generator): 
    global X_train, X_test, X_val, y_train, y_test, y_val
    X_train, X_test, X_val, y_train, y_test, y_val = model_generator.read_data("BTC-USD")


def test_generate_model(model_generator):
    global model
    model = model_generator.generate_model("BTC-USD")
    score = model.score(X_test, y_test)
    assert int(score * 100) > 90


def test_export_model(model_generator):
    model_generator.export_model(model, "test")
    file_path = "utils/models/test.model"
    with open(file_path, "rb") as f:
        model2 = pkl.load(f)
    remove(file_path)
    x = np.array([i for i in range(10)]).reshape(-1, 1)
    assert model.predict(x).all() == model2.predict(x).all()
