import pytest as pt
import pickle as pkl
import numpy as np
from os import remove

from utils import ModelGenerator


@pt.fixture
def model_generator():
    return ModelGenerator()


def model_score(model, X_test, X_val, y_test, y_val):
    """Score the model based on the given test sets.
    Args:
        model (Unknown for now): the model to be tested.

    Returns:
        float: the score of the model
    """

    score1 = model.score(X_test, y_test)
    score2 = model.score(X_val, y_val)
    return (score1 + score2) / 2


def test_read_data(model_generator):
    global X_test, X_val, y_test, y_val
    X_train, X_test, X_val, y_train, y_test, y_val = model_generator.read_data(
        "BTC-USD"
    )


def test_generate_model(model_generator):
    global model
    model = model_generator.generate_model("BTC-USD")
    score = model_score(model, X_test, X_val, y_test, y_val)
    assert score > 0.9


def test_export_model(model_generator):
    model_generator.export_model(model, "test")
    file_path = "utils/models/test.model"
    with open(file_path, "rb") as f:
        model2 = pkl.load(f)
    remove(file_path)
    x = np.array([i for i in range(10)]).reshape(-1, 1)
    assert model_generator.predict(model, x).all() == model_generator.predict(model2, x).all()
