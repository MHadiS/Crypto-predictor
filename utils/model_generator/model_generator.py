import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

import utils.settings as s


class ModelGenerator:
    """Generate the models for making predictions"""

    def __init__(self) -> None:
        """Initialize some variables"""
        self.poly_reg = PolynomialFeatures(degree=20)

    def read_data(self, ticker: str) -> tuple:
        """Read the data about a currency and split it-
            training testing and validation set

        Args:
            ticker (str): the currency ticker

        Returns:
            tuple: a tuple of the sets
        """
        df = pd.read_csv(f"utils/data/{ticker}.csv")
        X = np.array([i for i in range(df.shape[0])]).reshape(-1, 1)

        X_poly = self.poly_reg.fit_transform(X)
        y = df["Close"].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.1
        )
        return X_train, X_test, X_val, y_train, y_test, y_val

    def generate_model(self, currency: str):
        """Generate a model for a specific currency

        Args:
            currency (str): the currency ticker

        Returns:
            sklean.linear_model.ElasticNet: the model
        """
        X_train, _, _, y_train, _, _ = self.read_data(currency)
        model = ElasticNet()
        model.fit(X_train, y_train)
        return model

    def export_model(self, model: ElasticNet, currency: str):
        """Export the generated model

        Args:
            model (sklean.linear_model.ElasticNet): the generated model
            currency (str): the currency ticker
        """
        with open(f"utils/models/{currency}.model", "wb") as f:
            pkl.dump(model, f)

    def predict(self, model: ElasticNet, x: np.array) -> np.array:
        """Make prediction with the model model

        Args:
            model (sklean.linear_model.ElasticNet): the model
            x (np.array): the input of the model

        Returns:
            np.array: an array of the model predictions
        """
        x_poly = self.poly_reg.fit_transform(x)
        return model.predict(x_poly)

    def update(self):
        """Update the models with the new data"""
        for currency in s.CURRENCIES:
            model = self.generate_model(currency)
            self.export_model(model, currency)
