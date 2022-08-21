import pandas as pd
import numpy as np
import pickle as pkl
import utils.settings as s
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class ModelGenerator:
    """Generate the models for making predictions"""

    def read_data(self, ticker: str) -> tuple:
        """Read the data about a currency and split ti training testing and validation set

        Args:
            ticker (str): the currency ticker

        Returns:
            tuple: a tuple of the sets
        """
        df = pd.read_csv(f"utils/data/{ticker}.csv")
        X = np.array([i for i in range(df.shape[0])]).reshape(-1, 1)
        self.degree = 10
        self.poly_reg = PolynomialFeatures(degree=self.degree)
        X_poly = self.poly_reg.fit_transform(X)
        y = df["Close"].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, shuffle=False
        )
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1)
        return X_train, X_test, X_val, y_train, y_test, y_val

    def generate_model(self, currency: str):
        """Generate a model for a specific currency

        Args:
            currency (str): the currency ticker

        Returns:
            Unknown for now: the model
        """
        X_train, X_test, X_val, y_train, y_test, y_val = self.read_data(currency)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def export_model(self, model: any, currency: str):
        """Export the generated model

        Args:
            model (Unknown for now): the generated model
            currency (str): the currency ticker
        """
        with open(f"utils/models/{currency}.model", "wb") as f:
            pkl.dump(model, f)
    
    def predict(self, model, x: np.array) -> np.array:
        """Make prediction with the model model

        Args:
            model (Unknown for now): the model
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
