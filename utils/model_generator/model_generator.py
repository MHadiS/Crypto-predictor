import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class ModelGenerator:
    """Generate the models for making predictions
    """
    def read_data(self, ticker: str) -> tuple:
        """Read the data about a currency and split ti training testing and validation set

        Args:
            ticker (str): the currency ticker

        Returns:
            tuple: a tuple of the sets
        """
        df = pd.read_csv(f"utils/data/{ticker}.csv")
        X = np.array([i for i in range(df.shape[0])]).reshape(-1, 1)
        y = df["Close"].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1)
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def generate_model(self, currency: str):
        """ Generate a model for a specific currency

        Args:
            currency (str): the currency ticker

        Returns:
            DecisionTreeRegressor: the model
        """
        X_train, X_test, X_val, y_train, y_test, y_val = self.read_data(currency)
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        return model
    
    def export_model(self, model: DecisionTreeRegressor, currency: str):
        """Export the generated model

        Args:
            model (DecisionTreeRegressor): the generated model
            currency (str): the currency ticker
        """
        with open(f"utils/models/{currency}.csv", "wb") as f:
            pkl.dump(model, f)
