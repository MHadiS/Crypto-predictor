import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
        X = np.array([i for i in range(df.shape[0])])
        y = df["Close"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1)
        return X_train, X_test, X_val, y_train, y_test, y_val
