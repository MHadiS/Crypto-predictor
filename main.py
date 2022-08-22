import csv
import pickle as pkl
import pandas as pd
import numpy as np
import requests as r
import argparse as ap
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from utils import ModelGenerator, DataCollector, CURRENCIES


class Main:
    def __init__(self):
        """initializing some variables"""
        self.console = Console()
        parser = ap.ArgumentParser(description="Predict currencies prices")
        parser.add_argument(
            "-du",
            "--dont-update",
            action="store_true",
            help="do not update the models",
        )
        parser.add_argument(
            "-o",
            "--output",
            metavar="file",
            type=str,
            help="output the prediction in a csv file",
        )
        parser.add_argument(
            "-d",
            "--days",
            metavar="days",
            type=int,
            default=7,
            help="number of days after today to predict for",
        )
        parser.add_argument(
            "-p",
            "--plot",
            action="store_true",
            help="make a plot of data and models prediction",
        )
        self.args = parser.parse_args()
        self.model_generator = ModelGenerator()
        self.data_collector = DataCollector()

    @staticmethod
    def has_internet():
        """check internet connection

        Returns:
            bool: if internet connection is available,
            returns True otherwise returns False
        """
        # initializing URL
        url = "https://finance.yahoo.com/cryptocurrencies/"
        timeout = 10
        try:
            # requesting URL
            r.get(url, timeout=timeout)
            return True

        # catching exception
        except r.ConnectionError or r.Timeout:
            return False

    def update_models(self):
        """Update the models"""

        with self.console.status("[bold green] Updating models...") as _:
            self.data_collector.update()
            self.console.log("[cyan] Successfully updated data")
            self.model_generator.update()
            self.console.log("[cyan] Successfully updated models")

    @staticmethod
    def read_currency_data(currency: str):
        """Read a currency data into

        Args:
            currency (str): currency ticker symbol

        Returns:
            pd.DataFrame: the currency price data
        """
        return pd.read_csv(f"utils/data/{currency}.csv")

    def predict_price(self, model, currency: str, days: int):
        """Predict the price of a given currency

        Args:
            model (sklean.linear_model.ElasticNet): the model to predict the-
            price of currency
            currency (str): the currency that we want to predict the price of
            days (int): the days that we want to predict the price of-
            currency(the days after today)
        """

        df = self.read_currency_data(currency)
        current_price = df["Close"].loc[len(df) - 1]
        x = np.array(df.shape[0] + days).reshape(-1, 1)

        prediction = self.model_generator.predict(model, x)[0]
        return current_price, prediction

    def make_plot(self, model, currency: str):
        """Plot the currency price and the model prediction

        Args:
            model (sklean.linear_model.ElasticNet): the model
            currency (str): the currency ticker symbol
        """
        df = self.read_currency_data(currency)
        x = np.arange(0, len(df), 1).reshape(-1, 1)
        y = df["Close"].values.reshape(-1, 1)
        plt.figure(figsize=(20, 10))
        plt.title(f"{currency} plot")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.scatter(x, y)
        plt.plot(df.index, self.model_generator.predict(model, x),
                 color="orange")
        plt.legend(labels=("Data", "Prediction"), loc="upper left")
        plt.savefig(f"{currency}.png")

    def run(self):
        """Run the program"""
        if not self.args.dont_update:
            #  check the internet connection
            connected = self.has_internet()
            if not connected:
                self.console.log("[red bold] No internet connection available")
                exit()

            self.update_models()

        # make a table of predictions for each currencies
        table = Table(title=f"Table of predictions for {self.args.days} days later")
        columns = ["Currencies", "Predicted Price", "Currrent Price", "Change"]
        for column in columns:
            table.add_column(column)

        # if output is not None, write the header of csv file
        if self.args.output is not None:
            file = open(f"./{self.args.output}", "a")
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

        # add the rows
        for currency in CURRENCIES:

            # load the models
            with open(f"utils/models/{currency}.model", "rb") as f:
                model = pkl.load(f)

            # make the predictions for the currency
            current_price, prediction = self.predict_price(
                model, currency, self.args.days
            )

            # calculate the currency change
            change = prediction - current_price

            # add row to the table
            row = [currency, str(prediction), str(current_price), change]
            if self.args.output is not None:
                writer.writerow({columns[i]: row[i] for i in range(len(row))})

            if change > 0:
                row[-1] = f"[green]+{change}"

            elif change < 0:
                row[-1] = f"[red]{change}"

            else:
                row[-1] = "0"
            table.add_row(*row)

            # export the currency price and model prediction plot
            if self.args.plot:
                self.make_plot(model, currency)

        print("-" * 90)

        try:
            # close the file
            file.close()
            self.console.log(
                f"[bold green]{self.args.output} successfully exported"
            )
        except NameError:
            # print table
            self.console.log(table)


if __name__ == "__main__":
    app = Main()
    app.run()
