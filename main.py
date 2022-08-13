import csv
import pickle as pkl
import pandas as pd
import numpy as np
import requests as r
import argparse as ap
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
        self.args = parser.parse_args()
        self.model_generator = ModelGenerator()
        self.data_collector = DataCollector()

    @staticmethod
    def has_internet():
        """check internet connection

        Returns:
            bool: if internet connection is available, returns True otherwise returns False
        """
        # initializing URL
        url = "https://finance.yahoo.com/cryptocurrencies/"
        timeout = 10
        try:
            # requesting URL
            request = r.get(url, timeout=timeout)
            return True

        # catching exception
        except (r.ConnectionError, r.Timeout) as exception:
            return False

    def update_models(self):
        """Update the models"""

        with self.console.status("[bold green] Updating models...") as status:
            self.data_collector.update()
            self.console.log("[cyan] Successfully updated data")
            self.model_generator.update()
            self.console.log("[cyan] Successfully updated models")

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
        table = Table(title="Table of predictions")
        columns = ["Currencies", "Price", "Change"]
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
            current_price, prediction = self.model_generator.predict_price(
                model, currency, self.args.days
            )

            # calculate the currency change
            change = current_price - prediction

            # add row to the table
            row = [currency, str(prediction), change]
            if self.args.output is not None:
                writer.writerow({columns[i]: row[i] for i in range(len(row))})

            if change > 0:
                row[-1] = f"[green]+{change}"
            elif change < 0:
                row[-1] = f"[red]{change}"
            else:
                row[-1] = "0"
            table.add_row(*row)

        print("-" * 80)

        try:
            # close the file
            file.close()
            self.console.log(f"[bold green]{self.args.output} successfully exported")
        except NameError:
            # print table
            self.console.log(table)


if __name__ == "__main__":
    app = Main()
    app.run()
