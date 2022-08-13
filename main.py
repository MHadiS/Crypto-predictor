import pickle as pkl
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import requests as r

from utils import ModelGenerator, DataCollector, CURRENCIES


class Main:
    def __init__(self):
        self.console = Console()


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

    def run(self):
        """Run the program."""
        #  check the internet connection
        connected = self.has_internet()
        if not connected:
            self.console.log("[red bold] No internet connection available")
            exit()

        # loading for updating the models
        model_generator = ModelGenerator()
        data_collector = DataCollector()
        with self.console.status("[bold green] Updating models...") as status:
            data_collector.update()
            self.console.log("[cyan] Successfully updated data")
            model_generator.update()
            self.console.log("[cyan] Successfully updated models")

        # make a table of predictions for each currencies
        table = Table(title="Table of predictions")
        table.add_column("Currencies")
        table.add_column("Price")
        table.add_column("Change")
        for currency in CURRENCIES:

            # load the models
            with open(f"utils/models/{currency}.model", "rb") as f:
                model = pkl.load(f)

            # read the currency data
            df = pd.read_csv(f"utils/data/{currency}.csv")

            # find current price
            current_price = df["Close"].loc[len(df) - 1]

            # predict the currency price
            x = np.array(df.shape[0] + 7).reshape(-1, 1)
            prediction = model.predict(x)[0]

            # calculate the currency change
            change = current_price - prediction

            # add row to the table
            row = [currency, str(prediction)]
            if change > 0:
                row.append(f"[green]+{change}")
            elif change < 0:
                row.append(f"[red]{change}")
            else:
                row.append("0")
            table.add_row(*row)

        # print table
        print("-" * 80)
        self.console.log(table)


if __name__ == "__main__":
    app = Main()
    app.run()
