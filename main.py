import pickle as pkl
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import requests as r

from utils import ModelGenerator, DataCollector, CURRENCIES


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



def main():
    """Run the program.
    """
    console = Console()

    #  check the internet connection
    connected = has_internet()
    if not connected:
        console.log("[red bold] No internet connection available")
        exit()
    
    # loading for updating the models
    model_generator = ModelGenerator()
    data_collector = DataCollector()
    with console.status("[bold green] Updating models...") as status:
        data_collector.update()
        console.log("[cyan] Successfully updated data")
        model_generator.update()
        console.log("[cyan] Successfully updated models")

    # make a table of predictions for each currencies
    table = Table(title="Table of predictions")
    table.add_column("Currencies")
    table.add_column("Price")
    for currency in CURRENCIES:

        # load the models
        with open(f"utils/models/{currency}.model", "rb") as f:
            model = pkl.load(f)
        
        # add row to table
        df = pd.read_csv(f"utils/data/{currency}.csv")
        x = np.array(df.shape[0] + 7).reshape(-1, 1)
        prediction = model.predict(x)[0]
        table.add_row(currency, str(prediction))
    
    # print table
    print("-" * 80)
    console.log(table)
    

if __name__ == '__main__':
    main()