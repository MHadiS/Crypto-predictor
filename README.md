# CryptoPredictor : a crypto currency price predictor

## What my project does

This project predicts the price of crypto currencies with a polynomial regression algorithm.

## Contents

- [Desecration](#description)
  - [How my project works ?](#how-does-my-project-work)
  - [Why I used polynomial regression ?](#why-i-used-polynomial-regression)
- [Install Requirements](#install-requirements)
- [Usage](#usage)
  - [How to use the project ?](#how-to-use-the-project)
  - [How to run the tests ?](#how-to-run-the-tests)

## Description

### How does my project work

My project is divided into three parts `data_collector` `model_generator` and a `main.py` file that puts all of the abilities together and runs the program.

`data_collector` has to collect the currencies price data save the data as a csv file in utils/data directory and update the data. I used yfinance api to collect the data.

`model_generator` has to build the models with accuracy higher than 80% and export them to a file. I used sklearn to build the models and pickle to export them. I added elastic-net regularization because for some reason I explained in [this](#why-i-used-polynomial-regression) part.

`main.py` runs the program and shows the prediction in terminal with rich library.

### Why I used polynomial regression

Now I'm just a student in machine learning. I have just learned some regression algorithms and I want to write a project with the new things I learned.

I used polynomial regression because the data wasn't always linear but there was a big problem. The data changes with each update (after many many updates, a noticeable part of the data will change). It means if an 8 degree model is good for now maybe after 2000 updates the 8 degree model is not good for data so I used a 20 degree model and applied an elastic-net regularization to prevent it from over fitting.

## Install Requirements

1. python version 3 or higher
2. some libraries that you can install by running `install_dependencies.py`. if you are curious this list is for you.

```text
pandas
numpy
sklearn
pickle5
rich
requests
argparse
yfinance
pytest
```

## Usage

### How to use the project

For running the program run `main.py`

```command
python main.py
```

I add some useful options that you can see with using `-h` or `--help` option

```command
python main.py -h
```

Output:

```text
usage: main.py [-h] [-du] [-o file] [-d days] [-p]

Predict currencies prices

optional arguments:
  -h, --help            show this help message and exit
  -du, --dont-update    do not update the models
  -o file, --output file
                        output the prediction in a csv file
  -d days, --days days  number of days after today to predict for
  -p, --plot            make a plot of data and models prediction
```

for predicting other currencies price just add the ticker symbol of them to CURRENCIES constant in utils/setting.py

### How to run the tests

It's easy just use pytest and run this command:

```command
pytest tests
```

if you want you can run test with flack8

Flake8:

```command
flake8 .
```
