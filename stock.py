import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x, y

def get_stock_data_as_numpy(name: str):
    ticker = yf.Ticker(name)
    data = ticker.history(interval="1d", period="max")
    data = data.reset_index()
    data = data.drop(["Date", "Dividends", "Stock Splits"], axis=1)
    data = data[(data != 0).all(axis=1)]
    data = data.dropna()
    data = data.to_numpy()
    return data

def clean_data(data):
    # assume already have stock data
    data = data.reset_index()
    data = data.drop

    # normalization
    data = data.pct_change()
    data = data.dropna()
    data = data.to_numpy()

    return data

def process_data(data, n_windows: int = 64, n_futures = 1):
    n_columns = len(data[0])
    size = len(data)
    
    X = np.zeros((size - n_windows, n_windows, n_columns))
    y = np.zeros((size - n_windows, n_columns - 1))
    
    for i in range(size - n_windows):
        X[i] = data[i : i + n_windows]
        y[i] = data[i + n_windows, 0: n_columns - 1]

    return X, y

def load_stock_data(name: str, interval: str = "1d", period: str = "max"):
    # import data from yfinance 
    ticker = yf.Ticker(name)
    data = ticker.history(interval="1d", period="max")

    # clean data
    data = clean_data(data)
    X, y = process_data(data)
    return X, y

def load_raw_stock_data(name: str, interval: str = "1d", period: str = "max", drop_volume=False):
    ticker = yf.Ticker(name)
    data = ticker.history(interval="1d", period="max")
    data = data.reset_index()
    data = data.drop(["Date", "Dividends", "Stock Splits"], axis=1)
    if drop_volume : data = data.drop(["Volume"], axis=1)
    data = data[(data != 0).all(axis=1)]
    return data

def stock_data_loader(name, split: float = 0.8, batch_size: int = 64):
    # load data
    X, y = load_stock_data(name)
    
    # dataset
    dataset = StockDataset(X, y)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [split, 1 - split])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

