import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


class StockModel(nn.Module):
    n_features = 5
    n_layers = 1
    
    def __init__(self, n_windows: int = 64, output_size: int = 8):
        super(StockModel, self).__init__()
        self.lstm = nn.LSTM(self.n_features, n_windows, self.n_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_windows, n_windows)
        self.fc2 = nn.Linear(n_windows, n_windows)
        self.fc3 = nn.Linear(n_windows, n_windows)
        self.fc4 = nn.Linear(n_windows, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

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
    data = ticker.history(interval=interval, period=period)
    data = data.drop(["Dividends", "Stock Splits"], axis=1)
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


def load_kv_stock_v1(path: str):
    param_dict = torch.load(path)

    model = StockModel()
    model.load_state_dict(param_dict)
    model.eval()
    return model


def load_kv_stock_v2(path: str):
    param_dict = torch.load(path, map_location=torch.device('cpu'))

    model = StockModel(n_windows=128, output_size=32)
    model.load_state_dict(param_dict)
    model.eval()
    return model


def kv_stock_v1_predict(model, data, close_data):
    input = torch.tensor(data[-64:].reshape(1, 64, 5), dtype=torch.float32)
    output = model(input).detach().numpy().reshape(8,)

    for i in range(len(output)):
        output[i] = (1 + output[i]) * close_data
    return output


def kv_stock_v1_plot(ax, prediction, size):
    future_x = np.arange(size, size + 8)
    future_y = prediction 

    ax[0].plot(future_x, future_y, label="kv_stock_v1")
    ax[0].legend()


def kv_stock_v2_predict(model, data, close_data):
    input = torch.tensor(data[-128:].reshape(1, 128, 5), dtype=torch.float32)
    output = model(input).detach().numpy().reshape(32,)

    for i in range(len(output)):
        output[i] = (1 + output[i]) * close_data
    return output


def kv_stock_v2_plot(ax, prediction, size):
    future_x = np.arange(size, size + 32)
    future_y = prediction 

    ax[0].plot(future_x, future_y, label="kv_stock_v2")
    ax[0].legend()


def predict_multiple(model, data, close, input_size: int, output_size: int, k:int = 64):
    n_columns = len(data[0])
    x = np.zeros((k, input_size, n_columns))

    for i in range(k):
        x[i] = data[-input_size-k-i:-k-i]

    return x