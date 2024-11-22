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