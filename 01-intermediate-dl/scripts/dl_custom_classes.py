import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss
import polars as pl

class WaterDataset (Dataset):

    def __init__(self, path: str):
        super().__init__()
        self.df = pl.read_csv(source=path)
        self.data = self.df.to_numpy()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        features = self.data[index, :-1]
        label = self.data[index, -1]
        return features, label

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(9, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.sigmoid(self.layer3(x))
        return x