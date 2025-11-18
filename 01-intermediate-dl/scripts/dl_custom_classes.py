import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
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
        return (
            torch.tensor(features).float(),
            torch.tensor(label).float()
        )

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(9, 16) # (9 + 1) * 16 = 160 parameters
        init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu') # Weight initialization
        self.bnlayer1 = nn.BatchNorm1d(16) # Batch normalization
        
        self.layer2 = nn.Linear(16, 8) # (16 + 1) * 8 = 136
        init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        self.bnlayer2 = nn.BatchNorm1d(8)

        self.layer3 = nn.Linear(8, 1) # (8 + 1) * 1 = 9
        init.kaiming_uniform_(self.layer3.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = self.layer1(x)
        x = self.bnlayer1(x)
        x = F.elu(x)
        
        x = self.layer2(x)
        x = self.bnlayer2(x)
        x = F.elu(x)

        x = self.layer3(x)
        x = F.sigmoid(x)
        return x