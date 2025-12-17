import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torchvision import models
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

class MyCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, # Three channels: RGB 
                out_channels=32, # Number of filters, 32 filters
                kernel_size=3, # 3 x 3 filter size
                stride=1, # Shift 1 by 1
                padding=1 # Adds one layer of zeros around the border
            ), # Out: 32 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Out: 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Out: 64 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64 x 32 x 32

            nn.Flatten() 
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*32*32, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

class TransferResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # Loads ResNet-18 and use weights trained on ImageNet (1K classes)
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the final classification layer
        in_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feats, num_classes)
        
        # Freeze all layers except the classifier
        for name, p in self.net.named_parameters():
            p.requires_grad = name.startswith("fc")

    def forward(self, x):
        return self.net(x)

class FineTuneResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, pct_unfreeze=0.25):
        super().__init__()
        pct_unfreeze = float(max(0.0, min(1.0, pct_unfreeze)))
        self.net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feats, num_classes)

        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.net.fc.parameters():
            p.requires_grad = True

        group_names = ["layer4", "layer3", "layer2", "layer1", "bn1", "conv1"]
        total_groups = len(group_names)
        k = int(round(pct_unfreeze * total_groups))
        to_unfreeze = group_names[ :k]

        for name, module in self.net.named_modules():
            if any(name == g or name.startswith(g + ".") for g in to_unfreeze):
                for p in module.parameters(recurse=True):
                    p.requires_grad = True

    def forward(self, x):
        return self.net(x)
