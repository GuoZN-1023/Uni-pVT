import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ZDataset(Dataset):
    def __init__(self, csv_path, scaler_path=None, train=True):
        df = pd.read_csv(csv_path)
        feature_cols = ["T reduced", "p reduced", "omega", "p critical", "T critical", "T triple", "dipole moment"]
        self.X = df[feature_cols].values
        self.y = df["Z"].values.reshape(-1, 1)

        if train:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            if scaler_path:
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(self.scaler, scaler_path)
        else:
            if not scaler_path or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            self.X = self.scaler.transform(self.X)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(cfg):
    set_seed(42)
    dataset = ZDataset(cfg["paths"]["data"], cfg["paths"]["scaler"], train=True)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=g)
    return {
        "train": DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True),
        "val": DataLoader(val_set, batch_size=cfg["training"]["batch_size"]),
        "test": DataLoader(test_set, batch_size=cfg["training"]["batch_size"]),
    }