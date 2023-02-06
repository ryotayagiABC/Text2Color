from torch.utils.data import Dataset, DataLoader

import torch
import pandas as pd
from transformers import AlbertTokenizer, AlbertForPreTraining
import time
import numpy as np
import torchvision.transforms as transforms

class MY_DATA_SET(Dataset):
    def __init__(self, transform=None):
        TRAIN_DATA_NAME = "./data.csv"
        df = pd.read_csv(TRAIN_DATA_NAME, encoding="Shift-JIS")
        
        self.X = df["color"].values
        self.R = df["R"].values 
        self.G = df["G"].values 
        self.B = df["B"].values
        self.tokenizer = AlbertTokenizer.from_pretrained('ALINEAR/albert-japanese-v2')
        self.model1 = AlbertForPreTraining.from_pretrained('ALINEAR/albert-japanese-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.tokenizer.encode(self.X[index], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        # EMBEDDING MODEL
        X = self.model1(input_ids)
        X = X.prediction_logits
        X = torch.sum(X, 1).squeeze(1).squeeze(0)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  
        
        n_labels = 256
        r = torch.tensor(self.R[index], dtype=torch.float32).to(self.device)
        g = torch.tensor(self.G[index], dtype=torch.float32).to(self.device)
        b = torch.tensor(self.B[index], dtype=torch.float32).to(self.device)
        R = np.eye(n_labels)[self.R[index]]
        R = torch.tensor(R).to(self.device)
        G = np.eye(n_labels)[self.G[index]]
        G = torch.tensor(G).to(self.device)
        B = np.eye(n_labels)[self.B[index]]
        B = torch.tensor(B).to(self.device)

        return X, R, G, B, r, g, b