# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

from torch import nn 
import torch.nn.functional as F 

    
class Color(nn.Module):
 
    def __init__(self, in_dim=32000, num_class=256):
        super(Color, self).__init__() 
        self.fc = nn.Linear(32000, 1000)
        self.fc0 = nn.Linear(1000, 256)
        self.fc_R = nn.Linear(256, 256)
        self.fc_G = nn.Linear(256, 256)
        self.fc_B = nn.Linear(256, 256)
        self.fc_R1 = nn.Linear(256, 64)
        self.fc_G1 = nn.Linear(256, 64)
        self.fc_B1 = nn.Linear(256, 64)
        self.fc_R2 = nn.Linear(64, 64)
        self.fc_G2 = nn.Linear(64, 64)
        self.fc_B2 = nn.Linear(64, 64)
        self.fc_R3 = nn.Linear(64, 256)
        self.fc_G3 = nn.Linear(64, 256)
        self.fc_B3 = nn.Linear(64, 256) 

    def forward(self, x): 
        x = F.normalize(x, p=2, dim=-1)
        x = self.fc(x) 
        
        x = self.fc0(x) 
        
        R_out = self.fc_R(x)
        G_out = self.fc_G(x)
        B_out = self.fc_B(x)
        
        R_out = self.fc_R1(R_out)
        G_out = self.fc_G1(G_out)
        B_out = self.fc_B1(B_out) 
        
        R_out = self.fc_R2(R_out)
        G_out = self.fc_G2(G_out)
        B_out = self.fc_B2(B_out)
        
        R_out = self.fc_R3(R_out)
        G_out = self.fc_G3(G_out)
        B_out = self.fc_B3(B_out)  
        return R_out, G_out, B_out
