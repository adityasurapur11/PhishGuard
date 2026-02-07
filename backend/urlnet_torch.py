import torch
import torch.nn as nn

class URLNet(nn.Module):
    def __init__(self, input_dim=15):
        super(URLNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) # 3 Classes: 0=Safe, 1=Phishing, 2=Suspicious

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Output raw logits for 3 classes
        x = self.fc3(x) 

        return x