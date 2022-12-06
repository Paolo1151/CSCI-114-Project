import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.input_1 = nn.Linear(input_dim, 100)
        self.input_2 = nn.Linear(100, 50)
        self.input_3 = nn.Linear(50, 25)
        self.output = nn.Linear(25, output_dim)

    def forward(self, x):
        x = F.relu(self.input_1(x))
        x = F.relu(self.input_2(x))
        x = F.relu(self.input_3(x))
        y = torch.sigmoid(self.output(x))
        y = F.softmax(y, dim=-1)

        return y