import torch
from torch import nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_ch, hid_ch)
        self.fc2 = torch.nn.Linear(hid_ch, out_ch)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.uniform(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.uniform(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
