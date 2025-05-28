import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12)))  # 避免 log(0)
    
import torch.nn.functional as F

class StableLogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(diff + F.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0)))