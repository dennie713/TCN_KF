import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# chatgpt 提供的code
# class Chomp1d(nn.Module):
#     """消除卷積後的padding區域."""
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()

# class TemporalBlock(nn.Module):
#     """TCN 的基本構建塊，包括擴展卷積、激活和殘差連接."""
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)

# class TemporalConvNet(nn.Module):
#     """TCN 的完整網絡，由多個 TemporalBlock 組成."""
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.network(x)

#--------------------------------------------------------------------------------#
# paper作者提供的code
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        # self.relu1 = nn.ReLU()
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        # self.relu2 = nn.ReLU()
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size, stride, dropout): # num_classes是為了調整output size增加的
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # 調整dilation_rate
        # dilation_rate = [1, 4, 16, 64]
        dilation_rate = [4]
        for i in range(num_levels): 
            dilation_size = 2 ** i
            """dilation_size"""
            # dilation_size = dilation_rate[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # 增加全連接層以設定輸出類別數
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = self.network(x)
        # x = x.mean(dim=2)  # stride=2時需要用 global average pooling
        # 使用最後一個時間步的輸出進行分類
        x = self.fc(x[:, :, -1])  # 取出最後一個時間步的輸出

        # 確保輸出為正
        x = torch.exp(x)  # 確保輸出正數（用來當共變異的對角項）
        return x
        # return self.network(x) # 為了調整output size註解的

"""
模仿PAPER中LSTM-Q和LSTM-R的模型
"An Improved Kalman Filter Based on Long Short-MemoryRecurrent Neural Network for Nonlinear Radar Target Tracking"
"""
# 用於學習Q值的TCN模型
class TCN_Q(nn.Module):
    def __init__(self, input_dim, output_dim=1, kernel_size=4):
        super(TCN_Q, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=256,
                      kernel_size=kernel_size,
                      padding=kernel_size - 1,  # causal padding
                      dilation=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # → (batch, features, seq_len)
        y = self.tcn(x)
        y = y[:, :, -1]        # 取最後一個時間步的輸出 (batch, 256)
        q_diag = self.fc(y)    # (batch, 2)
        q_diag = torch.exp(q_diag)  # 確保輸出正數（用來當共變異的對角項）
        return q_diag
    
# 用於學習R值的TCN模型
class TCN_R(nn.Module):
    def __init__(self, input_dim, output_dim=2, kernel_size=3):
        super(TCN_R, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=128,
                      kernel_size=kernel_size,
                      padding=kernel_size - 1,  # causal padding
                      dilation=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: shape (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)         # → (batch, input_dim, seq_len)
        y = self.tcn(x)               # → (batch, 128, seq_len)
        y = y[:, :, -1]               # 取最後一個時間步 → (batch, 128)
        r_diag = torch.exp(self.fc(y))  # 輸出為正數 → (batch, 2)
        return r_diag