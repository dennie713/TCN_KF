class TCNConfig:
    def __init__(self):
        self.input_size = 3
        self.output_size = 1
        self.kernel_size = 6
        self.stride = 1
        self.dropout = 0.2
        self.num_channels = [256] # [4, 64, 128, 3] # [8, 16]

    def getTCNConfig(self):
        return self.input_size, self.output_size, self.kernel_size, self.stride, self.dropout, self.num_channels
