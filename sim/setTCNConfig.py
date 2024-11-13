class TCNConfig:
    def __init__(self):
        self.input_size = 4
        self.output_size = 2
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.0
        self.num_channels = [32] # [8, 16]

    def getTCNConfig(self):
        return self.input_size, self.output_size, self.kernel_size, self.stride, self.dropout, self.num_channels
