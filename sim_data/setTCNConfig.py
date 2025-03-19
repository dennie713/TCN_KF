class TCNConfig:
    def __init__(self):
<<<<<<< HEAD
        self.input_size = 7
        self.output_size = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.0
        self.num_channels = [32] # [8, 16]
=======
        self.input_size = 4
        self.output_size = 3
        self.kernel_size = 6
        self.stride = 1
        self.dropout = 0.2
        self.num_channels = [4, 64, 128, 3] # [4, 64, 128, 3] # [8, 16]
>>>>>>> 924f379 (v3)

    def getTCNConfig(self):
        return self.input_size, self.output_size, self.kernel_size, self.stride, self.dropout, self.num_channels
