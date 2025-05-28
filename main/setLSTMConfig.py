class LSTMConfig:
    def __init__(self):

        self.x_input_size = 3
        self.x_output_size = 1
        # self.P_input_size = 18
        # self.P_output_size = 9

        self.hidden_size = 256
        self.num_layers = 1
        self.dropout = 0.2

    def getLSTMConfig(self):
        return self.x_input_size, self.x_output_size, self.hidden_size, self.num_layers, self.dropout
