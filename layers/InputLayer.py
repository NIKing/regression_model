import numpy as np

class InputLayer():
    def __init__(self, input_dim = 6, output_dim= 6):
        self.output = None

        self.output_dim = output_dim
        self.input_dim = input_dim

    def __call__(self, inputs):
        self.output = np.array(inputs)

        return [inputs * self.output_dim for i in range(self.input_dim)]


