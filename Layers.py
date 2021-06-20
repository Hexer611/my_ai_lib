import numpy as np


class Dense_Layer:
    def __init__(self, input_size, neuron_size):
        self.weights = np.random.uniform(-1.0, 1.0, (input_size, neuron_size))
        self.biases = np.zeros((1, neuron_size)).astype(float)
        self.output = -1

    def f_prop(self, inp):
        self.output = np.dot(inp, self.weights) + self.biases
