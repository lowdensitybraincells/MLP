import numpy as np


def ReLU(x):
    x = np.array(x)
    return np.multiply(x, x>0, dtype=np.float32)

class Layer:
    def __init__(self, size, activation = ReLU):
        self.size = size
        self.activation = activation
        self.bias = np.zeros((size, 1))
        self.weights = np.random.rand(size, size)

    def process(self, x):
        z = np.matmul(self.weights, x) + self.bias
        a = self.activation(z)
        return a 

    def updateParams(self, newWeights, newBias):
        self.weights = newWeights
        self.bias = newBias 