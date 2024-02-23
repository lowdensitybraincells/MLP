import numpy as np
from extraFuncLib import ReLU
class Layer:
    def __init__(self, size, activation = ReLU):
        self.size = size
        self.activation = activation
        self.bias = np.zeros((size, 1))
        self.weights = np.random.rand(size, size)

    def process(self, x):
        z = np.matmul(self.weights, x) + self.bias
        a = self.activation(z)
        return z, a 

    def updateParams(self, newWeights, newBias):
        self.weights = newWeights
        self.bias = newBias 