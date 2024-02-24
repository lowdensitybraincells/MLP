import numpy as np
from extraFuncLib import ReLU
class Layer:
    def __init__(self, nodeCount, paramCount, activation = ReLU):
        self.paramCount = paramCount
        self.nodeCount = nodeCount
        self.activation = activation
        self.bias = np.zeros((nodeCount, 1))
        self.weights = np.random.rand(nodeCount, paramCount)

    def process(self, x):
        x = np.squeeze(x)
        x = np.expand_dims(x, axis=1)
        z = np.matmul(self.weights, x) + self.bias
        a = self.activation(z)

        return np.squeeze(z), np.squeeze(a) 

    def updateParams(self, newWeights, newBias):
        self.weights = newWeights
        self.bias = newBias 

    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias