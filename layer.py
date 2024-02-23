import numpy as np
from extraFuncLib import ReLU
class Layer:
    def __init__(self, nodeCount, paramCount, activation = ReLU):
        self.paramCount = paramCount
        self.nodeCount = nodeCount
        self.activation = activation
        self.bias = np.zeros((nodeCount, 1))
        self.weights = np.random.rand(nodeCount, paramCount)
        print(np.shape(self.weights))


    def process(self, x):

        print(np.shape(self.weights), np.shape(self.bias))
        z = np.matmul(self.weights, x) + self.bias
        a = self.activation(z)

        return z, a 

    def updateParams(self, newWeights, newBias):
        self.weights = newWeights
        self.bias = newBias 

    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias