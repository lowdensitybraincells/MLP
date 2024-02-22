import numpy as np
from layer import Layer


def identity(x):
    return x


class Network:
    def __init__(self, featureSize, size=10, layerSize = 100, activation=0):
        self.featureSize = featureSize
        self.size = size
        self.layers = np.zeros(size, layerSize, layerSize)
        if activation:
            print("Using given activation function")
            for i in range(layerSize-1):
                self.layers[i] = Layer(i, activation)
        else:
            print("Using ReLU as activation function")
            for i in range(layerSize-1):
                self.layers[i] = Layer(i)
        # last layer should be a linear combination hence identity function
        self.layers[-1] = Layer(i, identity)

    def iterate(self, image, label):
        self.forwardPropagate(image)
        self.backPropagate(label)
        pass

    def forwardPropagate(self, image):
        
        # converts image into a row vector and iterates on the layers
        # using previous output as the next input
        a = np.array(image)
        a = np.flatten(a)
        
        if np.size(a) != self.featureSize:
            raise ValueError("Input size and layer size do not match")

        for i in range(self.size):
            a = self.layers[i].process(a)
            
    def backPropagate(self):
        pass

    def predict(self, image):
        pass