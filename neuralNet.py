import numpy as np
from layer import Layer
from extraFuncLib import identity, step

class Network:
    def __init__(self, featureSize, size=10, layerSize=100, activation=0, activationDerivative=0):
        self.featureSize = featureSize
        self.size = size
        self.layers = np.zeros(size, dtype=Layer)
        self.layers[0] = Layer(layerSize, identity)

        if activation:
            print("Using given activation function")
            for i in range(1, size-1):
                self.layers[i] = Layer(layerSize, activation)
            self.activationDerivative = activationDerivative
        else:
            print("Using ReLU as activation function")
            for i in range(1, size-1):
                self.layers[i] = Layer(layerSize)
            self.activationDerivative = step
        #  layer should be a linear combination hence identity function
        self.layers[-1] = Layer(layerSize, identity)

    def iterate(self, image, label):
        z, a = self.forwardPropagate(image)
        self.backPropagate(z, a, label)
        pass

    def forwardPropagate(self, image):
        
        # converts image into a row vector and iterates on the layers
        # using previous output as the next input
        image = np.array(image)
        image = np.flatten(image)

        if np.size(image) != self.featureSize:
            raise ValueError("Input size and layer size do not match")

        a = np.zeros([self.size, np.shape(image)])
        z = np.zeros(np.size(a))

        # these are input layer outputs
        z[0] = image
        a[0] = image
        # outputs of the rest of the layers
        for i in range(1, self.size):
            z[i], a[i] = self.layers[i].process(a[i-1])

        return z, a
            
    def backPropagate(self, z, a, label):
        # output layer
        # for the last layer prediction a^L is equivalent to z^L
        # hence we can use sigma'(a^L) instead of sigma'(z^L)
        delta = np.multiply(prediction-label, step(z))

    def predict(self, image):
        pass