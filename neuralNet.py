import numpy as np
from layer import Layer
from extraFuncLib import identity, step

class Network:
    def __init__(self, data, labels, size=10, layerSize=100, paramCount=25, activation=0, activationDerivative=0):
        
        self.data = data
        self.labels = labels
        self.size = size
        self.layerSize = layerSize
        self.featureSize = np.size(data[0])
        self.predictionSize = np.size(labels[0])
        self.paramCount = paramCount

        self.inputLayer = Layer(self.paramCount, self.featureSize, identity)
        self.outputLayer = Layer(self.predictionSize, self.paramCount, identity)

        self.hiddenLayers = np.zeros(size, dtype=Layer)
        if activation:
            print("Using given activation function")
            for i in range(1, size-1):
                self.hiddenLayers[i] = Layer(self.layerSize, self.paramCount, activation)
            self.activationDerivative = activationDerivative
        else:
            print("Using ReLU as activation function")
            for i in range(1, size-1):
                self.hiddenLayers[i] = Layer(self.layerSize, self.paramCount)
            self.activationDerivative = step

    def iterate(self, image, label):
        z, a = self.forwardPropagate(image)
        self.backPropagate(z, a, label)
        pass

    def forwardPropagate(self, image):
        
        # converts image into a row vector and iterates on the layers
        # using previous output as the next input
        image = np.array(image)
        image = image.flatten()
        image = np.reshape(image, [np.size(image), 1])

        if np.size(image) != self.featureSize:
            raise ValueError("Input size and layer size do not match")

        a = np.zeros([self.paramCount, self.size])
        z = np.zeros([self.paramCount, self.size])

        # these are input layer outputs
        at, zt = self.inputLayer.process(image)
        a[:,0] = at.T
        z[:,0] = zt.T

        # outputs of the rest of the layers
        for i in range(1, self.size):
            print(np.shape(self.hiddenLayers[i].process(a[i-1])))
            z[i], a[i] = self.hiddenLayers[i].process(a[i-1])

        return z, a
            
    def backPropagate(self, z, a, label):
        # output layer
        delta = np.multiply(a[-1]-label, step(z))

        for i in range(1, self.size-1):
            delta_l = self.layers[i+1].bias
        
        print(self.layers[1].bias)



    def predict(self, image):
        pass
