import numpy as np
from layer import Layer
from extraFuncLib import identity, step

class Network:
    def __init__(self, data, labels, size=10, layerSize=100, learningRate=0.015, batchSize = 16, activation=0, activationDerivative=0):
        
        self.data = data
        self.labels = labels
        self.size = size
        self.layerSize = layerSize
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.featureSize = np.size(data[0])
        self.predictionSize = np.size(list(set(labels)))
        
        self.inputLayer = Layer(self.layerSize, self.featureSize, identity)
        self.outputLayer = Layer(self.predictionSize, self.layerSize, identity)

        self.hiddenLayers = np.zeros(size, dtype=Layer)
        if activation:
            print("Using given activation function")
            for i in range(size):
                self.hiddenLayers[i] = Layer(self.layerSize, self.layerSize, activation)
            self.activationDerivative = activationDerivative
        else:
            print("Using ReLU as activation function")
            for i in range(size):
                self.hiddenLayers[i] = Layer(self.layerSize, self.layerSize)
            self.activationDerivative = step

    def iterate(self, totalEpochs):
        for _ in range(totalEpochs):
            ## shuffle images
            rand_ind = np.random.rand(self.batchSize)
            rand_image = self.images[rand_ind] # get the label too
            Z = np.zeros([self.batchSize, self.layerSize, self.size+1])
            A = np.zeros([self.batchSize, self.layerSize, self.size+1])
            Z_L = np.zeros([self.batchSize, self.predictionSize])
            A_L = np.zeros([self.batchSize, self.predictionSize])
            for i in range(self.batchSize):
                Z[i,...], A[i,...], Z_L[i,...], A_L[i,...] = self.forwardPropagate(rand_image)

            #self.backPropagate(Z, A, Z_L, A_L, rand_ind)

        pass

    def forwardPropagate(self, image):
        # returns the outputs of the forward propagation
        # this includes a (output post non-linear function, if any)
        # and z (output pre non-linear function)
        # a and z are sizes (self.size + 2) and account for
        # input, hidden and output layers in that order


        # converts image into a row vector and iterates on the layers
        # using previous output as the next input
        image = np.array(image)
        image = image.flatten()

        if np.size(image) != self.featureSize:
            raise ValueError("Input size and layer size do not match")

        a = np.zeros([self.layerSize, self.size + 1])
        z = np.zeros([self.layerSize, self.size + 1])

        # these are input layer outputs
        z[:,0], a[:,0] = self.inputLayer.process(image)

        # outputs of the rest of the layers
        for i in range(self.size):
            z[:,i+1], a[:,i+1] = self.hiddenLayers[i].process(a[:,i])
        
        # final layer outputs
        z_L, a_L = self.outputLayer.process(a[:,-1])
        return z, a, z_L, a_L
            
    def backPropagate(self, z, a, z_L, a_L, label):
        # output layer
        delta = np.multiply(a_L-label, self.activationDerivative(z_L))

        # z[-2] represents the output of the last hidden layer
        delta_l = np.zeros([self.layerSize, self.size])
        delta_l[:, -1] = np.multiply(self.outputLayer.weights.T@delta, self.activationDerivative(z[:,-1,np.newaxis])).squeeze()

        for i in reversed(range(self.size-1)):
            delta_l[:, i] = np.multiply(self.hiddenLayers[i].weights.T@delta_l[:,i+1,np.newaxis], self.activationDerivative(np.expand_dims(z[:,i-1], axis=1))).squeeze()
            # nWeights = self.hiddenLayers[i].weights - self.learningRate/self.layerSize * sum(...)
            # nBias = self.hiddenLayers[i].bias - self.learningRate/self.layerSize * sum(...)
            # self.hiddenLayers[i].updateParams(nWeights, nBias) 


    
       
