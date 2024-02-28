import numpy as np
from layer import Layer
from extraFuncLib import identity, step

class Network:
    def __init__(self, size=10, layerSize=100, learningRate=0.015, activation=0, activationDerivative=0):
        
        self.size = size
        self.layerSize = layerSize
        self.learningRate = learningRate
        self.featureSize = 0
        self.predictionSize = 0 

        self.inputLayer = 0
        self.outputLayer = 0
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

    def train(self, data, labels, totalEpochs, batchSize):
        self.data = data
        self.labels = labels
        self.featureSize = np.size(data[0])
        self.predictionSize = np.size(labels[0])

        self.inputLayer = Layer(self.layerSize, self.featureSize, identity)
        self.outputLayer = Layer(self.predictionSize, self.layerSize, identity)

        # trains the model in set amount of epochs, using a specific batch size
        for _ in range(totalEpochs):
            # shuffles images
            randomised_indices = np.random.randint(low=0, high=labels[0].size-1, size=(labels[0].size))
            for i in range(np.int32(np.floor(randomised_indices.size/batchSize))):
                indices = randomised_indices[:batchSize]
                randomised_indices = randomised_indices[batchSize:]
                Z = np.zeros([batchSize, self.layerSize, self.size+1])
                A = np.zeros([batchSize, self.layerSize, self.size+1])
                Z_L = np.zeros([batchSize, self.predictionSize])
                A_L = np.zeros([batchSize, self.predictionSize])               
                for j, index in enumerate(indices):
                    Z[j,...], A[j,...], Z_L[j,...], A_L[j,...] = self.forwardPropagate(self.data[index,:,:])
                self.backPropagate(Z, A, Z_L, A_L, indices)
            print(f"finished epoch {_}")

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
            
    def backPropagate(self, z, a, z_L, a_L, indices):
        # output layer
        labels = np.array([self.labels[i] for i in indices])
        data = np.array([self.data[i] for i in indices])
        print(np.shape(a_L), np.shape(labels))
        return   
        delta = np.multiply(a_L-labels, self.activationDerivative(z_L))
        
        return
        # z[-2] represents the output of the last hidden layer
        delta_l = np.zeros([self.layerSize, self.size])
        delta_l[:, -1] = np.multiply(self.outputLayer.weights.T@delta, self.activationDerivative(z[:,-1,np.newaxis])).squeeze()

        for i in reversed(range(self.size-1)):
            delta_l[:, i] = np.multiply(self.hiddenLayers[i].weights.T@delta_l[:,i+1,np.newaxis], self.activationDerivative(np.expand_dims(z[:,i-1], axis=1))).squeeze()
            # nWeights = self.hiddenLayers[i].weights - self.learningRate/self.layerSize * sum(...)
            # nBias = self.hiddenLayers[i].bias - self.learningRate/self.layerSize * sum(...)
            # self.hiddenLayers[i].updateParams(nWeights, nBias) 


    
       
