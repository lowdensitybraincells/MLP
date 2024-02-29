import numpy as np
from layer import Layer
from extraFuncLib import identity, step

class Network:
    def __init__(self, size=8, layerSize=100, learningRate=0.015, activation=0, activationDerivative=0):
        
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
            randomised_indices = np.random.randint(low=0, high=labels.shape[0]-1, size=(labels.shape[0]))
            for i in range(np.int32(np.floor(randomised_indices.size/batchSize))):
                indices = randomised_indices[:batchSize]
                randomised_indices = randomised_indices[batchSize:]
                Z = np.zeros([batchSize, self.size+1, self.layerSize])
                A = np.zeros([batchSize, self.size+1, self.layerSize])
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
        # a and z are sizes (self.size + 1) and account for
        # input and hidden layers in that order


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
        return z.T, a.T, z_L, a_L
            
    def backPropagate(self, z, a, z_L, a_L, indices):
        labels = np.array([self.labels[i] for i in indices])
        data = np.array([self.data[i] for i in indices])
        data = np.reshape(data, [data.shape[0], data.shape[1]*data.shape[2]])
        # output layer 
        delta_L = a_L-labels
        nWeights = self.outputLayer.weights - self.learningRate/indices.size * np.einsum("ij,ki->jk", delta_L, a[:,-1,:].T )
        nBias = self.outputLayer.bias - self.learningRate/indices.size * np.sum(delta_L,axis=0)[:,np.newaxis]
        self.outputLayer.updateParams(nWeights, nBias)

        # hidden layers
        delta_l = np.zeros([indices.size, self.size, self.layerSize])
        delta_l[:,-1,:] = np.einsum("ij,kj->ki",self.outputLayer.weights.T,delta_L) * self.activationDerivative(z[:,-1,:])

        for i in reversed(range(1, self.size-1)):
            # calculates for the next layer
            delta_l[:,i-1,:] = np.einsum("ij,kj->ki",self.hiddenLayers[i].weights.T,delta_l[:,i,:]) * self.activationDerivative(z[:,i,:])
            
            # updates the current layer

            nWeights = self.hiddenLayers[i].weights - self.learningRate/indices.size * np.einsum("ij,ki->jk",delta_l[:,i,:], a[:,i,:].T )
            nBias = self.hiddenLayers[i].bias - self.learningRate/indices.size * np.sum(delta_l[:,i,:],axis=0)[:,np.newaxis]
            self.hiddenLayers[i].updateParams(nWeights, nBias) 
        
        nWeights = self.hiddenLayers[0].weights - self.learningRate/indices.size * np.einsum("ij,ki->jk",delta_l[:,0,:], a[:,0,:].T )
        nBias = self.hiddenLayers[0].bias - self.learningRate/indices.size * np.sum(delta_l[:,0,:],axis=0)[:,np.newaxis]
        self.hiddenLayers[0].updateParams(nWeights, nBias) 

        # input layer
        delta_in = np.einsum("ij,kj->kj",self.inputLayer.weights.T,delta_l[:,0,:])
        nWeights = self.inputLayer.weights - self.learningRate/indices.size * np.einsum("ij,ki->jk",delta_in, data.T )
        nBias = self.inputLayer.bias - self.learningRate/indices.size * np.sum(delta_in,axis=0)[:,np.newaxis]
        self.inputLayer.updateParams(nWeights, nBias) 