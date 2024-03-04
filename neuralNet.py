import numpy as np
from layer import Layer
from extraFuncLib import identity, step

class Network:
    def __init__(self, size=2, layerSize=500, learningRate=0.0007, activation=0, activationDerivative=0):

        
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

    def train(self, data, labels, images_test, labels_test, totalEpochs, batchSize):
        self.data = data
        self.labels = labels
        self.featureSize = np.size(data[0])
        self.predictionSize = np.size(labels[0])

        if self.inputLayer == 0 and self.outputLayer == 0:
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
                    Z[j,...], A[j,...], Z_L[j,...], A_L[j,...] = self.forwardPropagate(self.data[index,:])
                self.backPropagate(Z, A, Z_L, A_L, indices)
            
            print(f"finished epoch {_}: ", end = '')
            cnt = 0
            loss = 0
            for image, label in zip(images_test, labels_test):
                ind, out = self.predict(image)
                cnt += (ind == np.argmax(label))
                loss += np.sum(np.power((out - label), 2))
            loss = loss/labels_test.shape[0]
            print("ind = {}, accuracy = {:2.2%}, cnt = {}, loss = {}".format(ind, cnt/labels_test.shape[0], cnt, loss))

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
    
    def predict(self, data):
        _,_,_,out = self.forwardPropagate(data)
        ind = np.argmax(out)
        return ind, out


    def exportModel(self, fileName):
        
        if self.inputLayer == 0 and self.outputLayer == 0:
            raise ValueError("Model hasn't been trained: layers are not initialised")
        with open(fileName, "w") as file:
            np.array(self.inputLayer.weights.shape).tofile(file=file, sep=',')
            print("",file=file)
            self.inputLayer.weights.tofile(file, sep=',')
            print("",file=file)
            self.inputLayer.bias.tofile(file, sep=',')
            print("",file=file)

            for layer in self.hiddenLayers:
                layer.weights.tofile(file, sep=',')
                print("",file=file)
                layer.bias.tofile(file, sep=',')
                print("",file=file)

            
            np.array(self.outputLayer.weights.shape).tofile(file=file, sep=',')
            print("",file=file)
            self.outputLayer.weights.tofile(file, sep=',')
            print("",file=file)
            self.outputLayer.bias.tofile(file, sep=',')
            print("",file=file)
    
    def importModel(self, fileName):
        with open(fileName, "r") as file:
            if self.inputLayer != 0 or self.outputLayer != 0:
                raise ValueError("Model has already been trained")
            # input layer
            shape = np.fromfile(file=file, sep=',', count=2, dtype=np.int32)
            self.featureSize = np.size(shape[1])
            self.layerSize = np.size(shape[0])
            self.inputLayer = Layer(self.layerSize, self.featureSize, identity)
            weights = np.fromfile(file=file, sep=',').reshape(shape)
            bias = np.fromfile(file=file, sep=',')
            self.inputLayer.updateParams(weights, bias)
            # hidden layers
            for i in range(self.hiddenLayers.size):
                weights = np.fromfile(file=file, sep=',').reshape([self.layerSize, self.layerSize])
                bias = np.fromfile(file=file, sep=',')
                self.hiddenLayers[i].updateParams(weights, bias)
            # output layer
            shape = np.fromfile(file=file, sep=',', count=2, dtype=np.int32)
            self.predictionSize = shape[0]
            self.outputLayer = Layer(self.predictionSize, self.layerSize, identity)
            weights = np.fromfile(file=file, sep=',').reshape(shape)
            bias = np.fromfile(file=file, sep=',')
            self.outputLayer.updateParams(weights, bias)