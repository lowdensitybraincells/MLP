from dataReader import getData
import matplotlib.pyplot as plt
import numpy as np
import neuralNet

if __name__ == "__main__":
    images_train, labels_train, images_test, labels_test = getData()
    tmp = np.zeros([np.shape(labels_train)[0], 2])
    for i, _ in enumerate(labels_train):
        if labels_train[i][0] == 1:
            tmp[i] = np.array([1,0])
        else:
            tmp[i] = np.array([0,1])
    labels_train = tmp

    tmp = np.zeros([np.shape(labels_test)[0],2])
    for i,_ in enumerate(labels_test):
        if labels_test[i][0] == 1:
            tmp[i] = np.array([1,0])
        else:
            tmp[i] = np.array([0,1])
    labels_test = tmp

    net = neuralNet.Network()
    net.train(images_train, labels_train, images_test, labels_test, 15, 16)
    net.exportModel("model.txt")
    # net.importModel("model.txt")