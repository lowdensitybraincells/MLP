from dataReader import getData
import matplotlib.pyplot as plt
import numpy as np
import neuralNet
from extraFuncLib import identity, unity

if __name__ == "__main__":
    images_train, labels_train, images_test, labels_test = getData()
    tmp = np.ndarray([2])[np.newaxis, :]
    ind = []
    for i, _ in enumerate(labels_train):
        if labels_train[i][0] == 1:
            tmp = np.append(tmp, np.array([1,0])[np.newaxis, :], 0)
            ind.append(i)
        if labels_train[i][1] == 1:
            tmp = np.append(tmp, np.array([0,1])[np.newaxis, :], 0)
            ind.append(i)
    labels_train = tmp[1:,...]
    ind = np.array(ind)
    tmpimg = np.ndarray([ind.size, images_train.shape[1]])
    for i in range(ind.size):
        tmpimg[i,...] = images_train[ind[i],...]
    avg = np.average(tmpimg)
    for i in range(ind.size):
        tmpimg[i,...] = tmpimg[i,...]*avg/np.average(tmpimg[i,...])
    images_train = tmpimg

    tmp = np.ndarray([2])[np.newaxis, :]
    ind = []
    for i, _ in enumerate(labels_test):
        if labels_test[i][0] == 1:
            tmp = np.append(tmp, np.array([1,0])[np.newaxis, :], 0)
            ind.append(i)
        if labels_test[i][1] == 1:
            tmp = np.append(tmp, np.array([0,1])[np.newaxis, :], 0)
            ind.append(i)
    labels_test = tmp[1:,...]
    ind = np.array(ind)
    tmpimg = np.ndarray([ind.size, images_test.shape[1]])
    for i in range(ind.size):
        tmpimg[i,...] = images_test[ind[i],...]
    images_test = tmpimg

    # plt.imshow(255*images_test[0,...].reshape([28,28]), cmap='gray')
    # plt.show()

    # net = neuralNet.Network(activation=identity, activationDerivative=unity)
    net = neuralNet.Network(learningRate=0.2)
    net.train(images_train, labels_train, images_test, labels_test, 30, 16)
    net.exportModel("model.txt")
    # net.importModel("model.txt")