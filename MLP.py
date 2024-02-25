from dataReader import getData
import matplotlib.pyplot as plt
import numpy as np
import neuralNet

if __name__ == "__main__":
    images, labels = getData()
    net = neuralNet.Network(images, labels)
    net.iter(10) 
