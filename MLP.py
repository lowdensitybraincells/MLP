from dataReader import getData
import matplotlib.pyplot as plt
import numpy as np
import neuralNet

if __name__ == "__main__":
    labels, images = getData()
    net = neuralNet.Network(28*28)
    
