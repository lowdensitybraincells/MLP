from dataReader import getData
import matplotlib.pyplot as plt
import numpy as np
import layer

if __name__ == "__main__":
    labels, images = getData()
    L = layer.layer(2)
    L.updateParams(np.array([[1,1],[1,2]]), np.array([1,-2]))
    a = L.process([1, 1])
    b = L.process([1, -1])
    print(a, b)
    