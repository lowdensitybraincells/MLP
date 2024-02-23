import numpy as np

def identity(x):
    return np.array(x)

def ReLU(x):
    x = np.array(x)
    return np.multiply(x, x>0, dtype=np.float32)

def step(x):
    x = np.array(x)
    return np.array(x>0, dtype=np.float32)