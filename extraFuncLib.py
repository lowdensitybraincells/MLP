import numpy as np

def identity(x):
    return np.array(x)

def ReLU(x):
    x = np.array(x)
    return np.multiply(0.1*x, x<0, dtype=np.float32) + np.multiply(x, x>0, dtype=np.float32)

def step(x):
    x = np.array(x)
    return  0.1 + 0.9*np.array(x>0, dtype=np.float32)

def unity(x):
    return np.ones(np.shape(x))