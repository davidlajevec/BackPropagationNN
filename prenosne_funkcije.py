import numpy as np
# različne prenosne funkcije
def sigmoid(x, D=False):
    if not D:
        return 1 / (1 + np.exp(-x))
    else:
        return x * (1.0 - x)

def ELU(z, alpha=1, D=False):
    if not D:
        return np.where(z<=0, alpha*(np.exp(z)-1), z)
    else:
        return np.where(z<=0, ELU(z, alpha), 1)

def ReLU(z, D=False):
    # ReLU function and derivative
    if not D:
        return np.maximum(0, z)
    else:
        return np.where(z>=0, 1, 0)

def PReLU(x, alpha=0.8, D=False):
    if not D:
        return np.where(x > 0, x, x * alpha)
    else:
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

def tanh(x, D=False):
    a = np.tanh(x)
    if not D:
        return a
    else:
        return 1-a**2

def arctan(x, D=False):
    if not D:
        return np.arctan(x)
    else:
        return 1/(1+x**2)










