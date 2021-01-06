import numpy as np

# različne prenosne funkcije
def sigmoid(x, D=False):
    if not D:
        return 1 / (1 + np.exp(-x))
    else:
        return x * (1.0 - x)





