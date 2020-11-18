import torch

def preprocess(x):    
    x = x[::2, ::2, 0]
    x[x == 33] = 0
    x[x != 0] = 1
    return x.flatten()