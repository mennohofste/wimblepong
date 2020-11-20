import torch
import gym

def tensify(x, device):
    if type(x) is torch.Tensor:
        return x
    return torch.from_numpy(x).float().to(device)

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)

def preprocess(x):    
    x = x[::2, ::2, 0]
    x[x == 33] = 0
    x[x != 0] = 1
    return x.flatten()