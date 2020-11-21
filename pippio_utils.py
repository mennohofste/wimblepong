import torch
import gym
import numpy as np

def tensify(x, device):
    return torch.from_numpy(x).float().to(device)

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)

def preprocess(obs, prev_obs):
    if prev_obs is None:
        prev_obs = np.zeros_like(obs)

    obs = obs.sum(axis=-1) / (3 * 255)
    prev_obs = prev_obs.sum(axis=-1) / (3 * 255)
    result = np.stack([obs, prev_obs])
    return np.expand_dims(result, axis=0)