import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 0.99
        self.eps_clip = 0.2

        self.fc1 = nn.Linear(200 * 200, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

    def pre_process(self, x, prev_x):
        if prev_x is None:
            return self.state_to_tensor(x)
        return self.state_to_tensor(x) - self.state_to_tensor(prev_x)
        # return torch.cat([self.state_to_tensor(x), self.state_to_tensor(prev_x)])

    def state_to_tensor(self, x):
        x = torch.from_numpy(x)
        # Normalise color space
        x = torch.sum(x, dim=-1) / (3 * 255)
        x = torch.flatten(x)
        return x

    def get_loss(self, d_obs, action, action_prob_old, advantage):
        action_prob = torch.sum(self.forward(d_obs) * F.one_hot(action, num_classes=3), dim=1)
        r = action_prob / action_prob_old
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)
        return loss

    def get_action(self, state, evaluation=True):
        '''
        Interface function which returns an action of type int given an observation
        '''
        with torch.no_grad():
            action_prob = self.forward(state)

        if evaluation:
            return torch.argmax(action_prob).item()
        else:
            action = Categorical(action_prob).sample().item()
            return action, action_prob[action]
