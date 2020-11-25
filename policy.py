import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

EPS_CLIP = 0.1
BETA = 0.01

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.size = 9 * 9 * 64
        
        self.fc1 = nn.Linear(self.size, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

    def pre_process(self, x, prev_x, prev2_x, prev3_x):
        x = self.state_to_tensor(x)
        prev_x = self.state_to_tensor(prev_x)
        prev2_x = self.state_to_tensor(prev2_x)
        prev3_x = self.state_to_tensor(prev3_x)
        output = torch.stack([x, prev_x, prev2_x, prev3_x,])
        return output.unsqueeze(0)

    def state_to_tensor(self, x):
        if x is None:
            return torch.zeros((100, 100))
        # Normalise color space
        x = x[::2, ::2, 1]
        x[x == 1] = 0
        x[x != 0] = 1
        return torch.from_numpy(x).float()

    def get_loss(self, states, action, action_prob_old, advantage):
        action_prob = self.forward(states)
        action_prob = torch.sum(action_prob * F.one_hot(action, 2), dim=1)

        entropy = action_prob * action_prob_old.log() + \
            (1 - action_prob) * action_prob_old.log()
        r = action_prob / action_prob_old
        loss1 = r * advantage
        loss2 = torch.clip(r, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
        loss = torch.min(loss1, loss2)
        loss = torch.mean(loss - BETA * entropy)
        return -loss

    def get_action(self, state):
        '''
        Interface function which returns an action of type int given an observation
        '''
        with torch.no_grad():
            action_prob = self.forward(state).squeeze()
            dist = Categorical(action_prob)
            action = dist.sample().item()

        return action, action_prob[action]
