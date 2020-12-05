import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils import tensorboard

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.size = 9 * 9 * 64
        
        self.fc1 = nn.Linear(self.size, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class Agent(object):
    def __init__(self, env, player_id=1):
        self.device = 'cpu'
        self.policy = Policy().to(self.device)
        self.player_id = player_id
        self.name = 'Pippi-O'

        self.prev_x = torch.zeros((100, 100))
        self.prev2_x = torch.zeros((100, 100))
        self.prev3_x = torch.zeros((100, 100))

    def get_action(self, obs):
        x = self.state_to_tensor(obs.copy())
        state = torch.stack([x, self.prev_x, self.prev2_x, self.prev3_x]).unsqueeze(0)
        self.prev_x, self.prev2_x, self.prev3_x = x, self.prev_x, self.prev2_x

        with torch.no_grad():
            probs = self.policy.forward(state)

        return int(torch.argmax(probs))

    def pre_process(self, x, prev_x, prev2_x, prev3_x):
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

    def get_name(self):
        '''
        Interface function to retrieve the agents name
        '''
        return self.name

    def load_model(self, location='model.mdl'):
        '''
        Interface function to loads a trained model
        '''
        self.policy.load_state_dict(torch.load(
            location, map_location=self.device))
        self.policy = self.policy.to(self.device)


    def reset(self):
        '''
        Interface function that resets the agent
        '''
        return