import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils import tensorboard

from pippio_utils import tensify
from pippio_utils import get_space_dim
from pippio_utils import preprocess

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        # 200x200x2
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
        # 100x100x4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        # 50x50x8
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        # 25x25x16
        self.fc1 = nn.Linear(25 * 25 * 16, 256)
        self.fc_action = nn.Linear(256, 3)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        # np.savetxt('1.csv', x[:, :, 1].detach().cpu().numpy(), fmt='%d')
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))

        value = self.fc_value(x)
        action_prob = self.fc_action(x)
        action_prob = F.softmax(action_prob, dim=-1)
        return action_prob, value


class Agent(object):
    def __init__(self, env, player_id=1, writer=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy = Policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-3)

        self.global_iter = 0
        self.writer = writer

        self.player_id = player_id
        self.name = 'Pippi-O'

        self.gamma = 0.99
        self.epsilon = 0.2

        self.prev_state = None
        self.actions = []
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []

    def update_policy(self):
        # Load in all the saved data
        actions = torch.Tensor(self.actions).to(self.device).long()
        states = torch.stack(self.states).squeeze(-1).to(self.device)
        next_states = torch.stack(self.next_states).squeeze(-1).to(self.device)
        rewards = torch.Tensor(self.rewards).to(self.device)
        dones = torch.Tensor(self.dones).to(self.device)
        # Reset the memory lists
        self.actions = []
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        
        # Calculate the action probabilities and state values and convert them
        aprobs, state_values = self.policy.forward(states)
        aprobs = torch.sum(aprobs * F.one_hot(actions), dim=1)
        state_values = state_values.squeeze(-1)
        old_aprobs = aprobs.detach()

        _, next_state_values = self.policy.forward(next_states)
        next_state_values = (1 - dones) * next_state_values.squeeze(-1)

        # Advantage
        advantage = rewards + self.gamma * next_state_values - state_values

        # Actor loss
        ratio = (aprobs.log() - old_aprobs.log()).exp() * advantage.detach()
        ratio_clipped = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
        ratio_clipped = ratio_clipped * advantage.detach()
        actor_loss = -torch.min(ratio, ratio_clipped).mean()
        
        # Critic loss
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Logging
        self.writer.add_scalar('ppo/ratio', ratio.mean(), self.global_iter)
        self.writer.add_scalar('loss/actor_loss', actor_loss, self.global_iter)
        self.writer.add_scalar('loss/critic_loss', critic_loss, self.global_iter)
        self.writer.add_scalar('reward/iter_reward', rewards.sum(), self.global_iter)

    def store_outcome(self, action, state, next_state, reward, done):
        self.actions.append(action)
        self.states.append(tensify(state.squeeze(), self.device))
        self.next_states.append(tensify(next_state.squeeze(), self.device))
        self.rewards.append(reward)
        self.dones.append(done)
        self.global_iter += 1

    def get_action(self, state, evaluation=True):
        '''
        Interface function which returns an action of type int given an observation
        '''
        x = preprocess(state, self.prev_state)
        self.prev_state = state

        x = tensify(x, self.device)

        with torch.no_grad():
            aprob, _ = self.policy.forward(x)

        if evaluation:
            action = torch.argmax(aprob, dim=0)
        else:
            action = Categorical(aprob).sample()
        return action.item()

    def get_name(self):
        '''
        Interface function to retrieve the agents name
        '''
        return self.name

    def load_model(self, location):
        '''
        Interface function to loads a trained model
        '''
        self.policy.load_state_dict(torch.load(
            location, map_location=self.device))
        self.policy = self.policy.to(self.device)

    def save_model(self, location):
        '''
        Interface function to save the current model 
        '''
        torch.save(self.policy.state_dict(), location)

    def reset(self):
        '''
        Interface function that resets the agent
        '''
        return