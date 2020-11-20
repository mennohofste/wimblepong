import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils import tensorboard

from pippio_utils import tensify
from pippio_utils import get_space_dim

class Policy(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc_action = nn.Linear(16, n_actions)
        self.fc_value = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        value = self.fc_value(x)
        action_prob = self.fc_action(x)
        action_prob = F.softmax(action_prob, dim=-1)
        return action_prob, value


class Agent(object):
    def __init__(self, env, player_id=1, writer=None):
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dim = get_space_dim(env.observation_space)
        self.action_dim = get_space_dim(env.action_space)

        self.policy = Policy(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-3)

        self.global_iter = 0
        self.writer = writer

        self.player_id = player_id
        self.name = 'Pippi-O'

        self.gamma = 0.98
        self.epsilon = 0.2

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
        old_aprobs = torch.sum(aprobs * F.one_hot(actions), dim=1).detach()
        state_values = state_values.squeeze(-1)

        _, next_state_values = self.policy.forward(next_states)
        next_state_values = (1 - dones) * next_state_values.squeeze(-1)

        # Advantage
        advantage = rewards + self.gamma * next_state_values - state_values

        actor_losses = []
        for _ in range(5):# TODO: how many time do we want to iterate over this?
            aprobs, _ = self.policy.forward(states)
            aprobs = torch.sum(aprobs * F.one_hot(actions), dim=1)
            # # Actor loss
            ratio = (aprobs / old_aprobs) * advantage.detach()
            ratio_clipped = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
            ratio_clipped = ratio_clipped * advantage.detach()
            actor_losses.append(-torch.min(ratio, ratio_clipped).mean())
        actor_loss = torch.stack(actor_losses).mean()
        
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
        # self.writer.add_scalar('reward/iter_reward', rewards.sum(), self.global_iter)

    def store_outcome(self, action, state, next_state, reward, done):
        self.actions.append(action)
        self.states.append(tensify(state, self.device))
        self.next_states.append(tensify(next_state, self.device))
        self.rewards.append(reward)
        self.dones.append(done)
        self.global_iter += 1

    def get_action(self, observation, evaluation=True):
        '''
        Interface function which returns an action of type int given an observation
        '''
        x = tensify(observation, self.device)
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