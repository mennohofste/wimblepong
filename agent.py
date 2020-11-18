import torch
import torch.nn as nn
import torch.nn.functional as F

import pippio_utils

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 512)
        self.fc_action = nn.Linear(512, 3)
        self.fc_value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        state_value = self.fc_value(x)

        x = self.fc_action(x)
        action_prob = F.softmax(x, dim=0)

        return action_prob, state_value


class Agent(object):
    def __init__(self, env, player_id=1):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = Policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.player_id = player_id
        self.name = 'Pippi-O'

        self.gamma = 0.98
        self.epsilon = 0.2

        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def update_policy(self):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0).to(
            self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(
            self.device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

        _, states_values = self.policy.forward(states)
        states_values = states_values.squeeze(-1)
        _, next_states_values = self.policy.forward(next_states)
        next_states_values = (1 - done) * next_states_values.squeeze(-1)

        critic_loss = F.mse_loss(
            states_values, rewards + self.gamma * next_states_values.detach())

        advantage = rewards + self.gamma * next_states_values - states_values

        actor_loss = -torch.mean(action_probs * advantage.detach())

        loss = actor_loss + critic_loss
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def load_model(self, location):
        """
        Interface function to loads a trained model
        """
        self.policy.load_state_dict(torch.load(
            location, map_location=self.device))
        self.policy = self.policy.to(self.device)

    def save_model(self, location):
        """
        Interface function to save the current model 
        """
        torch.save(self.policy.state_dict(), location)

    def get_action(self, ob, evaluation=False):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        x = torch.from_numpy(ob).float().to(self.device)
        x = pippio_utils.preprocess(x)
        act_prob, _ = self.policy.forward(x)
        action_dist = torch.distributions.Categorical(act_prob)
        if evaluation:
            action = torch.argmax(act_prob)
        else:
            action = action_dist.sample()

        act_log_prob = action_dist.log_prob(action)
        return action, act_log_prob

    def reset(self):
        """
        Interface function that resets the agent
        """
        return

    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        state = pippio_utils.preprocess(state)
        next_state = pippio_utils.preprocess(next_state)
        
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
