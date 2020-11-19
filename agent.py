import torch
import torch.nn as nn
import torch.nn.functional as F

import pippio_utils


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 12)
        self.fc2 = nn.Linear(12, action_space)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class Agent(object):
    def __init__(self, env, player_id=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_space = pippio_utils.get_space_dim(env.observation_space)
        self.action_space = pippio_utils.get_space_dim(env.action_space)
        self.policy = Policy(state_space, self.action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.player_id = player_id
        self.name = 'Pippi-O'

        self.gamma = 0.98
        self.observations = []
        self.actions = []
        self.rewards = []

    def update_policy(self):
        all_actions = torch.stack(self.actions, dim=0).to(self.device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        self.observations, self.actions, self.rewards = [], [], []

        # Get the discounted_rewards
        discounted_rewards = pippio_utils.discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        
        weighted_probs = all_actions * discounted_rewards
        loss = torch.mean(weighted_probs)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad() 

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.device)
        aprob = self.policy.forward(x)
        if evaluation:
            action = torch.argmax(aprob).item()
        else:
            dist = torch.distributions.Categorical(aprob)
            action = dist.sample().item()
        return action, aprob

    def store_outcome(self, observation, action_taken, action_output, reward):
        dist = torch.distributions.Categorical(action_output)
        action_taken = torch.Tensor([action_taken]).to(self.device)
        log_action_prob = -dist.log_prob(action_taken)

        self.observations.append(observation)
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))

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

    def reset(self):
        """
        Interface function that resets the agent
        """
        return