import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import tensorboard
import random

import wimblepong
from policy import Policy

# Learning parameters
GAMMA = 0.99
EPS_PER_ITERATION = 20
EPOCHS = 5
MAX_TIMESTEPS = 500
LEARNING_RATE = 1e-4

def collect_data():
    global global_n
    
    state_history, action_history, action_prob_history, reward_history = [], [], [], []
    for _ in range(EPS_PER_ITERATION):
        obs, prev_obs, prev2_obs, prev3_obs = env.reset(), None, None, None

        for t in range(MAX_TIMESTEPS):
            state = policy.pre_process(obs, prev_obs, prev2_obs, prev3_obs)
            prev_obs, prev2_obs, prev3_obs = obs, prev_obs, prev2_obs

            with torch.no_grad():
                action, action_prob = policy.get_action(state)
            obs, reward, done, _ = env.step(action)

            state_history.append(state)
            action_history.append(action)
            action_prob_history.append(action_prob)
            reward_history.append(reward)

            if done:
                break

        writer.add_scalar('reward/episode_reward', sum(reward_history[-t:]), global_n)
        writer.add_scalar('reward/episode_length', t, global_n)
        global_n += 1
    
    return [state_history, action_history, action_prob_history, reward_history]

def compute_advantages(reward_history):
    R = 0
    discounted_rewards = torch.zeros(len(reward_history))
    for i, r in enumerate(reward_history[::-1]):
        # scored a point so reset the cumulation
        if r != 0:
            R = 0
        R = r + GAMMA * R
        discounted_rewards[-i] = R

    discounted_rewards -= discounted_rewards.mean()
    discounted_rewards /= discounted_rewards.std() + 1.0e-10
    return discounted_rewards

def update_policy(data):
    state_history = data.pop(0)
    action_history = data.pop(0)
    action_prob_history = data.pop(0)
    advantage_history = data.pop(0)

    # update policy
    for _ in range(EPOCHS):
        n_batch = len(action_history)
        idxs = random.sample(range(len(action_history)), n_batch)
        state_batch = torch.cat([state_history[idx] for idx in idxs], 0)
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
        advantage_batch = torch.FloatTensor([advantage_history[idx] for idx in idxs])

        opt.zero_grad()
        loss = policy.get_loss(state_batch, action_batch, action_prob_batch, advantage_batch)
        writer.add_scalar('loss/loss', -loss, global_n)
        loss.backward()
        opt.step()


if __name__ == "__main__":
    writer = tensorboard.SummaryWriter()
    global_n = 0

    # env = gym.make("CartPole-v0")
    env = gym.make("WimblepongVisualImprovedAI-v0")
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    policy.load_state_dict(torch.load(f'results/model.mdl'))

    i = 0
    while True:
        data = collect_data()
        reward_history = data.pop(3)
        advantage_history = compute_advantages(reward_history)
        data.append(advantage_history)
        writer.add_scalar('reward/average_reward', sum(reward_history) / EPS_PER_ITERATION, i)
        writer.add_scalar('reward/average_ep_len', len(reward_history) / EPS_PER_ITERATION, i)
        writer.add_scalar('loss/advantage', advantage_history.mean(), global_n)
        update_policy(data)

        if i % 100 == 0:
            torch.save(policy.state_dict(), f'results/model_more_eps_{i}.mdl')
        i += 1

    env.close()
