import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import tensorboard
import random

import wimblepong
from policy import Policy

# Learning parameters
EPS_PER_EPOCH = 100
MAX_TIMESTEPS = 500
LEARNING_RATE = 1e-3

def collect_data():
    global global_n
    
    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for _ in range(EPS_PER_EPOCH):
        obs, prev_obs = env.reset(), None
        for t in range(MAX_TIMESTEPS):
            d_obs = policy.pre_process(obs, prev_obs)
            with torch.no_grad():
                action, action_prob = policy.get_action(d_obs, False)

            prev_obs = obs
            obs, reward, done, _ = env.step(action)

            d_obs_history.append(d_obs)
            action_history.append(action)
            action_prob_history.append(action_prob)
            reward_history.append(reward)

            if done:
                break

        writer.add_scalar('reward/episode_reward', sum(reward_history[-t:]), global_n)
        writer.add_scalar('reward/episode_length', t, global_n)
        global_n += 1
    
    return [d_obs_history, action_history, action_prob_history, reward_history]

def compute_advantages(reward_history):
    # compute advantage
    R = 0
    discounted_rewards = torch.zeros(len(reward_history))

    for i, r in enumerate(reward_history[::-1]):
        if r != 0:  # scored/lost a point in pong, so reset reward sum
            R = 0
        R = r + policy.gamma * R
        discounted_rewards[i] = R

    discounted_rewards -= discounted_rewards.mean()
    discounted_rewards /= discounted_rewards.std()
    return discounted_rewards

def update_policy(data):
    d_obs_history = data.pop(0)
    action_history = data.pop(0)
    action_prob_history = data.pop(0)
    advantage_history = data.pop(0)

    # update policy
    for _ in range(10):
        n_batch = len(action_history) // 4
        idxs = random.sample(range(len(action_history)), n_batch)
        d_obs_batch = torch.stack([d_obs_history[idx] for idx in idxs])
        action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
        action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
        advantage_batch = torch.FloatTensor([advantage_history[idx] for idx in idxs])

        opt.zero_grad()
        loss = policy.get_loss(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
        loss.backward()
        opt.step()

def train_iteration(i):
    data = collect_data()
    reward_history = data.pop(3)
    advantage_history = compute_advantages(reward_history)
    data.append(advantage_history)
    writer.add_scalar('reward/average_reward', sum(reward_history) / EPS_PER_EPOCH, i)
    writer.add_scalar('reward/average_ep_len', len(reward_history) / EPS_PER_EPOCH, i)
    writer.add_scalar('loss/advantage', advantage_history.mean(), global_n)
    update_policy(data)

if __name__ == "__main__":
    writer = tensorboard.SummaryWriter()
    global_n = 0

    env = gym.make("WimblepongVisualSimpleAI-v0")
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    model_number = -1
    if model_number != -1:
        policy.load_state_dict(torch.load(f'results/new_{model_number}.mdl'))

    i = 0
    while True:
        train_iteration(i)
        if i % 10 == 0:
            torch.save(policy.state_dict(), f'results/model_{i}.mdl')
        i += 1

    env.close()
