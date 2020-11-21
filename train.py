#! .venv/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import numpy as np
from torch.utils import tensorboard
from pippio_utils import preprocess
from pippio_utils import tensify

import wimblepong
from agent import Agent

writer = tensorboard.SummaryWriter()
# Load the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
player = Agent(env, writer=writer)

# player.load_model('results/model_16.mdl')

env.set_names(player.get_name())

global_iter = 0
episode_n = 0
while True:
    episode_n += 1

    done = False
    reward_sum = 0
    obs = env.reset()
    next_state = preprocess(obs, None)
    while not done:
        prev_obs = obs
        state = next_state
        action = player.get_action(prev_obs, False)
        obs, reward, done, info = env.step(action)
        next_state = preprocess(obs, prev_obs)

        player.store_outcome(action, state, next_state, reward, done)

        reward_sum += reward
        global_iter += 1

    writer.add_scalar('reward/episode_reward', reward_sum, global_iter)

    # Update the model based on the last batch_size episodes
    batch_size = 100
    if episode_n % batch_size == batch_size - 1:
        player.update_policy()

    save_rate = 1000
    if episode_n % save_rate == 0:
        player.save_model(f'results/model_{episode_n // save_rate}.mdl')