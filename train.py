#! .venv/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import numpy as np
from torch.utils import tensorboard

import wimblepong
from agent import Agent

writer = tensorboard.SummaryWriter()
# Load the environment
# env = gym.make("WimblepongVisualSimpleAI-v0")
env = gym.make("CartPole-v0")
player = Agent(env, writer=writer)
# player.load_model('results/model_16.mdl')

episodes = 1000
# env.set_names(player.get_name())

global_iter = 0
episode_n = 0
while True:
    episode_n += 1

    done = False
    reward_sum = 0
    next_state = env.reset()
    while not done:
        state = next_state
        action = player.get_action(state, False)
        next_state, reward, done, info = env.step(action)

        player.store_outcome(action, state, next_state, reward, done)

        reward_sum += reward
        global_iter += 1

    writer.add_scalar('reward/episode_reward', reward_sum, global_iter)

    # Update the model based on the last batch_size episodes
    batch_size = 5
    if episode_n % batch_size == batch_size - 1:
        player.update_policy()

    save_rate = 10000
    if episode_n % save_rate == 0:
        quit()
        player.save_model(f'results/model_{episode_n // save_rate}.mdl')