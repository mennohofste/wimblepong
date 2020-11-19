#! .venv/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import numpy as np

import wimblepong
from agent import Agent

# Load the environment
# env = gym.make("WimblepongVisualSimpleAI-v0")
env = gym.make("WimblepongSimpleAI-v0")
player = Agent(env)
# player.load_model('results/model_16.mdl')

episodes = 1000
env.set_names(player.get_name())

rew_hist = []
rew_running_avg1 = -10
rew_running_avg2 = -10
rew_running_avg_hist1 = []
rew_running_avg_hist2 = []
ep = 0
while True:
    rew_sum = 0
    done = False
    ob = env.reset()

    # Run the episode itself
    length = 0
    while not done:
        action, action_prob = player.get_action(ob)

        ob, rew, done, info = env.step(action)

        player.store_outcome(ob, action, action_prob, rew)

        rew_sum += rew
        length += 1
        # env.render()

    rew_hist.append(rew_sum)
    rew_running_avg1 = 0.99 * rew_running_avg1 + 0.01 * rew_sum
    rew_running_avg_hist1.append(rew_running_avg1)
    rew_running_avg2 = 0.999 * rew_running_avg2 + 0.001 * rew_sum
    rew_running_avg_hist2.append(rew_running_avg2)
    ep += 1

    if ep % 10 == 0:
        print(f'Episode: {ep:>5}, length: {length:>5}, running1: {rew_running_avg1:>7}, running2: {rew_running_avg2:>7}')

    # Update the model based on the last batch_size episodes
    batch_size = 200
    if ep % batch_size == batch_size - 1:
        player.update_policy()

    save_rate = 1000
    if ep % save_rate == 0:
        plt.figure()
        plt.plot(rew_running_avg_hist1, label='Rolling average (100)')
        plt.plot(rew_running_avg_hist2, label='Rolling average (1000)')
        plt.legend()
        plt.title("Reward history")
        plt.savefig(f'results/fig_{ep // save_rate}.png')
        player.save_model(f'results/model_{ep // save_rate}.mdl')