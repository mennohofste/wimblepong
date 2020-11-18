#! .venv/bin/python
import matplotlib.pyplot as plt
import gym
import numpy as np

import wimblepong
from agent import Agent

# Load the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
player = Agent(env)

# location = 'model.mdl'
# player.save_model(location)
# player.load_model(location)

episodes = 10
env.set_names(player.get_name())

rew_running_avg = None
rew_hist = []
rew_running_avg_hist = []

for i in range(episodes):
    rew_sum = 0
    done = False
    ob = env.reset()

    while not done:
        action, action_prob = player.get_action(ob)
        prev_ob = ob

        ob, rew, done, info = env.step(action.detach().numpy())
        rew_sum += rew

        player.store_outcome(prev_ob, ob, action_prob, rew, done)

        # env.render()

    print(f'Episode {i} finished. Total reward: {rew_sum}')
    rew_hist.append(rew_sum)
    if rew_running_avg is None:
        rew_running_avg = rew_sum
    else:
        rew_running_avg = 0.99 * rew_running_avg + 0.01 * rew_sum
    rew_running_avg_hist.append(rew_running_avg)

    player.update_policy()

plt.plot(rew_hist)
plt.plot(rew_running_avg_hist)
plt.legend(["Reward", "Rolling average (100)"])
plt.title("Reward history")
plt.savefig(f'fig.png')
print("Training finished.")
player.save_model(f'model.mdl')
