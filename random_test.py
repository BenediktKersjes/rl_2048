from collections import defaultdict

import gym
import gym_2048
import numpy as np


def select_action_simple(env):
    arr = np.array([2, 1, 3, 0])
    for i in range(4):
        if env.is_valid(arr[i]):
            return arr[i]


USE_SIMPLE_POLICY = False
env = gym.make('2048-v0').unwrapped
env.seed(1337)

state = env.reset()
done = True

episode_length = 0
reward_sum = 0
highest = defaultdict(int)
for i in range(5000):
    while True:
        episode_length += 1
        if USE_SIMPLE_POLICY:
            action = select_action_simple(env)
        else:
            action = np.random.choice(np.arange(4))
        while True:
            if env.is_valid(action):
                break
            else:
                action += 1
                action = action % 4

        _, reward, done, _ = env.step(action)
        done = done or episode_length >= 5000
        reward_sum += reward

        if done:
            print("episode reward {}, episode score {}, highest tile {}, episode length {}".format(
                reward_sum, env.score, env.highest(), episode_length))
            reward_sum = 0
            episode_length = 0
            highest[env.highest()] += 1
            state = env.reset()
            break

print(highest)
