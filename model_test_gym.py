import time
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable

import gym
import gym_2048
from model import DQN
import matplotlib.pyplot as plt


def game_state_to_input(game_state, input_channels):
    game_state = game_state.flatten()

    if input_channels == 16:
        input_state = torch.FloatTensor(16, 4, 4).zero_()

        for i in range(4):
            for j in range(4):
                index = int(np.log2(game_state[i * 4 + j]) - 1) if game_state[i * 4 + j] != 0 else -1
                if index >= 0:
                    input_state[index, i, j] = 1

    else:
        input_state = torch.FloatTensor(4, 4).zero_()

        for i in range(4):
            for j in range(4):
                input_state[i, j] = np.log2(game_state[i * 4 + j]) / 15 if game_state[i * 4 + j] != 0 else 0
        input_state = input_state.unsqueeze(0)

    return input_state.unsqueeze(0)


def select_action(state, model, env):
    x = Variable(state, volatile=True).type(torch.FloatTensor)
    action = model(x)
    action = action.data.sort(descending=True)[1]
    for i in range(4):
        if env.is_valid(action[0][i]):
            return torch.from_numpy(np.array([action[0][i]])).type(torch.LongTensor)


if __name__ == "__main__":
    LOAD_FILE = ''
    env = gym.make('2048-v0').unwrapped

    model_dict = torch.load('./trained_models/' + LOAD_FILE + '.pth')['model']
    input_channels = model_dict['conv1.weight'].size()[1]
    model = DQN(input_channels).float()
    model.load_state_dict(torch.load('./trained_models/' + LOAD_FILE + '.pth')['model'])

    score = []
    highest = defaultdict(int)

    state = env.reset()
    i = 0
    reward_sum = 0
    while True:
        state = game_state_to_input(state, input_channels)
        action = select_action(state, model, env)
        state, reward, terminated, _ = env.step(action[0])

        reward = np.log2(reward) / 15 if reward != 0 else 0
        reward_sum += reward

        if terminated:
            i += 1
            score.append(float(env.score))
            highest[env.highest()] += 1
            print('Episode {} | Score {} | Max Tile {} | Reward {}'.format(i, env.score, env.highest(), reward_sum))
            state = env.reset()
            reward_sum = 0
            if i == 5000:
                break

    print(np.array(score).mean())
    print(highest)
