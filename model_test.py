import numpy as np
import torch
from torch.autograd import Variable

from game import Game
from model import DQN


class Event(object):
    def __init__(self, keysym):
        self.keysym = keysym


def map_to_event(action):
    if action == 0:
        return Event('Up')
    elif action == 1:
        return Event('Right')
    elif action == 2:
        return Event('Down')
    elif action == 3:
        return Event('Left')


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


def select_action(state, model, game):
    x = Variable(state, volatile=True).type(torch.FloatTensor)
    action = model(x)
    action = action.data.sort(descending=True)[1]
    for i in range(4):
        if game.is_valid(map_to_event(action[0][i])):
            return torch.from_numpy(np.array([action[0][i]])).type(torch.LongTensor)


LOAD_FILE = ''


if __name__ == "__main__":
    game = Game()
    
    model_dict = torch.load('./trained_models/' + LOAD_FILE + '.pth')['model']
    input_channes = model_dict['conv1.weight'].size()[1]
    model = DQN(input_channes).float()
    model.load_state_dict(model_dict)

    state = game_state_to_input(game.game_state, input_channes)

    final_scores = []
    i = 0
    for countEpisode in range(100):
        terminated = False
        while not terminated:
            action = select_action(state, model, game)
            _, terminated = game.move(map_to_event(action[0]))

            if terminated:
                i += 1
                final_scores.append(game.score)
                print('Episode {} | Score {} | Max Tile {}'.format(i, game.score, np.max(game.game_state)))
                game.restart()

            state = game_state_to_input(game.game_state, input_channes)

    final_scores = np.array(final_scores)
    print(np.mean(final_scores))
