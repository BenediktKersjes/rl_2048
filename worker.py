import datetime
import time

import gym
# noinspection PyUnresolvedReferences
import gym_2048
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import multiprocessing as mp
from torch.autograd import Variable
from torch.nn import functional


# noinspection PyArgumentList
class Worker(mp.Process):
    def __init__(self, shared_model, target_model, optimizer, move_counter, episodes_counter, start_time, lock,
                 args, last_1000, last_1000_loss, worker_id, eps_end):
        super(Worker, self).__init__()
        self.shared_model = shared_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.move_counter = move_counter
        self.episodes_counter = episodes_counter
        self.start_time = start_time
        self.lock = lock
        self.args = args
        self.last_1000 = last_1000
        self.last_1000_loss = last_1000_loss
        self.rewards = []
        self.loss = []

        self.env = gym.make('2048-v0').unwrapped
        self.eps_end = eps_end
        self.eps = self.args.eps_start
        self.env.seed(self.args.seed)
        self.worker_id = worker_id

        if self.worker_id == 0:
            self.count_tested = 0
            self.output_file_name = './trained_models/ADQN_{date:%Y_%m_%d__%H_%M_%S}.txt'\
                .format(date=datetime.datetime.now())

        if self.args.start_step != 0:
            self.eps -= self.args.start_step * ((self.args.eps_start - self.eps_end) / self.args.eps_decay)

    def game_state_to_input(self, game_state):
        if self.args.use_big_input:
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

    def input_to_game_matrix(self, input_state):
        matrix = np.zeros((4, 4))

        if self.args.use_big_input:
            indices = input_state.nonzero()
            for exponent, i, j in indices:
                matrix[int(i), int(j)] = int(2**int(exponent+1))

        else:
            for i in range(4):
                for j in range(4):
                    if input_state.data[0][i][j] != 0:
                        matrix[i, j] = int(2 ** (input_state.data[0][i][j] * 15))

        return matrix

    def select_action(self, state):
        sample = np.random.random()

        if sample > self.eps:
            x = Variable(state, volatile=True).type(torch.FloatTensor)
            action = self.shared_model(x).data.sort(descending=True)[1]
            for i in range(4):
                if self.env.is_valid(action[0][i]):
                    return torch.from_numpy(np.array([action[0][i]])).type(torch.LongTensor)
        else:
            arr = np.random.permutation([0, 1, 2, 3])
            for i in range(4):
                if self.env.is_valid(arr[i]):
                    return torch.from_numpy(np.array([arr[i]])).type(torch.LongTensor)

    def optimize_model(self, states, rewards, actions, next_states):
        if len(next_states) == 1 and next_states[0] is None:
            return

        non_final_mask = torch.LongTensor([i for i, s in enumerate(next_states) if s is not None])
        non_final_next_states = Variable(torch.cat([s for s in next_states if s is not None]), volatile=True)

        state_batch = Variable(torch.cat(states))
        action_batch = Variable(torch.cat(actions))
        reward_batch = Variable(torch.cat(rewards))

        state_values = self.shared_model(state_batch)

        state_action_values = state_values.gather(1, action_batch)
        next_state_q_values = Variable(torch.zeros(len(states), 4).type(torch.FloatTensor))
        next_state_values = Variable(torch.zeros(len(states)).type(torch.FloatTensor))

        non_valid_action_mask = torch.FloatTensor(len(non_final_next_states), 4)
        for i in range(len(non_final_next_states)):
            for j in range(4):
                if self.env.is_valid(j, self.input_to_game_matrix(non_final_next_states[i])):
                    non_valid_action_mask[i, j] = 0
                else:
                    non_valid_action_mask[i, j] = float('-inf')

        model_result = self.target_model(non_final_next_states)
        next_state_q_values.data.index_copy_(0, non_final_mask, model_result.data)
        next_state_q_values[non_final_mask] += Variable(non_valid_action_mask)
        next_state_values[non_final_mask] = next_state_q_values[non_final_mask].max(1)[0]

        expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch
        loss = functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        with self.lock:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.target_model.parameters(), self.args.max_grad_norm)
            self.share_model()
            self.optimizer.step()

            if self.episodes_counter.value <= 1000:
                self.last_1000_loss.append(loss.data[0])
            else:
                self.last_1000_loss[self.episodes_counter.value % 1000] = loss.data[0]

    def share_model(self):
        for local_param, shared_param in zip(self.target_model.parameters(), self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = local_param.grad

    def test_performance(self):
        state = self.game_state_to_input(self.env.reset())
        episode = self.episodes_counter.value
        eps = self.eps
        self.eps = 0

        while True:
            action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action[0])
            state = self.game_state_to_input(next_state)

            if done:
                print('Episode: {}, Test Score: {}, Highest Tile: {}'.format(
                    episode, self.env.score, self.env.highest()))
                print('MPS: {}, Moves: {}, Avg. 1000 {}'.format(
                    self.move_counter.value / (time.time() - self.start_time),
                    self.move_counter.value, sum(self.last_1000) / 1000.))

                if self.episodes_counter.value >= 1000:
                    self.loss.append(sum(self.last_1000_loss) / 1000.)
                    self.rewards.append(sum(self.last_1000) / 1000.)
                    plot_graphs(self.rewards, self.loss)
                    output_file = open(self.output_file_name, 'a')
                    output_file.write('{}, {}, {}, {}\n'
                                      .format(sum(self.last_1000) / 1000., self.move_counter.value,
                                              self.episodes_counter.value, sum(self.last_1000_loss) / 1000.))
                    output_file.flush()
                    output_file.close()
                break

        self.eps = eps

        if self.count_tested % 10 == 0:
            print('Saving model')
            model_name = "ADQN_" + '{date:%Y_%m_%d__%H_%M_%S}'.format(date=datetime.datetime.now())
            torch.save({
                'model': self.shared_model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, './trained_models/{}.pth'.format(model_name))

        self.count_tested += 1

    def run(self):
        moves = 0
        reward_sum = 0
        done = False
        t = 0

        state = self.env.reset()
        states = [(state, 0)]
        state = self.game_state_to_input(state)

        while True:
            if done:
                reward_sum = 0
                moves = 0

            batch_states = []
            batch_rewards = []
            batch_actions = []
            batch_next_states = []

            for step in range(self.args.num_steps):
                with self.lock:
                    self.move_counter.value += 1

                    if self.move_counter.value % 40000 == 0:
                        self.target_model.load_state_dict(self.shared_model.state_dict())

                moves += 1
                t += 1

                action = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action[0])

                reward = np.log2(reward) / 15 if reward != 0 else 0
                reward_sum += reward
                states.append((next_state, reward_sum))

                next_state = self.game_state_to_input(next_state)

                if done:
                    next_state = None

                batch_rewards.append(torch.FloatTensor([reward]))
                batch_actions.append(torch.LongTensor([[action[0]]]))
                batch_states.append(state)
                batch_next_states.append(next_state)

                state = next_state

                if self.eps > self.eps_end:
                    self.eps -= (self.args.eps_start - self.eps_end) / self.args.eps_decay

                if done:
                    test = False
                    with self.lock:
                        self.episodes_counter.value += 1

                        if self.episodes_counter.value <= 1000:
                            self.last_1000.append(reward_sum)
                        else:
                            self.last_1000[self.episodes_counter.value % 1000] = reward_sum

                        if self.worker_id == 0 and self.episodes_counter.value > 100 * (self.count_tested + 1):
                            test = True

                    if test:
                        self.test_performance()

                    if len(states) > 50:
                        state, reward_sum = states[int(len(states) / 2)]
                        self.env.Matrix = state.reshape((4, 4))
                        states = [(state, reward_sum)]
                    else:
                        state = self.env.reset()
                        states = [(state, 0)]
                    state = self.game_state_to_input(state)

                    break

            self.optimize_model(batch_states, batch_rewards, batch_actions, batch_next_states)


def plot_graphs(rewards, loss):
    plt.figure(1)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(np.array(rewards))

    if len(rewards) >= 100:
        means = [np.array(rewards[i:i+100]).mean() for i in range(0, len(rewards), 100)]
        plt.plot(means)

    plt.pause(0.001)

    plt.figure(2)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(np.array(loss))

    plt.pause(0.001)
