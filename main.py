from __future__ import print_function

import time
from collections import namedtuple

import torch
import torch.multiprocessing as mp

from model import DQN
from shared_optimizer import SharedAdam
from worker import Worker

WorkerArguments = namedtuple('WorkerArguments', 'seed lr start_step gamma eps_start eps_decay num_steps max_grad_norm '
                                                'use_big_input')

if __name__ == '__main__':
    NUM_WORKERS = 16
    LOAD_FILE = ''

    SEED = 1337
    LR = 0.0001
    START_STEP = 0
    GAMMA = 0.9
    EPS_START = 0.9
    EPS_DECAY = 200000
    NUM_STEPS = 20
    MAX_GRAD_NORM = 50
    USE_BIG_INPUT = False

    args = WorkerArguments(SEED, LR, START_STEP, GAMMA, EPS_START, EPS_DECAY, NUM_STEPS, MAX_GRAD_NORM, USE_BIG_INPUT)

    torch.manual_seed(SEED)

    shared_model = DQN(16 if USE_BIG_INPUT else 1).float()
    shared_model.share_memory()
    if LOAD_FILE != '':
        shared_model.load_state_dict(torch.load('./trained_models/' + LOAD_FILE + '.pth')['model'])

    target_model = DQN(16 if USE_BIG_INPUT else 1).float()
    target_model.share_memory()
    target_model.load_state_dict(shared_model.state_dict())

    optimizer = SharedAdam(shared_model.parameters(), lr=LR)
    optimizer.share_memory()
    if LOAD_FILE != '':
        optimizer.load_state_dict(torch.load('./trained_models/' + LOAD_FILE + '.pth')['optimizer'])

    processes = []

    with mp.Manager() as manager:
        lock = mp.Lock()
        move_counter = mp.Value('i', 0)
        episodes_counter = mp.Value('i', 0)
        last_1000 = manager.list()
        last_1000_loss = manager.list()
        start_time = time.time()

        for worker_id in range(NUM_WORKERS):
            eps_end = 0.5 if worker_id < 5 else (0.01 if worker_id < 10 else 0.1)
            p = Worker(shared_model, target_model, optimizer, move_counter, episodes_counter, start_time, lock, args,
                       last_1000, last_1000_loss, worker_id, eps_end)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
