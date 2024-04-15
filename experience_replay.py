import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done, action):
        self.buffer.append((obs, option, reward, next_obs, done, action))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done, action = zip(*self.rng.sample(self.buffer, batch_size))
        full_obs = [o[0] for o in obs]
        local_obs = [o[1] for o in obs]
        nfull_obs = [o[0] for o in next_obs]
        nlocal_obs = [o[1] for o in next_obs]
        return np.stack(full_obs), np.stack(local_obs), option, reward, np.stack(nfull_obs), np.stack(nlocal_obs), done, action

    def __len__(self):
        return len(self.buffer)
