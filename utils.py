import gymnasium as gym
import numpy as np
import torch

from typing import Any
from gymnasium.wrappers import AtariPreprocessing, TransformReward
from gymnasium.wrappers import FrameStack as FrameStack_

from fourrooms import Fourrooms


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class MultiTaskOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_obs = env.observation_space.n
        self.n_taxi_pos = 25 #TODO
        self.n_passenger_loc = 5
        self.n_destinations = 4
        self.observation_space = gym.spaces.Box(shape=(self.n_obs,), low=0, high=1, dtype=np.uint8)

    def observation(self, observation: Any) -> Any:
        destination = observation % 4
        rem = (observation - destination)/4
        passenger_location = rem%5
        #TODO: could this cause rounding errors?
        rem = (rem - passenger_location)/5
        taxi_pos = int(rem)
        zeros_state = np.zeros(self.n_taxi_pos + 1)
        zeros_state[taxi_pos] = 1
        if passenger_location == 4:
            zeros_state[-1] == 1
        zeros_task = np.zeros(self.n_obs)
        zeros_task[observation] = 1
        return [zeros_task, zeros_state]
    

def make_env(env_name, render_mode):

    if env_name == 'fourrooms':
        return Fourrooms(), False

    enabled_atari_envs = [
        "ALE/Asterix-v5",
        "ALE/MsPacman-v5",
        "ALE/Seaquest-v5",
        "ALE/Zaxxon-v5",
    ]

    env = gym.make(env_name, render_mode=render_mode)
    is_atari = "atari" in env.spec.entry_point
    if is_atari:
        assert env_name in enabled_atari_envs, env_name
        env = gym.make(env_name, frameskip=1)
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
        env = FrameStack(env, 4)
    breakpoint()
    if env_name == "Taxi":
        env = MultiTaskOneHotWrapper(env)

    print(f"--- created {str(env)}, atari={is_atari} ---")
    return env, is_atari

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
