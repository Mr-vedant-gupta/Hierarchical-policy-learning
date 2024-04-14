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


class DiscreteToOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_obs = env.observation_space.n
        self.observation_space = gym.spaces.Box(shape=(self.n_obs,), low=0, high=1, dtype=np.uint8)

    def observation(self, observation: Any) -> Any:
        zeros = np.zeros(self.n_obs)
        zeros[observation] = 1
        return zeros
    

def make_env(env_name):

    if env_name == 'fourrooms':
        return Fourrooms(), False

    enabled_atari_envs = [
        "ALE/Asterix-v5",
        "ALE/MsPacman-v5",
        "ALE/Seaquest-v5",
        "ALE/Zaxxon-v5",
    ]

    env = gym.make(env_name)
    is_atari = "atari" in env.spec.entry_point
    if is_atari:
        assert env_name in enabled_atari_envs, env_name
        env = gym.make(env_name, frameskip=1)
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
        env = FrameStack(env, 4)

    if env_name == "Taxi":
        env = DiscreteToOneHotWrapper(env)

    print(f"--- created {str(env)}, atari={is_atari} ---")
    return env, is_atari

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
