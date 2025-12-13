"""
Environment wrappers for Super Mario Bros preprocessing.
Includes grayscale conversion, resizing, frame stacking, and frame skipping.
"""

import gym
import numpy as np
from collections import deque
from gym.spaces import Box
from torchvision import transforms
import torch


class PreprocessFrame(gym.ObservationWrapper):
    """Convert RGB frames to grayscale and resize to 84x84."""
    
    def __init__(self, env):
        super().__init__(env)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])
        self.observation_space = Box(
            low=0, high=1, 
            shape=(1, 84, 84), 
            dtype=np.float32
        )
    
    def observation(self, obs):
        frame = self.transform(obs)
        return frame.squeeze(0).numpy()  # Remove channel dimension for numpy


class FrameStack(gym.Wrapper):
    """Stack k consecutive frames together."""
    
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=1,
            shape=(k, shp[1], shp[2]),
            dtype=np.float32
        )
    
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, done, info


class FrameSkip(gym.Wrapper):
    """Skip k frames, repeating the last action."""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame, taking max over last 2 frames."""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_mario_env(env_name='SuperMarioBros-1-1-v0', skip_frames=4, stack_frames=4):
    """
    Create a fully wrapped Super Mario Bros environment.
    
    Args:
        env_name: Gym environment name
        skip_frames: Number of frames to skip (repeat action)
        stack_frames: Number of frames to stack together
    
    Returns:
        Wrapped gym environment
    """
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    except ImportError:
        raise ImportError(
            "Please install gym-super-mario-bros: pip install gym-super-mario-bros"
        )
    
    env = gym.make(env_name, render_mode='rgb_array')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=skip_frames)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=stack_frames)
    
    return env

