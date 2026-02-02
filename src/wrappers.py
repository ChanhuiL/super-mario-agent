"""
Environment wrappers for Super Mario Bros.
Includes preprocessing, frame stacking, and action history tracking.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from gymnasium.spaces import Box


class SkipFrame(gym.Wrapper):
    """
    Skip n frames and repeat the last action.
    This reduces the computational load and helps with temporal credit assignment.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleResize(gym.ObservationWrapper):
    """
    Convert frames to grayscale and resize to 84x84.
    Reduces dimensionality while preserving spatial information.
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, observation):
        # Convert RGB to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Resize to target shape
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return resized


class FrameStack(gym.Wrapper):
    """
    Stack the last k frames to capture temporal information.
    This helps the network understand motion and velocity.
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        
        # Update observation space to reflect stacked frames
        old_space = env.observation_space
        self.observation_space = Box(
            low=0, high=255,
            shape=(k, *old_space.shape),
            dtype=np.uint8
        )

    def reset(self):
        obs, info = self.env.reset()
        # Fill the deque with the initial frame
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self.frames, dtype=np.uint8)


class ActionHistoryWrapper(gym.Wrapper):
    """
    Track the history of the last k actions taken.
    This is used as an additional input to the neural network.
    Returns a dict with 'frames' and 'action_history'.
    """
    def __init__(self, env, k=8):
        super().__init__(env)
        self.k = k
        self.action_history = deque(maxlen=k)
        
        # Modify observation space to be a Dict space
        from gymnasium.spaces import Dict
        self.observation_space = Dict({
            'frames': env.observation_space,
            'action_history': Box(low=0, high=env.action_space.n-1, 
                                  shape=(k,), dtype=np.int32)
        })

    def reset(self):
        obs = self.env.reset()
        # Initialize action history with zeros (no action)
        self.action_history.clear()
        for _ in range(self.k):
            self.action_history.append(0)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action_history.append(action)
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, frames):
        return {
            'frames': frames,
            'action_history': np.array(self.action_history, dtype=np.int32)
        }


def create_mario_env(world=1, stage=1, version='Vanilla'):
    """
    Create and configure the Super Mario Bros environment with all wrappers.
    
    Args:
        world: World number (1-8)
        stage: Stage number (1-4)
        version: Version (0, 1, 2, or 3)
    
    Returns:
        Wrapped gym environment
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    
    # Create the base environment
    env_name = f'SuperMarioBros-{world}-{stage}-{version}'
    env = gym_super_mario_bros.make(env_name)
    
    # Limit the action space to simple movements
    # SIMPLE_MOVEMENT has 7 actions: [['NOOP'], ['right'], ['right', 'A'], 
    # ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply wrappers in order
    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env, shape=(84, 84))
    env = FrameStack(env, k=4)
    env = ActionHistoryWrapper(env, k=8)
    
    return env
