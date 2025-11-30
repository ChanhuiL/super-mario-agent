"""
Experience replay buffer with prioritized sampling.
"""

import numpy as np
import torch
from collections import namedtuple
import random


Transition = namedtuple('Transition', 
    ('state', 'action', 'reward', 'next_state', 'done', 'action_history'))


class ReplayBuffer:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions with probability proportional to their TD error.
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6):
        """
        Args:
            capacity: Maximum size of buffer
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (starts at beta, goes to 1)
            beta_increment: How much to increment beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
    
    def push(self, *args):
        """Save a transition with maximum priority."""
        max_priority = self.priorities.max() if len(self.memory) > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions with priorities."""
        if len(self.memory) == 0:
            return []
        
        # Sample indices based on priorities
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        batch = [self.memory[idx] for idx in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory)


class NStepReplayBuffer:
    """N-step replay buffer for multi-step returns."""
    
    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = []
        self.n_step_buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, action_history=None):
        """Push transition and compute n-step return if ready."""
        transition = (state, action, reward, next_state, done, action_history)
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Compute n-step return
        n_step_reward = 0
        for i in range(self.n_step):
            n_step_reward += (self.gamma ** i) * self.n_step_buffer[i][2]
        
        n_step_state = self.n_step_buffer[0][0]
        n_step_action = self.n_step_buffer[0][1]
        n_step_next_state = self.n_step_buffer[-1][3]
        n_step_done = self.n_step_buffer[-1][4]
        n_step_action_history = self.n_step_buffer[0][5]
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Transition(
            n_step_state, n_step_action, n_step_reward,
            n_step_next_state, n_step_done, n_step_action_history
        )
        self.position = (self.position + 1) % self.capacity
        
        # Remove oldest transition
        if not n_step_done:
            self.n_step_buffer.pop(0)
        else:
            self.n_step_buffer = []
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

