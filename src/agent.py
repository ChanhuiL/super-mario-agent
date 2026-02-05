"""
Mario Agent implementing Rainbow DQN with all the enhancements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from .model import RainbowDQN
from .replay import PrioritizedReplayBuffer


class MarioAgent:
    """
    Rainbow DQN Agent for Super Mario Bros.
    
    Features:
    - Double DQN
    - Dueling architecture
    - Distributional RL (C51)
    - Noisy Nets (replaces epsilon-greedy)
    - Prioritized Experience Replay
    - Multi-step returns
    """
    def __init__(self, num_actions, device='cuda', save_dir=None):
        self.num_actions = num_actions
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.batch_size = 32
        self.lr = 0.00025
        self.target_update_freq = 10000  # Update target network every N steps
        self.learning_starts = 50000  # Start learning after N steps
        self.n_step = 3  # Multi-step returns
        
        # Distributional RL parameters
        self.num_atoms = 51
        self.v_min = -20
        self.v_max = 150
        
        # Networks
        self.online_net = RainbowDQN(
            num_actions=num_actions,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)
        
        self.target_net = RainbowDQN(
            num_actions=num_actions,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)
        
        # Initialize target network with online network weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=100000
        )
        
        # Multi-step buffer
        self.n_step_buffer = []
        
        # Counters
        self.steps = 0
        self.episodes = 0
        
        # Save directory
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def act(self, frames, action_history):
        """
        Select action using the online network.
        Uses noisy layers for exploration (no epsilon-greedy needed).
        
        Args:
            frames: (4, 84, 84) numpy array
            action_history: (8,) numpy array
        
        Returns:
            action: int
        """
        # Convert to tensors
        frames = torch.FloatTensor(frames).unsqueeze(0).to(self.device) / 255.0
        action_history = torch.LongTensor(action_history).unsqueeze(0).to(self.device)
        
        # Reset noise for exploration
        self.online_net.reset_noise()
        
        with torch.no_grad():
            q_values, _ = self.online_net(frames, action_history)
            action = q_values.argmax(1).item()
        
        return action

    def store_transition(self, frames, action_history, action, reward, 
                        next_frames, next_action_history, done):
        """
        Store transition in n-step buffer and replay memory.
        """
        # Add to n-step buffer
        self.n_step_buffer.append(
            (frames, action_history, action, reward, next_frames, next_action_history, done)
        )
        
        # If we have enough steps, compute n-step return and store
        if len(self.n_step_buffer) >= self.n_step:
            # Get the oldest transition
            frames_0, ah_0, action_0, _, _, _, _ = self.n_step_buffer[0]
            _, _, _, _, frames_n, ah_n, done_n = self.n_step_buffer[-1]
            
            # Compute n-step return
            n_step_reward = sum([self.gamma ** i * t[3] for i, t in enumerate(self.n_step_buffer)])
            
            # Store in replay memory
            self.memory.push(frames_0, ah_0, action_0, n_step_reward, 
                           frames_n, ah_n, done_n)
            
            # Remove oldest transition
            self.n_step_buffer.pop(0)
        
        # If episode done, flush remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                frames_0, ah_0, action_0, _, _, _, _ = self.n_step_buffer[0]
                _, _, _, _, frames_n, ah_n, done_n = self.n_step_buffer[-1]
                
                n_step_reward = sum([self.gamma ** i * t[3] for i, t in enumerate(self.n_step_buffer)])
                
                self.memory.push(frames_0, ah_0, action_0, n_step_reward,
                               frames_n, ah_n, done_n)
                
                self.n_step_buffer.pop(0)

    def learn(self):
        """
        Sample from replay buffer and update network.
        Uses distributional loss (C51) and Double DQN.
        
        Returns:
            loss: float or None
        """
        if len(self.memory) < self.learning_starts:
            return None
        
        # Sample batch
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        # Unpack batch
        frames = torch.FloatTensor(np.array([t.frames for t in batch])).to(self.device) / 255.0
        action_history = torch.LongTensor(np.array([t.action_history for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t.action for t in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).to(self.device)
        next_frames = torch.FloatTensor(np.array([t.next_frames for t in batch])).to(self.device) / 255.0
        next_action_history = torch.LongTensor(np.array([t.next_action_history for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q distribution
        _, current_dist = self.online_net(frames, action_history)
        current_dist = current_dist[range(self.batch_size), actions.squeeze()]
        
        # Next Q values for Double DQN (use online network to select action)
        with torch.no_grad():
            next_q_values, _ = self.online_net(next_frames, next_action_history)
            next_actions = next_q_values.argmax(1)
            
            # Get distribution from target network
            _, next_dist = self.target_net(next_frames, next_action_history)
            next_dist = next_dist[range(self.batch_size), next_actions]
            
            # Compute target distribution (Categorical DQN)
            target_dist = self._get_target_dist(rewards, next_dist, dones)
        
        # Cross-entropy loss
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(1)
        
        # Importance sampling weights
        loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors = -(target_dist * torch.log(current_dist + 1e-8)).sum(1)
            self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
        # Reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        return loss.item()

    def _get_target_dist(self, rewards, next_dist, dones):
        """
        Compute target distribution for C51.
        
        Args:
            rewards: (batch,)
            next_dist: (batch, num_atoms)
            dones: (batch,)
        
        Returns:
            target_dist: (batch, num_atoms)
        """
        atoms = self.online_net.atoms
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # Compute projected distribution
        target_dist = torch.zeros(self.batch_size, self.num_atoms, device=self.device)
        
        for i in range(self.batch_size):
            # print(rewards[i])
            if dones[i]:
                # Terminal state: all mass at reward
                Tz = rewards[i].clamp(self.v_min, self.v_max)
                b = (Tz - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                target_dist[i, l] += (u - b)
                target_dist[i, u] += (b - l)
            else:
                # Non-terminal: project distribution
                for j in range(self.num_atoms):
                    Tz = rewards[i] + self.gamma ** self.n_step * atoms[j]
                    Tz = Tz.clamp(self.v_min, self.v_max)
                    b = (Tz - self.v_min) / delta_z
                    l = b.floor().long()
                    u = b.ceil().long()
                    
                    target_dist[i, l] += next_dist[i, j] * (u - b)
                    target_dist[i, u] += next_dist[i, j] * (b - l)
        
        return target_dist

    def update_target_network(self):
        """Copy weights from online network to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, suffix=''):
        """Save model checkpoint."""
        save_path = self.save_dir / f'mario_net_{suffix}.pth'
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, save_path)
        print(f'Saved checkpoint to {save_path}')

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f'Loaded checkpoint from {path}')
