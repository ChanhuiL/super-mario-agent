"""
DQN Agent with Rainbow enhancements for Super Mario Bros.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class MarioAgent:
    """
    Rainbow DQN Agent with:
    - Double DQN
    - Dueling architecture
    - Distributional RL (C51)
    - Prioritized experience replay
    - N-step returns
    - Noisy networks (or epsilon-greedy)
    """
    
    def __init__(
        self,
        state_shape,
        num_actions,
        device,
        lr=2.5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1_000_000,
        target_update=10_000,
        use_noisy=True,
        use_stn=True,
        use_prioritized=True,
        n_step=3,
        num_atoms=51,
        v_min=-10,
        v_max=10,
        action_history_length=4
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.use_noisy = use_noisy
        self.use_prioritized = use_prioritized
        self.n_step = n_step
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.action_history_length = action_history_length
        
        # Networks
        from model import RainbowDQN
        self.policy_net = RainbowDQN(
            in_channels=state_shape[0],
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            use_noisy=use_noisy,
            use_stn=use_stn
        ).to(device)
        
        self.target_net = RainbowDQN(
            in_channels=state_shape[0],
            num_actions=num_actions,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            use_noisy=use_noisy,
            use_stn=use_stn
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        from replay_buffer import PrioritizedReplayBuffer, NStepReplayBuffer, ReplayBuffer
        
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(capacity=1_000_000, alpha=0.6, beta=0.4)
        else:
            self.memory = ReplayBuffer(capacity=1_000_000)
        
        # Action history
        self.action_history = deque(maxlen=action_history_length)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy or noisy network.
        
        Args:
            state: Current state [C, H, W]
            training: Whether in training mode
        
        Returns:
            action: Selected action index
        """
        if self.use_noisy:
            # Noisy networks handle exploration internally
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_history_tensor = self._get_action_history_tensor()
                q_values, _ = self.policy_net(state_tensor, action_history_tensor)
                action = q_values.argmax(1).item()
        else:
            # Epsilon-greedy
            epsilon = self._get_epsilon()
            if training and random.random() < epsilon:
                action = random.randrange(self.num_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_history_tensor = self._get_action_history_tensor()
                    q_values, _ = self.policy_net(state_tensor, action_history_tensor)
                    action = q_values.argmax(1).item()
        
        return action
    
    def _get_epsilon(self):
        """Compute current epsilon value."""
        if self.steps >= self.epsilon_decay:
            return self.epsilon_end
        return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
               (self.steps / self.epsilon_decay)
    
    def _get_action_history_tensor(self):
        """Convert action history deque to tensor."""
        if len(self.action_history) == 0:
            return None
        history = list(self.action_history)
        # Pad with zeros if needed
        while len(history) < self.action_history_length:
            history.insert(0, 0)
        return torch.LongTensor([history]).to(self.device)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        action_history = list(self.action_history) if len(self.action_history) > 0 else None
        self.memory.push(state, action, reward, next_state, done, action_history)
        self.action_history.append(action)
    
    def update(self, batch_size=32):
        """
        Update the network using a batch from replay buffer.
        
        Returns:
            loss: Training loss
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        if self.use_prioritized:
            batch, indices, weights = self.memory.sample(batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.memory.sample(batch_size)
            weights = torch.ones(batch_size).to(self.device)
            indices = None
        
        # Unpack batch
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Get action histories
        action_histories = []
        for e in batch:
            if e.action_history is not None and len(e.action_history) > 0:
                hist = list(e.action_history)
                while len(hist) < self.action_history_length:
                    hist.insert(0, 0)
                action_histories.append(hist)
            else:
                action_histories.append([0] * self.action_history_length)
        action_histories = torch.LongTensor(action_histories).to(self.device)
        
        # Current Q-values (distributional)
        q_values, probs = self.policy_net(states, action_histories)
        q_values = q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values using Double DQN
        with torch.no_grad():
            # Use policy net to select action
            next_q_values, _ = self.policy_net(next_states, action_histories)
            next_actions = next_q_values.argmax(1)
            
            # Use target net to evaluate
            _, next_probs = self.target_net(next_states, action_histories)
            next_probs = next_probs[range(batch_size), next_actions]  # [batch_size, num_atoms]
            
            # Compute target distribution
            rewards = rewards  # [batch_size, 1]
            dones = dones  # [batch_size, 1]
            atoms = self.policy_net.atoms.unsqueeze(0).unsqueeze(0)  # [1, 1, num_atoms]
            
            # Project onto support: T_z = r + γ * z (for each atom)
            # Shape: [batch_size, 1, num_atoms]
            target_atoms = rewards.unsqueeze(2) + (1 - dones.float().unsqueeze(2)) * self.gamma * atoms
            target_atoms = target_atoms.clamp(self.v_min, self.v_max)
            
            # Compute projection (C51 algorithm)
            atom_delta = (self.v_max - self.v_min) / (self.num_atoms - 1)
            target_z = (target_atoms - self.v_min) / atom_delta  # [batch_size, 1, num_atoms]
            target_z = target_z.squeeze(1)  # [batch_size, num_atoms]
            
            # Clamp to valid range
            target_z = target_z.clamp(0, self.num_atoms - 1)
            
            # Distribute probabilities using linear interpolation
            lower = target_z.floor().long()  # [batch_size, num_atoms]
            upper = target_z.ceil().long()   # [batch_size, num_atoms]
            
            # Handle edge cases
            lower = lower.clamp(0, self.num_atoms - 1)
            upper = upper.clamp(0, self.num_atoms - 1)
            
            # Initialize target distribution
            target_probs = torch.zeros(batch_size, self.num_atoms, device=self.device)
            
            # Compute weights for linear interpolation
            lower_weight = upper.float() - target_z  # [batch_size, num_atoms]
            upper_weight = target_z - lower.float()  # [batch_size, num_atoms]
            
            # Distribute probabilities (vectorized using scatter_add)
            # For each sample, distribute next_probs[i] to target_probs[i] at positions lower[i] and upper[i]
            for i in range(batch_size):
                # Get probabilities and indices for this sample
                probs = next_probs[i]  # [num_atoms]
                lower_idx = lower[i]   # [num_atoms]
                upper_idx = upper[i]   # [num_atoms]
                lower_w = lower_weight[i]  # [num_atoms]
                upper_w = upper_weight[i]  # [num_atoms]
                
                # Distribute probabilities
                target_probs[i].scatter_add_(0, lower_idx, probs * lower_w)
                target_probs[i].scatter_add_(0, upper_idx, probs * upper_w)
        
        # Compute loss (KL divergence for distributional RL)
        # probs shape: [batch_size, num_actions, num_atoms]
        # Get probabilities for taken actions: [batch_size, num_atoms]
        action_probs = probs[range(batch_size), actions]  # [batch_size, num_atoms]
        log_probs = torch.log(action_probs + 1e-8)  # [batch_size, num_atoms]
        
        # KL divergence: sum(target_probs * log(target_probs / action_probs))
        # = sum(target_probs * log(target_probs)) - sum(target_probs * log(action_probs))
        # We minimize the negative log likelihood (cross-entropy)
        loss = -torch.sum(target_probs * log_probs, dim=1)  # [batch_size]
        loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized and indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(q_values.squeeze() - torch.sum(target_probs * self.policy_net.atoms.unsqueeze(0), dim=1))
            self.memory.update_priorities(indices, td_errors.cpu().numpy())
        
        # Reset noise in noisy networks
        if self.use_noisy:
            self.policy_net.reset_noise()
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, filepath)
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)

