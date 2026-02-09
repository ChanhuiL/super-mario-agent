"""
Training script for Super Mario RL Agent.
Implements the main training loop with logging and checkpointing.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

from src.wrappers import create_mario_env
from src.agent import MarioAgent


class MetricLogger:
    """Logger for training metrics."""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.save_dir / 'tensorboard')
        
        # Episode metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_q_values = []
        
        # Moving averages
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        
    def log_step(self, reward, loss, q_value):
        """Log step-level metrics."""
        self.ep_rewards.append(reward)
        if loss is not None:
            self.ep_avg_losses.append(loss)
        if q_value is not None:
            self.ep_avg_q_values.append(q_value)
    
    def log_episode(self, episode, steps, epsilon=None):
        """Log episode-level metrics."""
        ep_reward = sum(self.ep_rewards)
        ep_length = len(self.ep_rewards)
        ep_avg_loss = np.mean(self.ep_avg_losses) if self.ep_avg_losses else 0
        ep_avg_q = np.mean(self.ep_avg_q_values) if self.ep_avg_q_values else 0
        
        # Moving average (last 100 episodes)
        self.moving_avg_ep_rewards.append(ep_reward)
        self.moving_avg_ep_lengths.append(ep_length)
        if len(self.moving_avg_ep_rewards) > 100:
            self.moving_avg_ep_rewards.pop(0)
            self.moving_avg_ep_lengths.pop(0)
        
        avg_reward = np.mean(self.moving_avg_ep_rewards)
        avg_length = np.mean(self.moving_avg_ep_lengths)
        
        # Write to TensorBoard
        self.writer.add_scalar('Episode/Reward', ep_reward, episode)
        self.writer.add_scalar('Episode/Length', ep_length, episode)
        self.writer.add_scalar('Episode/Average_Loss', ep_avg_loss, episode)
        self.writer.add_scalar('Episode/Average_Q', ep_avg_q, episode)
        self.writer.add_scalar('Episode/Moving_Avg_Reward', avg_reward, episode)
        self.writer.add_scalar('Episode/Moving_Avg_Length', avg_length, episode)
        
        if epsilon is not None:
            self.writer.add_scalar('Train/Epsilon', epsilon, episode)
        
        # Reset episode metrics
        self.ep_rewards.clear()
        self.ep_lengths.clear()
        self.ep_avg_losses.clear()
        self.ep_avg_q_values.clear()
        
        return ep_reward, ep_length, avg_reward, avg_length, ep_avg_loss


def train(
    num_episodes=10000,
    save_interval=500,
    log_interval=10,
    device='cuda',
    save_dir='./mario_runs'
):
    """
    Main training loop.
    
    Args:
        num_episodes: Total number of episodes to train
        save_interval: Save checkpoint every N episodes
        log_interval: Log metrics every N episodes
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints and logs
    """
    # Create save directory with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = Path(save_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    print("Initializing environment...")
    env = create_mario_env(world=1, stage=1, version='Vanilla')
    num_actions = env.action_space.n  # Access the base action space
    
    # Initialize agent
    print("Initializing agent...")
    agent = MarioAgent(
        num_actions=num_actions,
        device=device,
        save_dir=save_dir / 'checkpoints'
    )
    
    # Initialize logger
    logger = MetricLogger(save_dir)
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"Save directory: {save_dir}")
    print(f"Action space size: {num_actions}\n")

    start_time = time.perf_counter()
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        # Reset environment
        obs = env.reset()
        frames = obs['frames']
        action_history = obs['action_history']
        
        done = False
        episode_reward = 0
        episode_steps = 0
        update_interval = 5
        loss = 0
        
        while not done:
            # Select action
            action = agent.act(frames, action_history)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_frames = next_obs['frames']
            next_action_history = next_obs['action_history']
            
            done = terminated or truncated
            true_reward = reward
            reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
            
            # Store transition
            agent.store_transition(
                frames, action_history, action, reward,
                next_frames, next_action_history, done
            )
            
            # Learn
            if update_interval > 0:
                update_interval -= 1
            else:
                loss = agent.learn()
                update_interval = 5
            
            # Get Q-value for logging
            with torch.no_grad():
                frames_t = torch.FloatTensor(frames).unsqueeze(0).to(agent.device) / 255.0
                ah_t = torch.LongTensor(action_history).unsqueeze(0).to(agent.device)
                q_values, _ = agent.online_net(frames_t, ah_t)
                max_q = q_values.max().item()
            
            # Log step
            logger.log_step(true_reward, loss, max_q)
            
            # Update state
            frames = next_frames
            action_history = next_action_history
            episode_reward += reward
            episode_steps += 1
            agent.steps += 1
            
            # Update target network
            if agent.steps % agent.target_update_freq == 0:
                agent.update_target_network()
        
        # Log episode
        agent.episodes = episode
        ep_reward, ep_length, avg_reward, avg_length, avg_loss = logger.log_episode(episode, agent.steps)
        
        # Print progress
        if episode % log_interval == 0:
            time_spent, start_time = (
                time.perf_counter() - start_time,
                time.perf_counter(),
            )
            print(f"Episode {episode:5d} | "
                  f"Steps: {agent.steps:7d} | "
                  f"Reward: {ep_reward:6.1f} | "
                  f"Length: {ep_length:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Time Spent: {time_spent:.4f}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            agent.save(suffix=f'episode_{episode}')
    
    # Save final checkpoint
    agent.save(suffix='final')
    print(f"\nTraining complete! Final checkpoint saved to {agent.save_dir}")
    
    env.close()
    logger.writer.close()


if __name__ == '__main__':
    # Training configuration
    config = {
        'num_episodes': 10000,
        'save_interval': 500,
        'log_interval': 10,
        'device': 'cuda',
        'save_dir': './mario_runs'
    }
    
    train(**config)
