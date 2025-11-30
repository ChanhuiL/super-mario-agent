"""
Training script for Super Mario Agent with monitoring and evaluation.
"""

import torch
import numpy as np
import argparse
from collections import deque
import os
from datetime import datetime
import json

from environment import make_mario_env
from agent import MarioAgent


class MetricsLogger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon': [],
            'steps': []
        }
    
    def log(self, **kwargs):
        """Log metrics."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save(self, filename='metrics.json'):
        """Save metrics to file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self, episode):
        """Print summary of recent performance."""
        if len(self.metrics['episode_rewards']) > 0:
            recent_rewards = self.metrics['episode_rewards'][-100:]
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
            print(f"Episode {episode} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Max Reward: {max_reward:.2f} | "
                  f"Steps: {self.metrics['steps'][-1] if self.metrics['steps'] else 0}")


def train(args):
    """Main training loop."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = make_mario_env(
        env_name=args.env_name,
        skip_frames=args.skip_frames,
        stack_frames=args.stack_frames
    )
    
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    print(f"State shape: {state_shape}, Actions: {num_actions}")
    
    # Create agent
    print("Initializing agent...")
    agent = MarioAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        use_noisy=args.use_noisy,
        use_stn=args.use_stn,
        use_prioritized=args.use_prioritized,
        n_step=args.n_step,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        action_history_length=args.action_history_length
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        agent.load(args.load_checkpoint)
    
    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"mario_{timestamp}")
    logger = MetricsLogger(log_dir)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    episode_reward = 0
    episode_length = 0
    state = env.reset()
    
    best_reward = float('-inf')
    episode = 0
    
    try:
        while agent.steps < args.max_steps:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.memory) > args.min_buffer_size:
                loss = agent.update(batch_size=args.batch_size)
                if loss is not None:
                    logger.log(losses=loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Log metrics
            if agent.steps % args.log_interval == 0:
                epsilon = agent._get_epsilon() if not args.use_noisy else 0.0
                logger.log(epsilon=epsilon, steps=agent.steps)
            
            # Episode finished
            if done:
                episode += 1
                agent.episodes = episode
                
                logger.log(
                    episode_rewards=episode_reward,
                    episode_lengths=episode_length
                )
                
                # Print progress
                if episode % args.print_interval == 0:
                    logger.print_summary(episode)
                
                # Save checkpoint
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    checkpoint_path = os.path.join(log_dir, 'best_model.pt')
                    agent.save(checkpoint_path)
                    print(f"New best reward: {best_reward:.2f}, saved to {checkpoint_path}")
                
                # Periodic checkpoint
                if episode % args.save_interval == 0:
                    checkpoint_path = os.path.join(log_dir, f'checkpoint_ep{episode}.pt')
                    agent.save(checkpoint_path)
                
                # Reset for next episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                agent.action_history.clear()
        
        print("\nTraining completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Save final model and metrics
        final_path = os.path.join(log_dir, 'final_model.pt')
        agent.save(final_path)
        logger.save()
        print(f"\nFinal model saved to {final_path}")
        print(f"Metrics saved to {log_dir}/metrics.json")


def evaluate(args):
    """Evaluate a trained agent."""
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Create environment
    env = make_mario_env(
        env_name=args.env_name,
        skip_frames=args.skip_frames,
        stack_frames=args.stack_frames
    )
    
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Create agent
    agent = MarioAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        device=device,
        use_noisy=False,  # Disable noise for evaluation
        use_stn=args.use_stn,
        use_prioritized=False
    )
    
    # Load checkpoint
    if not args.load_checkpoint:
        raise ValueError("Must provide --load_checkpoint for evaluation")
    
    agent.load(args.load_checkpoint)
    agent.policy_net.eval()
    
    print(f"Evaluating agent from {args.load_checkpoint}")
    print("=" * 60)
    
    num_episodes = args.eval_episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Length: {episode_length}")
    
    print("=" * 60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Super Mario Agent')
    
    # Environment
    parser.add_argument('--env-name', type=str, default='SuperMarioBros-1-1-v0',
                        help='Gym environment name')
    parser.add_argument('--skip-frames', type=int, default=4,
                        help='Number of frames to skip')
    parser.add_argument('--stack-frames', type=int, default=4,
                        help='Number of frames to stack')
    
    # Training
    parser.add_argument('--max-steps', type=int, default=10_000_000,
                        help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--min-buffer-size', type=int, default=50_000,
                        help='Minimum buffer size before training')
    
    # Exploration
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.1,
                        help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=int, default=1_000_000,
                        help='Epsilon decay steps')
    parser.add_argument('--use-noisy', action='store_true',
                        help='Use noisy networks instead of epsilon-greedy')
    
    # Network
    parser.add_argument('--use-stn', action='store_true',
                        help='Use Spatial Transformer Network')
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='Number of atoms for distributional RL')
    parser.add_argument('--v-min', type=float, default=-10,
                        help='Minimum value for distributional RL')
    parser.add_argument('--v-max', type=float, default=10,
                        help='Maximum value for distributional RL')
    parser.add_argument('--action-history-length', type=int, default=4,
                        help='Length of action history')
    
    # Replay buffer
    parser.add_argument('--use-prioritized', action='store_true',
                        help='Use prioritized experience replay')
    parser.add_argument('--n-step', type=int, default=3,
                        help='N-step returns')
    
    # Target network
    parser.add_argument('--target-update', type=int, default=10_000,
                        help='Target network update frequency')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='Logging interval')
    parser.add_argument('--print-interval', type=int, default=10,
                        help='Print interval (episodes)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval (episodes)')
    
    # Checkpointing
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation instead of training')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    
    # Misc
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args)
    else:
        train(args)

