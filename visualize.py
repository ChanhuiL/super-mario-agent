"""
GUI for visualizing agent performance with real-time metrics and Q-value distributions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from pathlib import Path
from collections import deque

from src.wrappers import create_mario_env
from src.agent import MarioAgent


class MarioVisualizer:
    """Real-time visualization of Mario agent performance."""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
        # Metrics tracking
        self.rewards_history = deque(maxlen=100)
        self.q_values_history = deque(maxlen=100)
        self.actions_history = deque(maxlen=50)
        self.episode_rewards = []
        
        # Current episode state
        self.current_reward = 0
        self.current_step = 0
        self.episode_num = 0
        
        # Setup figure
        self.setup_figure()
        
    def setup_figure(self):
        """Create the visualization figure."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create grid
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Game screen (top-left, large)
        self.ax_screen = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_screen.set_title('Game Screen', fontsize=14, fontweight='bold')
        self.ax_screen.axis('off')
        
        # Q-value distribution (top-right)
        self.ax_q_dist = self.fig.add_subplot(gs[0, 2])
        self.ax_q_dist.set_title('Q-Value Distribution', fontsize=12)
        self.ax_q_dist.set_xlabel('Action')
        self.ax_q_dist.set_ylabel('Expected Q-Value')
        
        # Action history (middle-right)
        self.ax_actions = self.fig.add_subplot(gs[1, 2])
        self.ax_actions.set_title('Action History', fontsize=12)
        self.ax_actions.set_xlabel('Time Step')
        self.ax_actions.set_ylabel('Action')
        
        # Reward over time (bottom-left)
        self.ax_reward = self.fig.add_subplot(gs[2, 0])
        self.ax_reward.set_title('Episode Reward', fontsize=12)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Total Reward')
        
        # Q-value over time (bottom-middle)
        self.ax_q_time = self.fig.add_subplot(gs[2, 1])
        self.ax_q_time.set_title('Average Q-Value', fontsize=12)
        self.ax_q_time.set_xlabel('Step')
        self.ax_q_time.set_ylabel('Avg Q-Value')
        
        # Metrics text (bottom-right)
        self.ax_metrics = self.fig.add_subplot(gs[2, 2])
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Metrics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
    
    def update_visualization(self, frame_rgb, q_values, action, info):
        """Update all visualization components."""
        # Clear all axes
        self.ax_screen.clear()
        self.ax_q_dist.clear()
        self.ax_actions.clear()
        self.ax_reward.clear()
        self.ax_q_time.clear()
        self.ax_metrics.clear()
        
        # 1. Game screen
        self.ax_screen.imshow(frame_rgb)
        self.ax_screen.set_title(f'Game Screen - Step {self.current_step}', 
                                fontsize=14, fontweight='bold')
        self.ax_screen.axis('off')
        
        # 2. Q-value distribution
        action_names = ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'A', 'Left']
        colors = ['red' if i == action else 'skyblue' for i in range(len(q_values))]
        self.ax_q_dist.bar(range(len(q_values)), q_values, color=colors)
        self.ax_q_dist.set_xticks(range(len(q_values)))
        self.ax_q_dist.set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
        self.ax_q_dist.set_title('Q-Value Distribution', fontsize=12)
        self.ax_q_dist.set_ylabel('Expected Q-Value')
        self.ax_q_dist.grid(axis='y', alpha=0.3)
        
        # 3. Action history
        if len(self.actions_history) > 0:
            steps = list(range(len(self.actions_history)))
            self.ax_actions.plot(steps, list(self.actions_history), 
                                marker='o', linestyle='-', markersize=4)
            self.ax_actions.set_yticks(range(7))
            self.ax_actions.set_yticklabels(['N', 'R', 'R+A', 'R+B', 'R+AB', 'A', 'L'], 
                                           fontsize=8)
            self.ax_actions.set_title('Recent Actions', fontsize=12)
            self.ax_actions.set_xlabel('Time Step')
            self.ax_actions.grid(True, alpha=0.3)
        
        # 4. Episode rewards
        if len(self.episode_rewards) > 0:
            self.ax_reward.plot(self.episode_rewards, marker='o', markersize=3)
            self.ax_reward.set_title('Episode Rewards', fontsize=12)
            self.ax_reward.set_xlabel('Episode')
            self.ax_reward.set_ylabel('Total Reward')
            self.ax_reward.grid(True, alpha=0.3)
        
        # 5. Q-values over time
        if len(self.q_values_history) > 0:
            self.ax_q_time.plot(list(self.q_values_history), color='green')
            self.ax_q_time.set_title('Average Q-Value', fontsize=12)
            self.ax_q_time.set_xlabel('Recent Steps')
            self.ax_q_time.set_ylabel('Avg Q')
            self.ax_q_time.grid(True, alpha=0.3)
        
        # 6. Metrics text
        metrics_text = f"""
Episode: {self.episode_num}
Step: {self.current_step}
Reward: {self.current_reward:.1f}

Position: {info.get('x_pos', 0)}
Time: {info.get('time', 0)}

Selected Action:
  {action_names[action]}
  
Max Q-Value:
  {q_values.max():.2f}
        """
        self.ax_metrics.text(0.1, 0.5, metrics_text, 
                            fontsize=11, verticalalignment='center',
                            family='monospace')
        self.ax_metrics.axis('off')
        
        plt.draw()
        plt.pause(0.001)
    
    def run_episode(self):
        """Run a single episode with visualization."""
        obs = self.env.reset()
        done = False
        
        self.current_reward = 0
        self.current_step = 0
        self.episode_num += 1
        
        self.agent.online_net.eval()
        
        print(f"\nStarting Episode {self.episode_num}...")
        
        while not done:
            # Get state
            frames = obs['frames']
            action_history = obs['action_history']
            
            # Get Q-values and action
            with torch.no_grad():
                frames_t = torch.FloatTensor(frames).unsqueeze(0).to(self.agent.device) / 255.0
                ah_t = torch.LongTensor(action_history).unsqueeze(0).to(self.agent.device)
                q_values, _ = self.agent.online_net(frames_t, ah_t)
                q_values_np = q_values.cpu().numpy()[0]
                action = q_values_np.argmax()
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.current_reward += reward
            self.current_step += 1
            
            # Update tracking
            self.actions_history.append(action)
            self.q_values_history.append(q_values_np.mean())
            
            # Get RGB frame for visualization
            # Since we're using grayscale, convert back to RGB for display
            frame_rgb = np.stack([frames[-1]] * 3, axis=-1)  # Last frame, replicate to RGB
            
            # Update visualization
            self.update_visualization(frame_rgb, q_values_np, action, info)
        
        # Episode finished
        self.episode_rewards.append(self.current_reward)
        print(f"Episode {self.episode_num} finished!")
        print(f"  Reward: {self.current_reward:.1f}")
        print(f"  Steps: {self.current_step}")
        print(f"  Final Position: {info.get('x_pos', 0)}")
    
    def run(self, num_episodes=5):
        """Run multiple episodes."""
        for _ in range(num_episodes):
            self.run_episode()
            print("\nPress Enter to continue to next episode (or Ctrl+C to quit)...")
            try:
                input()
            except KeyboardInterrupt:
                print("\nStopping visualization...")
                break
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Mario agent performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print("=" * 60)
    print("SUPER MARIO RL AGENT - VISUALIZATION")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)
    
    # Create environment and agent
    print("\nInitializing environment and agent...")
    env = create_mario_env(world=1, stage=1, version="Vanilla")
    agent = MarioAgent(num_actions=7, device=args.device)
    agent.load(checkpoint_path)
    
    print("Starting visualization...\n")
    
    # Create and run visualizer
    visualizer = MarioVisualizer(env, agent)
    visualizer.run(num_episodes=args.episodes)
    
    env.close()


if __name__ == '__main__':
    main()
