"""
Play Super Mario with a trained agent and visualize the gameplay.
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path

from src.wrappers import create_mario_env
from src.agent import MarioAgent


def play_episode(env, agent, render=True, record=False):
    """
    Play a single episode with the trained agent.
    
    Args:
        env: Mario environment
        agent: Trained agent
        render: Whether to render the gameplay
        record: Whether to record frames for video
    
    Returns:
        total_reward: Episode reward
        frames: List of frames (if record=True)
    """
    obs = env.reset()
    frames_list = []
    
    done = False
    total_reward = 0
    step = 0
    
    agent.online_net.eval()  # Set to evaluation mode
    
    while not done:
        # Get current state
        frames = obs['frames']
        action_history = obs['action_history']
        
        # Select action (deterministic)
        action = agent.act(frames, action_history)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1
        
        # Render
        if render:
            env.render()
            time.sleep(0.016)  # ~60 FPS
        
        # Record frame
        if record:
            frames_list.append(frames)
        
        # Print info
        if step % 100 == 0:
            print(f"Step {step:4d} | Reward: {total_reward:6.1f} | "
                  f"X-pos: {info.get('x_pos', 0):4d}")
    
    print(f"\nEpisode finished!")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Total Steps: {step}")
    print(f"Final X-Position: {info.get('x_pos', 0)}")
    print(f"Level Complete: {'Yes' if info.get('flag_get', False) else 'No'}")
    
    return total_reward, frames_list if record else None


def main():
    parser = argparse.ArgumentParser(description='Play Super Mario with trained agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--world', type=int, default=1,
                        help='World number (1-8)')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage number (1-4)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print("=" * 60)
    print("SUPER MARIO RL AGENT - PLAY MODE")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"World: {args.world}-{args.stage}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)
    
    # Create environment
    print("\nInitializing environment...")
    env = create_mario_env(world=args.world, stage=args.stage, version="Vanilla")
    
    # Get action space size
    # The action space is wrapped, so we need to get it from the base
    num_actions = 7  # SIMPLE_MOVEMENT has 7 actions
    
    # Create agent
    print("Loading agent...")
    agent = MarioAgent(num_actions=num_actions, device=args.device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    agent.load(checkpoint_path)
    
    print("\nReady to play!\n")
    
    # Play episodes
    rewards = []
    for episode in range(1, args.episodes + 1):
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{args.episodes}")
        print('='*60)
        
        reward, _ = play_episode(
            env, agent,
            render=not args.no_render,
            record=False
        )
        rewards.append(reward)
        
        if episode < args.episodes:
            print("\nPress Enter to continue to next episode...")
            input()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Episodes Played: {args.episodes}")
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Best Reward: {np.max(rewards):.1f}")
    print(f"Worst Reward: {np.min(rewards):.1f}")
    print("=" * 60)
    
    env.close()


if __name__ == '__main__':
    main()
