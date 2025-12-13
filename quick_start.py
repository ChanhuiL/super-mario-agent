"""
Quick start script for training Super Mario Agent.
This is a simplified version for quick testing.
"""

from environment import make_mario_env
from agent import MarioAgent
import torch

def quick_train(max_episodes=100):
    """Quick training run for testing."""
    
    print("=" * 60)
    print("Super Mario Agent - Quick Start")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create environment
    print("Creating environment...")
    env = make_mario_env(
        env_name='SuperMarioBros-1-1-v0',
        skip_frames=4,
        stack_frames=4
    )
    
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {num_actions}\n")
    
    # Create agent (simplified - no Rainbow features for quick start)
    print("Initializing agent...")
    agent = MarioAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        device=device,
        lr=2.5e-4,
        use_noisy=False,  # Use epsilon-greedy for simplicity
        use_stn=False,    # Disable STN for faster training
        use_prioritized=False,  # Use standard replay buffer
        target_update=1000,  # Update more frequently for quick testing
        epsilon_decay=100_000  # Faster epsilon decay
    )
    print("Agent initialized!\n")
    
    # Training loop
    print("Starting training...")
    print("Press Ctrl+C to stop early\n")
    print("-" * 60)
    
    episode = 0
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    
    try:
        while episode < max_episodes:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.memory) > 1000:  # Start training after small buffer
                loss = agent.update(batch_size=32)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Episode finished
            if done:
                episode += 1
                agent.episodes = episode
                
                # Print progress
                epsilon = agent._get_epsilon()
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Length: {episode_length:4d} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Steps: {agent.steps}")
                
                # Reset for next episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                agent.action_history.clear()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save model
    save_path = 'quick_start_model.pt'
    agent.save(save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to train')
    args = parser.parse_args()
    
    quick_train(max_episodes=args.episodes)

