"""Script to evaluate a trained CQL model."""
import os
import argparse
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.cql_model import CQLModel

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained CQL model')
    parser.add_argument('--model-dir', type=str, required=True,
                      help='Directory containing trained model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                      help='Path to config file')
    parser.add_argument('--num-episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_episode(model: CQLModel, initial_state: np.ndarray, max_steps: int = 200) -> float:
    """Evaluate model for one episode."""
    total_reward = 0
    state = initial_state
    
    for _ in range(max_steps):
        # Get action from model
        action = model.predict(state.reshape(1, -1))[0]
        
        # Simulate pendulum dynamics (simplified)
        theta = np.arctan2(state[1], state[0])
        theta_dot = state[2]
        
        # Apply action
        dt = 0.05
        g = 10.0
        l = 1.0
        m = 1.0
        
        # Update state
        theta_ddot = (-3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * action[0])
        theta_dot_new = theta_dot + theta_ddot * dt
        theta_new = theta + theta_dot_new * dt
        
        # Compute reward
        reward = -(theta_new**2 + 0.1 * theta_dot_new**2 + 0.001 * action[0]**2)
        total_reward += reward
        
        # Update state
        state = np.array([
            np.cos(theta_new),
            np.sin(theta_new),
            theta_dot_new
        ])
        
        # Check if episode should end
        if abs(theta_new) < 0.1 and abs(theta_dot_new) < 0.1:
            break
    
    return total_reward

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Load model
    model_path = Path(args.model_dir) / 'best_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    
    # Initialize model with config
    model_config = config['model'].copy()
    model_config['state_dim'] = 3
    model_config['action_dim'] = 1
    model = CQLModel(**model_config)
    model.load(str(model_path))
    
    # Evaluate model
    rewards = []
    print("\nEvaluating model...")
    
    for episode in range(args.num_episodes):
        # Random initial state
        theta = np.random.uniform(-np.pi, np.pi)
        theta_dot = np.random.uniform(-8, 8)
        initial_state = np.array([np.cos(theta), np.sin(theta), theta_dot])
        
        # Run episode
        episode_reward = evaluate_episode(model, initial_state)
        rewards.append(episode_reward)
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, 'b-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Save plot
    plot_path = Path(args.model_dir) / 'evaluation.png'
    plt.savefig(plot_path)
    print(f"\nEvaluation plot saved to {plot_path}")
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")

if __name__ == '__main__':
    main()
