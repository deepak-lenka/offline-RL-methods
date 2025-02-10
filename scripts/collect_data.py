"""Script for collecting offline RL data."""
import os
import argparse
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from src.data.collector import DataCollector

def parse_args():
    parser = argparse.ArgumentParser(description='Collect offline RL data')
    parser.add_argument('--env-id', type=str, default='Pendulum-v1',
                      help='Gym environment ID')
    parser.add_argument('--num-episodes', type=int, default=1000,
                      help='Number of episodes to collect')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='Directory to save data')
    return parser.parse_args()

def random_policy(env):
    """Random policy for data collection."""
    return env.action_space.sample()

def collect_episode(env, collector, policy=None):
    """Collect a single episode."""
    state, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        if policy is None:
            action = random_policy(env)
        else:
            action = policy(state)
        
        next_state, reward, done, truncated, _ = env.step(action)
        
        collector.collect_interaction(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done or truncated
        )
        
        state = next_state

def main():
    args = parse_args()
    
    # Create data collector
    os.makedirs(args.output_dir, exist_ok=True)
    collector = DataCollector(storage_path=args.output_dir)
    
    # Collect data
    print(f"Collecting {args.num_episodes} episodes from {args.env_id}...")
    collector.collect_episodes(
        env_name=args.env_id,
        num_episodes=args.num_episodes,
        random=True
    )
    
    # Save data
    print(f"Saving data to {args.output_dir}...")
    collector.save_buffer()
    print("Done!")
    
if __name__ == '__main__':
    main()
