"""Script to generate synthetic data for testing the RL model."""
import os
import numpy as np
from pathlib import Path

def generate_pendulum_data(num_samples: int = 10000) -> dict:
    """Generate synthetic data mimicking a pendulum environment.
    
    The state space consists of [cos(theta), sin(theta), theta_dot]
    The action space is [-2, 2] representing torque
    """
    # Generate random states
    thetas = np.random.uniform(-np.pi, np.pi, num_samples)
    theta_dots = np.random.uniform(-8, 8, num_samples)
    
    states = np.column_stack([
        np.cos(thetas),
        np.sin(thetas),
        theta_dots
    ])
    
    # Generate random actions (torque)
    actions = np.random.uniform(-2, 2, (num_samples, 1))
    
    # Compute next states using simplified pendulum dynamics
    dt = 0.05  # time step
    g = 10.0   # gravity
    l = 1.0    # pendulum length
    m = 1.0    # mass
    
    # Update theta and theta_dot using basic physics
    next_theta_dots = theta_dots + (3 * g / (2 * l) * np.sin(thetas) + 3.0 / (m * l**2) * actions.squeeze()) * dt
    next_thetas = thetas + next_theta_dots * dt
    
    next_states = np.column_stack([
        np.cos(next_thetas),
        np.sin(next_thetas),
        next_theta_dots
    ])
    
    # Compute rewards (want upright pendulum with minimal velocity and action)
    rewards = -(thetas**2 + 0.1 * theta_dots**2 + 0.001 * actions.squeeze()**2)
    
    # Terminal states (when pendulum is close to upright position with low velocity)
    dones = np.logical_and(
        np.abs(thetas) < 0.1,
        np.abs(theta_dots) < 0.1
    ).astype(np.float32)
    
    return {
        'states': states.astype(np.float32),
        'actions': actions.astype(np.float32),
        'next_states': next_states.astype(np.float32),
        'rewards': rewards.astype(np.float32),
        'dones': dones.astype(np.float32)
    }

def main():
    # Generate data
    print("Generating synthetic pendulum data...")
    data = generate_pendulum_data(num_samples=100000)
    
    # Create data directory
    data_dir = Path('data/pendulum')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    print(f"Saving data to {data_dir}...")
    for key, value in data.items():
        np.save(data_dir / f'{key}.npy', value)
    
    print("Data generation complete!")

if __name__ == '__main__':
    main()
