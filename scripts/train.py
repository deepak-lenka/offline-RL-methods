"""Main script for training offline RL models."""
import os
import argparse
import yaml
from pathlib import Path
import torch
import numpy as np

from src.data.collector import DataCollector
from src.training.trainer import OfflineRLTrainer
from src.evaluation.metrics import OfflineRLEvaluator
from src.deployment.deployer import ModelDeployer

def parse_args():
    parser = argparse.ArgumentParser(description='Train offline RL model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                      help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save models')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_dataloader(states, actions, rewards, next_states, dones, batch_size):
    """Create a dataloader for training."""
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield (
                states[batch_indices],
                actions[batch_indices],
                rewards[batch_indices],
                next_states[batch_indices],
                dones[batch_indices]
            )

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_path = Path(args.data_dir)
    states = np.load(data_path / 'states.npy')
    actions = np.load(data_path / 'actions.npy')
    rewards = np.load(data_path / 'rewards.npy')
    next_states = np.load(data_path / 'next_states.npy')
    dones = np.load(data_path / 'dones.npy')
    
    # Update state and action dimensions based on data
    config['model']['state_dim'] = states.shape[1]
    config['model']['action_dim'] = actions.shape[1]
    
    # Initialize components
    trainer = OfflineRLTrainer(
        algorithm="cql",
        model_params=config['model'],
        experiment_name=config['training']['experiment_name']
    )
    
    # Create evaluation dataset
    eval_dataset = {
        'states': states[:1000],  # Use first 1000 samples for evaluation
        'actions': actions[:1000],
        'rewards': rewards[:1000],
        'next_states': next_states[:1000],
        'dones': dones[:1000]
    }
    
    # Create dataloader
    batch_size = config['training'].get('batch_size', 256)
    steps_per_epoch = len(states) // batch_size
    dataloader = create_dataloader(
        states, actions, rewards, next_states, dones,
        batch_size=batch_size
    )
    
    # Train model
    print("Training model...")
    best_eval_reward = float('-inf')
    
    for epoch in range(config['training']['num_epochs']):
        epoch_metrics = []
        
        # Train for one epoch
        for _ in range(steps_per_epoch):
            batch = next(dataloader)
            metrics = trainer.model.train_step(
                states=batch[0],
                actions=batch[1],
                rewards=batch[2],
                next_states=batch[3],
                dones=batch[4]
            )
            epoch_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        print(f"Epoch {epoch}: {avg_metrics}")
        
        # Evaluate periodically
        if (epoch + 1) % config['training'].get('eval_interval', 10) == 0:
            print("\nEvaluating model...")
            eval_metrics = trainer.evaluate(eval_dataset)
            print(f"Evaluation metrics: {eval_metrics}")
            
            # Save if better
            if eval_metrics['eval_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['eval_reward']
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                trainer.model.save(save_path)
                print(f"New best model saved with reward: {best_eval_reward}\n")
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    trainer.model.load(best_model_path)
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = trainer.evaluate(eval_dataset)
    print(f"Final metrics: {final_metrics}")

if __name__ == '__main__':
    main()
