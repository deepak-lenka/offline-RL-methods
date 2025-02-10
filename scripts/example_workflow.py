"""Example workflow for offline RL training."""
import os
import gymnasium as gym
import numpy as np
from pathlib import Path

from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.models.cql_model import CQLModel
from src.training.trainer import OfflineRLTrainer
from src.evaluation.metrics import OfflineRLEvaluator
from src.deployment.deployer import ModelDeployer

def main():
    # Setup directories
    base_dir = Path(os.getcwd())
    data_dir = base_dir / "data"
    model_dir = base_dir / "models"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("1. Collecting data...")
    collector = DataCollector(storage_path=str(data_dir))
    
    # Collect some random trajectories
    for episode in range(100):
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            
            collector.collect_interaction(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done or truncated
            )
            
            state = next_state
    
    # Save collected data
    data_path = collector.save_buffer('initial_data.parquet')
    
    print("2. Preprocessing data...")
    # Load and preprocess data
    dataset = collector.load_dataset(data_path)
    preprocessor = DataPreprocessor()
    
    states = np.stack(dataset['state'])
    actions = np.stack(dataset['action'])
    rewards = np.array(dataset['reward'])
    next_states = np.stack(dataset['next_state'])
    dones = np.array(dataset['done'])
    
    preprocessor.fit(states, actions, rewards)
    states, actions, rewards, next_states = preprocessor.transform(
        states, actions, rewards, next_states
    )
    
    print("3. Creating and training model...")
    # Create and train model
    model = CQLModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=3e-4
    )
    
    model = CQLModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        learning_rate=3e-4
    )
    
    # Train for a few epochs
    for epoch in range(10):
        metrics = model.train_step(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )
        print(f"Epoch {epoch}: {metrics}")
        
        # Save periodically
        if (epoch + 1) % 5 == 0:
            model.save(str(model_dir / f"model_epoch_{epoch+1}.pt"))
    
    print("4. Evaluating model...")
    # Evaluate model
    evaluator = OfflineRLEvaluator(
        experiment_name="pendulum_example",
        log_to_wandb=True
    )
    
    eval_metrics = evaluator.compute_policy_metrics(
        actions=actions,
        rewards=rewards,
        next_states=next_states
    )
    print("Evaluation metrics:", eval_metrics)
    
    print("5. Deploying model...")
    # Deploy model
    deployer = ModelDeployer(
        deployment_dir=str(model_dir),
        config_dir=str(base_dir / "config")
    )
    
    model_path = str(model_dir / "final_model.pt")
    deployment_path = deployer.deploy_model(
        model_path=model_path,
        model_version="1.0.0",
        metadata={"eval_metrics": eval_metrics}
    )
    print(f"Model deployed to: {deployment_path}")
    
    print("6. Testing deployed model...")
    # Test deployed model
    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        state_proc = preprocessor.state_scaler.transform(state.reshape(1, -1))
        action = model.predict(state_proc)[0]
        action = preprocessor.action_scaler.inverse_transform(action.reshape(1, -1))[0]
        
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
    
    print(f"Test episode reward: {total_reward}")

if __name__ == "__main__":
    main()
