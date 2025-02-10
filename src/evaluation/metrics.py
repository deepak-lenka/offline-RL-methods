"""Evaluation metrics for offline RL models."""
from typing import Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
import wandb

class OfflineRLEvaluator:
    """Evaluator for offline RL models."""
    
    def __init__(self, experiment_name: str, log_to_wandb: bool = True):
        """Initialize the evaluator.
        
        Args:
            experiment_name: Name of the experiment
            log_to_wandb: Whether to log metrics to W&B
        """
        self.experiment_name = experiment_name
        self.log_to_wandb = log_to_wandb
        if log_to_wandb:
            wandb.init(project=experiment_name)
    
    def evaluate_policy(
        self,
        model,
        env_name: str,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate policy in environment.
        
        Args:
            model: The model to evaluate
            env_name: Name of the environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        env = gym.make(env_name)
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                action = model.predict(np.array([state]))[0]
                state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        metrics = {
            'average_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'average_episode_length': np.mean(episode_lengths)
        }
        
        if self.log_to_wandb:
            wandb.log(metrics)
            
        return metrics
    
    def evaluate_policy(
        self,
        model,
        env_name: str,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """Compute metrics for policy evaluation.
        
        Args:
            actions: Actions taken by the policy
            rewards: Rewards received
            next_states: Resulting states
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
        
        if self.log_to_wandb:
            wandb.log(metrics)
            
        return metrics
    
    def compare_policies(
        self,
        policy_a_metrics: Dict[str, float],
        policy_b_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare two policies for A/B testing.
        
        Args:
            policy_a_metrics: Metrics for policy A
            policy_b_metrics: Metrics for policy B
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {}
        for metric in policy_a_metrics:
            if metric in policy_b_metrics:
                diff = policy_a_metrics[metric] - policy_b_metrics[metric]
                rel_diff = diff / policy_b_metrics[metric]
                comparison[f'{metric}_absolute_diff'] = diff
                comparison[f'{metric}_relative_diff'] = rel_diff
        
        if self.log_to_wandb:
            wandb.log(comparison)
            
        return comparison
    
    def log_experiment(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> None:
        """Log experiment metrics and metadata.
        
        Args:
            metrics: Dictionary of metrics
            metadata: Additional metadata to log
        """
        if self.log_to_wandb:
            if metadata:
                wandb.log({**metrics, **metadata})
            else:
                wandb.log(metrics)
