"""Tests for CQL model implementation."""
import numpy as np
import torch
import pytest
import wandb
from src.models.cql_model import CQLModel, QNetwork

@pytest.fixture(autouse=True)
def wandb_init():
    """Initialize wandb before each test."""
    try:
        wandb.init(mode="disabled")
        yield
    finally:
        wandb.finish()

def test_qnetwork_initialization():
    """Test Q-Network initialization and forward pass."""
    state_dim = 3
    action_dim = 1
    batch_size = 32
    
    # Initialize network
    qnet = QNetwork(state_dim, action_dim)
    
    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    q_values = qnet(states, actions)
    
    assert q_values.shape == (batch_size, 1)
    assert not torch.isnan(q_values).any()
    assert not torch.isinf(q_values).any()

def test_cql_model_initialization():
    """Test CQL model initialization."""
    state_dim = 3
    action_dim = 1
    
    model = CQLModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        learning_rate=0.0001,
        tau=0.005,
        discount=0.99,
        cql_alpha=1.0
    )
    
    assert isinstance(model.q1, QNetwork)
    assert isinstance(model.q2, QNetwork)
    assert isinstance(model.target_q1, QNetwork)
    assert isinstance(model.target_q2, QNetwork)

def test_cql_model_training_step():
    """Test CQL model training step."""
    state_dim = 3
    action_dim = 1
    batch_size = 32
    
    # Initialize model
    model = CQLModel(state_dim=state_dim, action_dim=action_dim)
    
    # Create dummy batch
    states = np.random.randn(batch_size, state_dim).astype(np.float32)
    actions = np.random.randn(batch_size, action_dim).astype(np.float32)
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_states = np.random.randn(batch_size, state_dim).astype(np.float32)
    dones = np.zeros(batch_size).astype(np.float32)
    
    # Run training step
    metrics = model.train_step(states, actions, rewards, next_states, dones)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'q1_loss' in metrics
    assert 'q2_loss' in metrics
    assert 'total_q1_loss' in metrics
    assert 'total_q2_loss' in metrics
    assert all(not np.isnan(v) for v in metrics.values())
    assert all(not np.isinf(v) for v in metrics.values())

def test_cql_model_prediction():
    """Test CQL model prediction."""
    state_dim = 3
    action_dim = 1
    batch_size = 32
    
    # Initialize model
    model = CQLModel(state_dim=state_dim, action_dim=action_dim)
    
    # Test prediction
    states = np.random.randn(batch_size, state_dim).astype(np.float32)
    actions = model.predict(states)
    
    assert actions.shape == (batch_size, action_dim)
    assert not np.isnan(actions).any()
    assert not np.isinf(actions).any()
    assert np.all(actions >= -1) and np.all(actions <= 1)  # Action bounds

def test_cql_model_evaluation():
    """Test CQL model evaluation."""
    state_dim = 3
    action_dim = 1
    batch_size = 32
    
    # Initialize model
    model = CQLModel(state_dim=state_dim, action_dim=action_dim)
    
    # Create dummy evaluation dataset
    eval_dataset = {
        'states': np.random.randn(batch_size, state_dim).astype(np.float32),
        'actions': np.random.randn(batch_size, action_dim).astype(np.float32),
        'rewards': np.random.randn(batch_size).astype(np.float32),
        'next_states': np.random.randn(batch_size, state_dim).astype(np.float32),
        'dones': np.zeros(batch_size).astype(np.float32)
    }
    
    # Run evaluation
    metrics = model.evaluate(eval_dataset)
    
    assert isinstance(metrics, dict)
    assert 'eval_reward' in metrics
    assert not np.isnan(metrics['eval_reward'])
    assert not np.isinf(metrics['eval_reward'])
