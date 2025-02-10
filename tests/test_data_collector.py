"""Tests for the data collector module."""
import pytest
import numpy as np
from src.data.collector import DataCollector

def test_data_collector_initialization():
    collector = DataCollector(storage_path="test_data")
    assert collector.storage_path == "test_data"
    assert len(collector.buffer) == 0

def test_collect_interaction():
    collector = DataCollector(storage_path="test_data")
    
    state = np.random.rand(4)
    action = np.random.rand(2)
    reward = 1.0
    next_state = np.random.rand(4)
    done = False
    
    collector.collect_interaction(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done
    )
    
    assert len(collector.buffer) == 1
    interaction = collector.buffer[0]
    assert 'timestamp' in interaction
    np.testing.assert_array_equal(interaction['state'], state)
    np.testing.assert_array_equal(interaction['action'], action)
    assert interaction['reward'] == reward
    np.testing.assert_array_equal(interaction['next_state'], next_state)
    assert interaction['done'] == done
