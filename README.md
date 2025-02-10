# Conservative Q-Learning (CQL) Implementation

This project implements Conservative Q-Learning (CQL), an offline reinforcement learning algorithm, with a focus on training stability and performance. The implementation includes a complete training pipeline, synthetic data generation, and comprehensive testing.

## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── cql_model.py    # CQL algorithm implementation
│   └── training/
│       └── trainer.py      # Training infrastructure
├── scripts/
│   ├── train.py           # Main training script
│   └── generate_data.py   # Synthetic data generation
├── config/
│   └── default_config.yaml # Model and training configuration
├── tests/
│   └── test_cql_model.py  # Unit tests for CQL implementation
└── data/
    └── pendulum/          # Synthetic pendulum environment data
```

## Features

- **CQL Implementation**:
  - Dual Q-networks with target networks
  - Conservative regularization to prevent overestimation
  - Batch normalization and gradient scaling for stability
  
- **Training Infrastructure**:
  - Configurable hyperparameters via YAML
  - Weights & Biases integration for experiment tracking
  - Automatic model checkpointing

- **Data Generation**:
  - Synthetic pendulum environment data
  - Configurable dataset size and parameters

- **Testing**:
  - Comprehensive unit tests
  - Model initialization tests
  - Training and prediction validation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Log in to Weights & Biases:
```bash
wandb login
```

## Usage

1. Generate synthetic training data:
```bash
python scripts/generate_data.py
```

2. Train the model:
```bash
python scripts/train.py --data-dir data/pendulum --output-dir models/pendulum
```

3. Run tests:
```bash
python -m pytest tests/test_cql_model.py -v
```

## Configuration

Adjust hyperparameters in `config/default_config.yaml`:
- Learning rate
- Network architecture
- CQL regularization strength
- Training parameters

## Results

The model achieves stable training with:
- Consistent Q-value learning
- Stable TD error
- Conservative value estimates

Training progress can be monitored through Weights & Biases dashboard.

### Example Evaluation Results

```bash
$ python scripts/evaluate.py --model-dir models/pendulum --num-episodes 10

Evaluating model...
Episode 0: Reward = -1479.16
Episode 1: Reward = -1780.81
Episode 2: Reward = -1452.59
Episode 3: Reward = -1914.46
Episode 4: Reward = -1543.96
Episode 5: Reward = -294.54
Episode 6: Reward = -888.37
Episode 7: Reward = -942.42
Episode 8: Reward = -1845.33
Episode 9: Reward = -1529.87

Evaluation Summary:
Average Reward: -1367.15 ± 483.43
Min Reward: -1914.46
Max Reward: -294.54
```

Note: These results are from a model trained for only 10 epochs. Performance can be improved by:
- Training for more epochs (100-200 recommended)
- Tuning hyperparameters
- Collecting more training data
- Experimenting with different network architectures
