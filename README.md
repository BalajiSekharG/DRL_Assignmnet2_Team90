# Assignment 2: Robust Reinforcement Learning under Stochastic Action Failure

## Overview

This solution implements DQN and DDQN agents on the LunarLander-v3 environment with stochastic actuator failures, as specified in Assignment 2.

## Files

- `Q_learning_DQN_DDQN.py` - Main solution file containing all implementations
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Solution

Execute the main script:
```bash
python Q_learning_DQN_DDQN.py
```

## Solution Components

### Part (a): Modified Environment Implementation
- `StochasticLunarLander` class: Custom wrapper that implements:
  - 15% probability of thruster actions being replaced with "Do Nothing"
  - Modified reward: R = R_base - 0.3 × 1(a≠0) + B
  - Landing bonus B = 50 if safe landing conditions are met
- `verify_modified_environment()`: Function to verify implementation correctness

### Part (b): DQN Implementation
- `ReplayBuffer`: Experience replay buffer for storing transitions
- `QNetwork`: Neural network for Q-value approximation
- `DQNAgent`: DQN agent with:
  - Q-network and target network
  - Experience replay
  - ε-greedy exploration
  - Target network updates

### Part (c): DDQN Implementation
- `DDQNAgent`: Extends DQNAgent with Double Q-learning target computation
  - Uses online network for action selection
  - Uses target network for action evaluation

### Part (d): Performance Evaluation
- Training on both original and modified environments
- Generates four comparison plots:
  1. Episode Reward vs Training Episode
  2. Average Predicted Q-value vs Training Episode
  3. Successful Landing Rate vs Training Episode
  4. Average Thruster Activations per Episode

### Part (e): Discussion
The script automatically provides analysis for all discussion questions:
1. Q-value difference between DQN and DDQN under stochastic failures
2. Credit-assignment problem difficulty
3. Fuel penalty effect on landing strategy
4. Algorithm performance comparison
5. Limitations and improvements

## Output Files

After execution, the following files are generated:
- `plots/episode_rewards.png` - Episode reward comparison
- `plots/q_values.png` - Q-value comparison
- `plots/success_rate.png` - Landing success rate comparison
- `plots/thruster_activations.png` - Thruster usage comparison
- `results.pkl` - Training results (for further analysis)
- `verification_stats.pkl` - Environment verification statistics

## Hyperparameters

Default hyperparameters used (same for all agents):
- Learning rate: 1e-3
- Gamma (discount factor): 0.99
- Epsilon start: 1.0
- Epsilon end: 0.01
- Epsilon decay: 0.995
- Buffer size: 100,000
- Batch size: 64
- Target update frequency: 100
- Training episodes: 1000
- Max steps per episode: 1000

## Group Contribution

**IMPORTANT:** Before submitting, add your group member names and contribution percentages at the top of `Q_learning_DQN_DDQN.py` in the docstring section:

```python
"""
Group Contribution:
- Member Name 1: XX%
- Member Name 2: XX%
- Member Name 3: XX%
"""
```

## Submission Requirements

1. Execute the code in the virtual lab
2. Take screenshots with timestamps in the virtual lab
3. Create a PDF report including:
   - Group contribution declaration
   - Complete code with comments
   - All outputs and plots
   - Verification statistics
   - Discussion answers
4. Submit as a single PDF file named: `Team # - Q_learning_DQN_DDQN.pdf`

## Notes

- The code uses random seeds for reproducibility
- All functions include detailed comments as required
- The implementation follows the exact specifications in the assignment
- Training may take several hours depending on your hardware
