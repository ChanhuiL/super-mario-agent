# Super Mario Agent

A state-of-the-art Deep Reinforcement Learning agent for playing Super Mario Bros using Rainbow DQN enhancements.

## Features

### Core Architecture
- **Multi-branch DQN**: Separate streams for current frame, screenshot history, and action history
- **Spatial Transformer Network (STN)**: Learns to focus on relevant screen regions
- **Dueling Architecture**: Separates state value and advantage estimation
- **Distributional RL (C51)**: Models full value distribution instead of just mean

### Rainbow DQN Enhancements
- **Double DQN**: Reduces overestimation bias
- **Prioritized Experience Replay**: Samples more informative transitions
- **N-step Returns**: Faster reward propagation
- **Noisy Networks**: Parameterized exploration without epsilon-greedy

### Training Features
- Experience replay buffer (1M capacity)
- Target network updates
- Gradient clipping for stability
- Comprehensive logging and checkpointing

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Super Mario Bros environment:
```bash
pip install gym-super-mario-bros
```

## Usage

### Training

Basic training with default settings:
```bash
python train.py
```

Training with all Rainbow enhancements:
```bash
python train.py --use-noisy --use-stn --use-prioritized --n-step 3
```

Custom training configuration:
```bash
python train.py \
    --env-name SuperMarioBros-1-1-v0 \
    --max-steps 5000000 \
    --batch-size 64 \
    --lr 1e-4 \
    --use-noisy \
    --use-stn \
    --use-prioritized \
    --target-update 10000
```

### Evaluation

Evaluate a trained model:
```bash
python train.py --eval --load-checkpoint ./logs/mario_*/best_model.pt --eval-episodes 20
```

### Command Line Arguments

#### Environment
- `--env-name`: Gym environment name (default: `SuperMarioBros-1-1-v0`)
- `--skip-frames`: Number of frames to skip (default: 4)
- `--stack-frames`: Number of frames to stack (default: 4)

#### Training
- `--max-steps`: Maximum training steps (default: 10M)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 2.5e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--min-buffer-size`: Minimum buffer size before training (default: 50K)

#### Exploration
- `--epsilon-start`: Initial epsilon (default: 1.0)
- `--epsilon-end`: Final epsilon (default: 0.1)
- `--epsilon-decay`: Epsilon decay steps (default: 1M)
- `--use-noisy`: Use noisy networks (disables epsilon-greedy)

#### Network Architecture
- `--use-stn`: Enable Spatial Transformer Network
- `--num-atoms`: Number of atoms for distributional RL (default: 51)
- `--v-min`: Minimum value for distributional RL (default: -10)
- `--v-max`: Maximum value for distributional RL (default: 10)
- `--action-history-length`: Length of action history (default: 4)

#### Replay Buffer
- `--use-prioritized`: Use prioritized experience replay
- `--n-step`: N-step returns (default: 3)

#### Logging
- `--log-dir`: Directory for logs (default: `./logs`)
- `--print-interval`: Print interval in episodes (default: 10)
- `--save-interval`: Save checkpoint interval in episodes (default: 100)

## Project Structure

```
.
├── environment.py      # Environment wrappers and preprocessing
├── model.py           # Neural network architectures (RainbowDQN, STN, etc.)
├── replay_buffer.py   # Experience replay buffers (standard, prioritized, n-step)
├── agent.py           # DQN agent with training logic
├── train.py           # Training and evaluation scripts
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Architecture Details

### Multi-Branch Network

The agent uses three separate branches:
1. **Current Frame Branch**: CNN processes the latest stacked frames
2. **Screenshot History Branch**: CNN processes previous screenshots (currently uses same state, can be extended)
3. **Action History Branch**: LSTM processes the last k actions

All branches are concatenated before the dueling heads.

### Spatial Transformer Network

The STN learns affine transformations to focus on important screen regions:
- Localization network: CNN that predicts transformation parameters
- Grid sampling: Applies learned transformation to input

### Distributional RL (C51)

Instead of predicting Q-values directly, the network predicts a probability distribution over 51 atoms (value bins). The final Q-value is computed as the expectation over this distribution.

### Noisy Networks

Noisy linear layers replace epsilon-greedy exploration by adding learnable noise to network parameters, allowing the network to learn its own exploration strategy.

## Training Tips

1. **Start Simple**: Begin with basic DQN (no flags) to ensure everything works
2. **Add Features Gradually**: Enable `--use-stn`, then `--use-prioritized`, then `--use-noisy`
3. **Monitor Metrics**: Check logs in `./logs/` directory
4. **Save Checkpoints**: Models are saved automatically at best performance and periodically
5. **GPU Recommended**: Training is much faster on GPU

## Expected Performance

- **World 1-1**: Should learn to complete the level after ~1-5M frames
- **Sample Efficiency**: Rainbow enhancements reduce training time by 2-5x compared to vanilla DQN
- **Convergence**: Typically converges within 5-10M frames on World 1-1

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For `gym-super-mario-bros`, you may need: `pip install gym-super-mario-bros[accept-rom-license]`

### CUDA Out of Memory
- Reduce `--batch-size` (e.g., 16 or 8)
- Disable `--use-stn` to reduce memory usage

### Slow Training
- Use GPU: `--cpu` flag forces CPU (remove it for GPU)
- Reduce `--max-steps` for testing
- Increase `--skip-frames` to speed up environment

## References

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- [Distributional RL (C51)](https://arxiv.org/abs/1707.06887)
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

## License

This project is for educational purposes. Super Mario Bros is a trademark of Nintendo.

