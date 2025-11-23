# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Safe Reinforcement Learning project implementing Independent PPO (IPPO) for multi-agent goal-reaching tasks in custom MiniGrid environments. The project focuses on multi-agent coordination where agents learn to reach their individual goals while potentially receiving shared rewards for coordination.

## Environment Setup

```bash
conda create -n marl python=3.10 -y
conda activate marl
pip install -r requirements.txt
```

## Common Commands

### Training
```bash
# Train IPPO agents for 500K timesteps
python train_ippo.py --mode train --timesteps 500000

# Train with custom parameters
python train_ippo.py --mode train --timesteps 1000000 --grid-size 10 --max-steps 150
```

### Evaluation
```bash
# Evaluate trained agents
python train_ippo.py --mode eval --episodes 50

# Quick evaluation with fewer episodes
python train_ippo.py --mode eval --episodes 10
```

### Monitoring
```bash
# View training progress with TensorBoard
tensorboard --logdir ./tensorboard/ippo/
```

### Environment Testing
```bash
# Test the multi-agent environment directly
python multiagent_minigrid_env.py
```

## Architecture Overview

### Core Components

1. **MultiAgentGoalReachingEnv** (`multiagent_minigrid_env.py`):
   - Custom PettingZoo parallel environment
   - 2 agents (red/blue) with individual goals (green/yellow)
   - Grid-based observations with RGB encoding
   - Action space: left, right, up, down, stay
   - Reward structure: +1.0 for goal, +5.0 shared bonus, -0.01 time penalty

2. **SingleAgentWrapper** (`train_ippo.py`):
   - Converts multi-agent environment to single-agent view for SB3
   - Handles other agents with random actions during training
   - Enables independent policy training for each agent

3. **IPPO Training Pipeline** (`train_ippo.py`):
   - Independent PPO for each agent using Stable-Baselines3
   - Separate neural networks, optimizers, and experience buffers
   - Automatic checkpointing and TensorBoard logging

### Key Design Patterns

- **Independent Learning**: Each agent trains separately without coordination during training
- **Evaluation Coordination**: All agents use trained policies during evaluation
- **Environment Abstraction**: SingleAgentWrapper abstracts multi-agent complexity from SB3
- **Modular Architecture**: Separate files for environment, training, and legacy implementations

### File Structure

- `train_ippo.py` - Main training script (RECOMMENDED - simplified version)
- `multiagent_minigrid_env.py` - Custom multi-agent MiniGrid environment
- `ippo_multiagent_minigrid.py` - Original complex trainer (reference only)
- `MiniGrid + SB3 + HJA.ipynb` - Research notebook with Hamilton-Jacobi analysis

### Default Hyperparameters

- Learning rate: 3e-4
- Batch size: 64
- N steps: 2048
- N epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

## Output Files

After training:
- `ippo_agent_0_final.zip` / `ippo_agent_1_final.zip` - Final trained policies
- `./checkpoints/ippo/agent_X/` - Training checkpoints
- `./tensorboard/ippo/agent_X/` - TensorBoard logs

## Development Notes

- The simplified `train_ippo.py` is the recommended entry point
- GPU training is automatically used when available via PyTorch CUDA detection
- The environment uses full observability (can be modified for partial observability)
- Random seeds can be set for reproducible experiments
- The project is set up for extending to MAPPO (centralized critic) implementations