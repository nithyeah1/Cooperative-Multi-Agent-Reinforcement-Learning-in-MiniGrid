# Multi-Agent MiniGrid with IPPO

Independent PPO (IPPO) implementation for multi-agent goal-reaching tasks in MiniGrid environments.

## Setup

1. Install dependencies:
```bash
conda create -n marl python=3.10 -y
conda activate marl
pip install -r requirements.txt
```

## Project Structure

- `multiagent_minigrid_env.py` - Custom multi-agent MiniGrid environment
- `train_ippo.py` - Main training and evaluation script (SIMPLIFIED VERSION - USE THIS!)
- `ippo_multiagent_minigrid.py` - Original complex trainer class (reference only)
- `requirements.txt` - Python dependencies

## Quick Start

### Train IPPO Agents

Train two agents independently using PPO:

```bash
python train_ippo.py --mode train --timesteps 500000
```

Arguments:
- `--timesteps`: Total training timesteps (default: 500,000)
- `--grid-size`: Grid size (default: 8)
- `--max-steps`: Max steps per episode (default: 100)

### Evaluate Trained Agents

```bash
python train_ippo.py --mode eval --episodes 50
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir ./tensorboard/ippo/
```

## Environment Details

**MultiAgentGoalReachingEnv:**
- 2 agents (red and blue)
- Each agent has its own goal (green and yellow)
- Agents receive:
  - +1.0 reward for reaching their own goal
  - +5.0 shared bonus if both reach goals (coordination reward)
  - -0.01 time penalty per step
- Grid observation: (grid_size, grid_size, 3) RGB image
- Actions: 0=left, 1=right, 2=up, 3=down, 4=stay

## Algorithm: Independent PPO (IPPO)

Each agent trains independently with its own PPO policy:
- **Pros**: Simple, stable, parallelizable
- **Cons**: Ignores other agents during training (non-stationarity)
- **Use case**: Baseline for comparison with MAPPO

### Key Implementation Details

1. **SingleAgentWrapper**: Converts multi-agent env to single-agent view
2. During training: Other agents take random actions
3. During evaluation: All agents use their trained policies
4. Each agent has separate:
   - PPO policy network
   - Value network
   - Optimizer
   - Experience buffer

## Expected Results

With 500K timesteps training:
- Individual goal success rate: ~60-80%
- Joint success rate (both goals): ~40-60%
- Average episode length: 50-80 steps

## Next Steps: MAPPO

To show MAPPO > IPPO, you'll want to:

1. **Increase coordination requirements** in the environment:
   - Add doors requiring 2 agents to open
   - Make goals sequential (agent B can only reach goal after agent A does)
   - Add shared resources or obstacles

2. **Implement MAPPO**:
   - Centralized critic: Uses global state during training
   - Decentralized actors: Each agent has own policy (like IPPO)
   - Shared value function helps with credit assignment

3. **Compare metrics**:
   - Joint success rate (should be higher for MAPPO)
   - Sample efficiency (MAPPO should learn faster)
   - Coordination quality

## Hyperparameters

Default PPO hyperparameters in `train_ippo.py`:
- Learning rate: 3e-4
- Batch size: 64
- N steps: 2048
- N epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

## File Outputs

After training:
- `ippo_agent_0_final.zip` - Agent 0 trained policy
- `ippo_agent_1_final.zip` - Agent 1 trained policy
- `./checkpoints/ippo/agent_X/` - Training checkpoints
- `./tensorboard/ippo/agent_X/` - TensorBoard logs

## Customization

### Modify the Environment

Edit `multiagent_minigrid_env.py`:
- Change grid size
- Add obstacles/walls
- Modify reward structure
- Change observation space (e.g., partial observability)

### Add More Agents

In `train_ippo.py`, change:
```python
train_ippo(num_agents=3, ...)
```



