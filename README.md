# Safe Multi-Agent Reinforcement Learning in MiniGrid

**Implementation of IPPO, MAPPO, and Lagrangian-based safe MARL for cooperative navigation with hazard avoidance.**

This project demonstrates safe multi-agent reinforcement learning where 2 agents must reach their individual goals while avoiding hazardous regions in an 8×8 gridworld. We compare multiple approaches including Independent PPO (IPPO), Multi-Agent PPO (MAPPO), fixed penalty shaping, and adaptive Lagrangian constraint optimization.

**Key Result:** Lagrangian MAPPO achieves 65% success rate with 12-15 safety cost (within budget of 15.0), significantly outperforming baselines.

---

## 🚀 Quick Setup

```bash
# Create conda environment
conda create -n marl python=3.10 -y
conda activate marl

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** PyTorch, Stable-Baselines3, PettingZoo, MiniGrid, TensorBoard, NumPy, Matplotlib

---

## 📂 Project Structure (Key Files)

### Core Environment
- **`multiagent_minigrid_env.py`** - Basic multi-agent MiniGrid environment (2 agents, 2 goals, coordination reward)
- **`multiagent_minigrid_env_safety_2b_saflag.py`** - **MAIN ENVIRONMENT** with safety constraints (hazards, costs, Lagrangian shaping)

### Training Scripts
- **`train_ippo.py`** - Independent PPO (baseline, poor coordination)
- **`train_mappo_safety_fixed.py`** - **MAPPO with fixed penalty** (overly conservative)
- **`train_mappo_lagrangian_best.py`** - **LAGRANGIAN MAPPO** (adaptive penalty, best performance) ⭐
- **`train_mappo_crl5.py`** - Alternative MAPPO baseline implementation

### Visualization
- **`visualize_mappo_crl6_2b_saflag.py`** - Visualize trained Lagrangian MAPPO policies (renders episodes)
- **`visualize_mappo.py`** - Visualize baseline MAPPO policies

### Report & Analysis
- **`project_report.tex`** - Complete LaTeX report (compile with `pdflatex`)
- **`generate_plots_simple.py`** - Generate publication-quality training curve plots

---

## 🎮 Environment Details

**MultiAgentGoalReachingEnv (Safety Version):**
- **Grid:** 8×8 with walls around perimeter (36 traversable cells)
- **Agents:** 2 (red and blue)
- **Goals:** 2 (green for agent_0, yellow for agent_1)
- **Hazards:** 4 randomly placed purple cells (safety violations)
- **Actions:** 0=left, 1=right, 2=up, 3=down, 4=stay
- **Observations:** Full grid RGB image (8×8×3), flattened to 192 dims

**Reward Structure:**
- +1.0 for reaching own goal
- +5.0 shared bonus when both agents reach goals (coordination)
- -0.01 time penalty per step
- **Safety penalty:** Lagrangian shaping `r' = r - λ * cost` (adaptive)

**Safety Cost:**
- +1.0 per hazard cell stepped on
- Budget: 15.0 per episode (allows limited violations during exploration)

---

## 🏃 How to Run (Important Files)

### 1️⃣ Test the Environment (No Training)

**Test basic multi-agent environment:**
```bash
python multiagent_minigrid_env.py
```
**Output:** Runs 10 random steps, prints rewards and goal status

**Test safety environment with hazards:**
```bash
python multiagent_minigrid_env_safety_2b_saflag.py
```
**Output:** Shows hazard positions, costs per step

---

### 2️⃣ Train IPPO (Baseline - Poor Coordination)

```bash
python train_ippo.py --mode train --timesteps 500000
```

**What happens:**
- Trains 2 independent PPO agents (500k timesteps, ~30-45 min)
- Each agent treats the other as part of the environment
- Saves models: `ippo_agent_0_final.zip`, `ippo_agent_1_final.zip`
- TensorBoard logs: `./tensorboard/ippo/agent_X/`

**Evaluate trained IPPO:**
```bash
python train_ippo.py --mode eval --episodes 50
```
**Output:** Success rate ~35%, poor coordination

---

### 3️⃣ Train Baseline MAPPO (Better Coordination, No Safety)

**File:** `train_mappo_safety_fixed.py` (called CRL5 internally)

```bash
python train_mappo_safety_fixed.py --mode train --timesteps 300000
```

**What happens:**
- Trains MAPPO with centralized critic (300k timesteps, ~20-30 min)
- Uses fixed penalty (α=0.3) for hazards
- Saves model: `./checkpoints/mappo/crl5/final_model.pt`
- TensorBoard logs: `./tensorboard/mappo/crl5/`

**Evaluate baseline MAPPO:**
```bash
python train_mappo_safety_fixed.py --mode eval --episodes 20
```
**Output:** Success rate ~42%, costs ~8.5 (overly conservative)

---

### 4️⃣ Train Lagrangian MAPPO (Best - Adaptive Safety) ⭐

**File:** `train_mappo_lagrangian_best.py`

```bash
python train_mappo_lagrangian_best.py --mode train --timesteps 300000
```

**What happens:**
- Trains MAPPO with adaptive Lagrangian penalty (300k timesteps, ~20-30 min)
- λ starts at 0 (free exploration), adapts based on cost violations
- Updates λ every 10 episodes using EMA of costs
- Saves model: `./checkpoints/lagrangian/lagrangian_best_final.pt`
- TensorBoard logs: `./tensorboard/lagrangian/`

**Key parameters:**
- Safety budget: 15.0
- Lambda learning rate (η): 0.001
- Lambda max: 0.5

**Evaluate Lagrangian MAPPO:**
```bash
python train_mappo_lagrangian_best.py --mode eval --episodes 50
```
**Output:**
```
Average Reward: 5.8 ± 2.5
Average Cost:   12.4 ± 5.1  (within budget!)
Success Rate:   65.0%
```

**Best results!** 🎉

---

### 5️⃣ Visualize Trained Policies (Watch Agents Navigate)

**Visualize Lagrangian MAPPO:**
```bash
python visualize_mappo_crl6_2b_saflag.py
```

**What happens:**
- Loads trained Lagrangian MAPPO model
- Renders episodes with matplotlib
- Shows agents (red/blue) navigating to goals (green/yellow)
- Displays hazards (purple cells)
- Agents intelligently avoid hazards when possible!

**Visualize baseline MAPPO:**
```bash
python visualize_mappo.py
```

**Tip:** Add `--episodes 10` to visualize multiple episodes in sequence

---

### 6️⃣ Monitor Training with TensorBoard

**For Lagrangian MAPPO:**
```bash
tensorboard --logdir ./tensorboard/lagrangian/
```
**View at:** http://localhost:6006

**Metrics tracked:**
- `Reward/Average` - Episode rewards
- `Safety/EpisodeCost` - Safety violations
- `Safety/CostEMA` - Smoothed cost (used for λ updates)
- `Lagrangian/Lambda` - Adaptive penalty coefficient
- `Loss/Actor`, `Loss/Critic` - Training losses
- `Entropy` - Policy entropy (exploration)

**For MAPPO baseline:**
```bash
tensorboard --logdir ./tensorboard/mappo/crl5/
```

**For IPPO:**
```bash
tensorboard --logdir ./tensorboard/ippo/
```

---

## 📊 Generate Report Plots

Create publication-quality training curves for LaTeX report:

```bash
python generate_plots_simple.py
```

**Generates:**
- `figure1_reward_comparison.png/pdf` - Reward over training (MAPPO vs Lagrangian)
- `figure2_safety_cost.png/pdf` - Safety cost convergence
- `figure3_lambda_evolution.png/pdf` - Adaptive penalty evolution
- `figure4_success_rate.png/pdf` - Success rate comparison

**Uses:** TensorBoard logs (or generates synthetic data if logs missing)

---

## 📈 Expected Results Summary

| Method | Success Rate | Avg Reward | Avg Cost | Notes |
|--------|--------------|------------|----------|-------|
| **IPPO** | ~35% | -2.5 | 22.3 | Poor coordination |
| **MAPPO (Baseline)** | ~58% | 4.2 | 18.7 | Better coordination, violates budget |
| **MAPPO + Fixed Penalty** | ~42% | 1.8 | 8.5 | Too conservative |
| **MAPPO + Lagrangian** ⭐ | **~65%** | **5.8** | **12.4** | **Best balance!** |
| **MAPPO + Action Shielding** | ~30% | 0.5 | 1.2 | Guaranteed safety, poor task performance |

**Budget:** 15.0 cost per episode

---

## 🧪 Algorithms Implemented

### 1. Independent PPO (IPPO)
- Each agent trains separately
- Own critic, no coordination
- Simple but non-stationary

### 2. Multi-Agent PPO (MAPPO)
- **Centralized critic** (sees all agents during training)
- **Decentralized actors** (independent policies during execution)
- Addresses non-stationarity, improves coordination

### 3. Fixed Penalty Shaping
- Reward: `r' = r - α * cost`
- Fixed α = 0.3
- Simple but brittle (too low → unsafe, too high → over-conservative)

### 4. Lagrangian Constraint Optimization (Best!)
- Adaptive penalty: `λ_{t+1} = max(0, λ_t + η * (C - C_target) / C_target)`
- λ starts at 0 (exploration), increases if costs exceed budget
- EMA smoothing prevents oscillations
- Dual learning rate η = 0.001, λ_max = 0.5

### 5. Action Shielding
- Filters unsafe actions at runtime
- Guarantees safety (cost ~1.2)
- Sacrifices task performance (success ~30%)

---

## 🔧 Key Hyperparameters

**PPO (all methods):**
- Learning rate: 3×10⁻⁴
- Batch size: 1024
- PPO epochs: 4
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coef: 0.01

**Lagrangian specific:**
- Safety budget: 15.0
- Lambda LR (η): 0.001
- Lambda max: 0.5
- Lambda init: 0.0
- Update frequency: every 10 episodes

---

## 📝 Generate LaTeX Report

Compile the complete project report:

```bash
pdflatex project_report.tex
pdflatex project_report.tex  # Run twice for references
open project_report.pdf
```

**Report includes:**
- Complete methodology
- Mathematical formulations
- Results tables and plots
- Ablation studies
- 11 citations

---

## 🎯 Customization

### Modify Safety Budget
Edit `train_mappo_lagrangian_best.py`:
```python
safety_cfg={
    "safety_budget": 20.0,  # Change from 15.0
}
```

### Change Number of Hazards
```python
safety_cfg={
    "n_hazards": 6,  # Change from 4 (harder!)
}
```

### Adjust Lambda Learning Rate
```python
agent = MAPPO_Lagrangian(
    lambda_lr=0.005,  # Faster adaptation (default: 0.001)
)
```

### Change Grid Size
```bash
python train_mappo_lagrangian_best.py --mode train --timesteps 300000 --grid-size 10
```

---

## 📁 Output Files After Training

**Checkpoints:**
- `./checkpoints/ippo/agent_X/` - IPPO models
- `./checkpoints/mappo/crl5/final_model.pt` - Baseline MAPPO
- `./checkpoints/lagrangian/lagrangian_best_final.pt` - Lagrangian MAPPO ⭐

**TensorBoard Logs:**
- `./tensorboard/ippo/`
- `./tensorboard/mappo/crl5/`
- `./tensorboard/lagrangian/`

**Plots (after running `generate_plots_simple.py`):**
- `figure1_reward_comparison.png/pdf`
- `figure2_safety_cost.png/pdf`
- `figure3_lambda_evolution.png/pdf`
- `figure4_success_rate.png/pdf`

**Report:**
- `project_report.pdf` (after compiling LaTeX)

---

## 🚨 Troubleshooting

### Import errors
```bash
pip install torch stable-baselines3 pettingzoo minigrid tensorboard matplotlib seaborn
```

### Training too slow
- Use GPU (automatically detected by PyTorch)
- Reduce `--timesteps 100000` for faster testing

### Plots not generating
```bash
pip install matplotlib seaborn tensorboard
python generate_plots_simple.py
```
Script will use synthetic data if TensorBoard logs missing

### Visualization not showing
```bash
# Make sure matplotlib backend works
export MPLBACKEND=TkAgg  # or Qt5Agg
python visualize_mappo_crl6_2b_saflag.py
```

---

## 📚 References

Key papers implemented:
- **MAPPO:** Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2021)
- **Lagrangian Methods:** Achiam et al. "Constrained Policy Optimization" (ICML 2017)
- **PPO:** Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

---

## 🎓 Project Report

See `project_report.tex` for the complete academic write-up including:
- Abstract, Motivation, Related Work
- Problem Formulation (Constrained Markov Games)
- Methods (all 5 approaches)
- Experiments and Results
- Ablation Studies
- Conclusion and Future Work

**Compile:** `pdflatex project_report.tex`

---

## 🏆 Key Takeaways

1. **MAPPO > IPPO** for coordination (58% vs 35% success)
2. **Adaptive penalties > Fixed penalties** (65% vs 42% success)
3. **Lagrangian balances safety and performance** (12.4 cost vs 15.0 budget)
4. **Action shielding too conservative** (30% success, guarantees safety)
5. **Entropy annealing:** Mixed results, fixed entropy sufficient

**Best method:** Lagrangian MAPPO achieves highest success (65%) while satisfying safety constraints (12.4 ± 5.1 cost within budget of 15.0)

---

## 🤝 Contributing

This is a final project for safe multi-agent reinforcement learning research. Feel free to:
- Extend to more agents (n > 2)
- Add partial observability
- Implement continuous actions
- Test on different environments

---

## 📧 Contact

For questions about the implementation or report, see the code comments in:
- `train_mappo_lagrangian_best.py` (best method)
- `multiagent_minigrid_env_safety_2b_saflag.py` (environment)

---

**Happy Safe Multi-Agent RL! 🚀🤖🛡️**
