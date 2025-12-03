"""
Visualize MAPPO-trained agents on vector-based MiniGrid
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from multiagent_minigrid_env_mappo import MultiAgentGoalReachingEnv
from train_mappo_crl4 import MAPPOActorCritic


def draw(env, step, ep, ax):
    ax.clear()
    g = env.grid_size
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.grid(True, color='gray', alpha=0.4)
    ax.set_title(f"Episode {ep+1}, Step {step}")
    colors = ['red', 'blue']
    goals = ['green', 'yellow']
    for i, ag in enumerate(env.agents):
        gx, gy = env.goals[ag]
        ax.add_patch(Circle((gx, gy), 0.3, color=goals[i], alpha=0.6))
        x, y = env.agents_dict[ag]["pos"]
        ax.add_patch(Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, color=colors[i]))
    plt.pause(0.25)
    plt.draw()


def visualize(model_path, episodes=5, grid_size=8):
    env = MultiAgentGoalReachingEnv(grid_size=grid_size)
    obs_dim, act_dim = env.observation_space('agent_0').shape[0], env.action_space('agent_0').n
    model = MAPPOActorCritic(obs_dim, act_dim, n_agents=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = {a: False for a in env.agents}
        total = 0
        step = 0
        while not all(done.values()):
            actions = {}
            for a in env.agents:
                o = torch.tensor(obs[a], dtype=torch.float32)
                act, _, _ = model.act(o)
                actions[a] = act.item()
            obs, r, done, trunc, _ = env.step(actions)
            total += sum(r.values())
            step += 1
            draw(env, step, ep, ax)
        rewards.append(total)
        print(f"Ep {ep+1}: Reward={total:.2f}")
    plt.ioff()
    plt.show()
    print(f"Avg Reward={np.mean(rewards):.2f}")


if __name__ == "__main__":
    visualize("./checkpoints/mappo/mappo_4/mappo_485.pt", episodes=10)
