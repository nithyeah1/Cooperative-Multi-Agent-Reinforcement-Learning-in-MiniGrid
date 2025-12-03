"""
Lagrangian MAPPO Visualization Script
-------------------------------------

Allows you to watch trained agents move in the safety grid,
with hazards, goals, rewards, and costs rendered clearly.

Usage:
    python visualize_lagrangian_policy.py --checkpoint <path> --episodes 3
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from multiagent_minigrid_env_safety_2b_saflag import MultiAgentGoalReachingEnv
from train_mappo_crl6_2b_saflag import MAPPO_Lagrangian   # REUSE YOUR MODEL CLASS


# =============================================================
#  UTILITY — RENDER GRID AS MATPLOTLIB IMAGE
# =============================================================

def render_grid(env, title=""):
    """Render the environment using its internal renderer."""
    img = env.render()  # returns np array (grid_size*cell, grid_size*cell, 3)

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(1.0)      # delay (can adjust)
    plt.close()


# =============================================================
#  VISUALIZATION FUNCTION
# =============================================================

def visualize_policy(checkpoint_path, n_episodes=5, grid_size=8, max_steps=100, device="cpu"):
    """
    Loads trained Lagrangian MAPPO and visualizes rollouts with rendering.
    """

    print("\n========== LAGRANGIAN MAPPO VISUALIZATION ==========")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    trained_lambda = ckpt.get("lambda", 0.0)
    print(f"Loaded λ (Lagrange multiplier) = {trained_lambda:.4f}")

    # Create environment (same safety settings as evaluation)
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=2,
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=False,
        safety_cfg={
            "enabled": True,
            "n_hazards": 6,
            "use_lagrangian": True,
            "lambda_coeff": trained_lambda,
            "safety_budget": 3.0,
        },
        render_mode="rgb_array"
    )

    obs_dim = int(np.prod(env.observation_space("agent_0").shape))
    act_dim = env.action_space("agent_0").n

    # Rebuild agent
    agent = MAPPO_Lagrangian(
        n_agents=2,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device
    )
    for i, actor in enumerate(agent.actors):
        actor.load_state_dict(ckpt["actors"][i])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.lambda_coeff = trained_lambda

    # =============================================================
    #   EPISODE VISUALIZATION LOOP
    # =============================================================

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        step = 0
        ep_reward = 0
        ep_cost = 0

        print(f"\n------ EPISODE {ep+1}/{n_episodes} ------")

        # Render initial state
        render_grid(env, title=f"Episode {ep+1} - Step {step}")

        while not done:
            # Prepare observations
            obs_list_np = [obs_dict[f"agent_{i}"].flatten() for i in range(2)]

            # Get actions from trained policy
            actions, _, _ = agent.select_actions(obs_list_np)

            # Environment step
            action_dict = {f"agent_{i}": actions[i] for i in range(2)}
            obs_dict, reward_dict, done_dict, trunc_dict, info_dict = env.step(action_dict)

            step += 1
            ep_reward += sum(reward_dict.values())
            ep_cost += sum(info["cost"] for info in info_dict.values())
            done = all(done_dict.values()) or all(trunc_dict.values())

            # Render current step
            title = (
                f"Episode {ep+1} | Step {step}\n"
                f"Reward={ep_reward:.2f}, Cost={ep_cost:.2f}"
            )
            render_grid(env, title=title)

        print(f"Episode finished: Reward={ep_reward:.2f}, Cost={ep_cost:.2f}")

    env.close()
    print("\n========== VISUALIZATION ENDED ==========\n")


# =============================================================
#  MAIN CLI
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Lagrangian MAPPO Policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    visualize_policy(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        device=args.device
    )
