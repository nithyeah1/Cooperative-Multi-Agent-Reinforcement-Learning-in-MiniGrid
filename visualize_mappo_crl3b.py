import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from train_mappo_crl3b import MAPPOActorCritic
from multiagent_minigrid_env import MultiAgentGoalReachingEnv


def visualize_mappo(model_path, grid_size=8, max_steps=100, delay=0.3):
    """
    Real-time visualization for NEW MAPPO version (with corrected GAE/PPO).
    """

    print("\n🔍 Loading environment + model...\n")

    # -----------------------------
    # Initialize environment
    # -----------------------------
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=2,
        max_steps=max_steps,
        shared_reward=True
    )

    obs_dim = int(np.prod(env.observation_space('agent_0').shape))
    act_dim = env.action_space('agent_0').n
    n_agents = len(env.agents)

    # -----------------------------
    # Initialize model
    # -----------------------------
    model = MAPPOActorCritic(obs_dim, act_dim, n_agents)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------
    # Visualization setup
    # -----------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    observations, _ = env.reset()
    observations = {a: torch.tensor(observations[a].flatten(), dtype=torch.float32)
                    for a in env.agents}

    terminated = {a: False for a in env.agents}
    truncated = {a: False for a in env.agents}
    episode_reward = {a: 0 for a in env.agents}

    steps = 0

    print("=== MAPPO Visualization ===")
    print("Agent 0: RED → Goal GREEN")
    print("Agent 1: BLUE → Goal YELLOW\n")

    # ========================================================================
    # LIVE EPISODE LOOP
    # ========================================================================
    while not (all(terminated.values()) or all(truncated.values())):

        ax.clear()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Step {steps}/{max_steps}")

        # ----------------------------------------------------------
        # Draw walls from env.grid
        # ----------------------------------------------------------
        for x in range(grid_size):
            for y in range(grid_size):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == "wall":
                    ax.add_patch(Rectangle((x, y), 1, 1,
                                           facecolor="gray",
                                           edgecolor="black"))

        # ----------------------------------------------------------
        # Draw each agent's goal
        # goal colors: agent0 -> green, agent1 -> yellow
        # ----------------------------------------------------------
        for agent_name, goal_pos in env.goals.items():
            gx, gy = goal_pos
            if agent_name == "agent_0":
                fill, edge = "lightgreen", "green"
            else:
                fill, edge = "yellow", "orange"

            ax.add_patch(
                Rectangle((gx, gy), 1, 1,
                          facecolor=fill,
                          edgecolor=edge,
                          linewidth=2)
            )

        # ----------------------------------------------------------
        # Choose action from MODEL
        # ----------------------------------------------------------
        actions = {}
        for agent in env.agents:
            obs_vec = observations[agent]
            with torch.no_grad():
                act, _, _ = model.act(obs_vec)
            actions[agent] = act.item()

        # ----------------------------------------------------------
        # Draw agents (red & blue)
        # ----------------------------------------------------------
        for agent in env.agents:
            x, y = env.agent_positions[agent]
            color = "red" if agent == "agent_0" else "blue"
            ax.add_patch(
                plt.Circle((x + 0.5, y + 0.5), 0.35,
                           color=color,
                           ec="black",
                           linewidth=2)
            )

        plt.pause(delay)

        # ----------------------------------------------------------
        # Environment step
        # ----------------------------------------------------------
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        for a in env.agents:
            episode_reward[a] += rewards[a]

        observations = {a: torch.tensor(next_obs[a].flatten(), dtype=torch.float32)
                        for a in env.agents}

        steps += 1

    # ========================================================================
    # END OF EPISODE
    # ========================================================================
    total_reward = sum(episode_reward.values())
    success = all(env.goals_reached.values())

    print("\n=== Episode Complete ===")
    print(f"Total Steps: {steps}")
    print(f"Total Reward (shared): {total_reward:.2f}")

    for a in env.agents:
        mark = "✓" if env.goals_reached[a] else "✗"
        print(f"{a}: {mark} | Reward = {episode_reward[a]:.2f}")

    if success:
        print("\n🎉 SUCCESS — both agents reached their goals!\n")
    else:
        print("\n❌ Agents did NOT reach all goals.\n")

    plt.ioff()
    plt.show()
    env.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.3)

    args = parser.parse_args()

    visualize_mappo(
        model_path=args.model,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        delay=args.delay
    )
