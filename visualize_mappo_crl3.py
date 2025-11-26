import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from train_mappo_crl3 import MAPPOActorCritic
from multiagent_minigrid_env import MultiAgentGoalReachingEnv


def visualize_mappo(model_path='./checkpoints/mappo/mappo_2/mappo_update_240.pt', grid_size=8, max_steps=100, delay=0.3):
    """Visualize trained MAPPO agents in MultiAgent MiniGrid"""

    env = MultiAgentGoalReachingEnv(grid_size=grid_size, num_agents=2, max_steps=max_steps, shared_reward=True)

    obs_dim = np.prod(env.observation_space('agent_0').shape)
    act_dim = env.action_space('agent_0').n

    model = MAPPOActorCritic(obs_dim, act_dim, n_agents=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    observations, _ = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    episode_reward = {agent: 0 for agent in env.agents}
    steps = 0

    print("\n=== MAPPO Visualization ===")
    print(f"Agent 0 (RED) → Goal (GREEN)")
    print(f"Agent 1 (BLUE) → Goal (YELLOW)")

    while not (all(terminated.values()) or all(truncated.values())):
        ax.clear()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Step {steps}/{max_steps}')

        # Draw walls
        for x in range(grid_size):
            for y in range(grid_size):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == 'wall':
                    rect = Rectangle((x, y), 1, 1, facecolor='gray', edgecolor='black')
                    ax.add_patch(rect)

        # Draw goals
        for agent_name, goal_pos in env.goals.items():
            goal_color = 'lightgreen' if agent_name == 'agent_0' else 'yellow'
            edge_color = 'green' if agent_name == 'agent_0' else 'orange'
            goal_x, goal_y = goal_pos
            rect = Rectangle((goal_x, goal_y), 1, 1, facecolor=goal_color, edgecolor=edge_color, linewidth=2)
            ax.add_patch(rect)

        # Choose actions
        actions = {}
        for agent in env.agents:
            obs_i = torch.tensor(observations[agent].flatten(), dtype=torch.float32)
            action, _, _ = model.act(obs_i)
            actions[agent] = action.item()

        # Draw agents
        for agent_name, agent_obj in env.agents_dict.items():
            x, y = agent_obj.pos
            color = 'red' if agent_name == 'agent_0' else 'blue'
            circle = plt.Circle((x + 0.5, y + 0.5), 0.35, color=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)

        plt.pause(delay)

        observations, rewards, terminated, truncated, infos = env.step(actions)
        for agent in env.agents:
            episode_reward[agent] += rewards[agent]

        steps += 1

    total_reward = sum(episode_reward.values())
    success = all(env.goals_reached.values())

    print("\n=== Episode Complete ===")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    for agent in env.agents:
        status = '✓ Reached' if env.goals_reached[agent] else '✗ Not Reached'
        print(f"{agent}: {status} | Reward: {episode_reward[agent]:.2f}")

    if success:
        print("\n🎉 SUCCESS! Both agents reached their goals!\n")
    else:
        print("\n❌ Failed - Not all goals reached\n")

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize trained MAPPO agents')
    parser.add_argument('--model', type=str, default='./checkpoints/mappo/mappo_3/mappo_update_240.pt', help='Path to trained MAPPO model')
    parser.add_argument('--grid-size', type=int, default=8, help='Grid size')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between frames')

    args = parser.parse_args()

    visualize_mappo(args.model, args.grid_size, args.max_steps, args.delay)