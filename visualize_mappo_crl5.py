"""
Visualize trained MAPPO agents (CRL5)
Watch them navigate to their goals!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from multiagent_minigrid_env import MultiAgentGoalReachingEnv
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from train_mappo_crl5
from train_mappo_crl5 import MAPPO


def visualize_episode(checkpoint_path, grid_size=8, max_steps=100, save_gif=False):
    """Visualize one episode of MAPPO agents"""

    # Load checkpoint
    print("Loading trained MAPPO model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create environment
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=2,
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=True
    )

    obs_dim = np.prod(env.observation_space('agent_0').shape)
    act_dim = env.action_space('agent_0').n

    # Create MAPPO and load weights
    mappo = MAPPO(n_agents=2, obs_dim=obs_dim, act_dim=act_dim, device='cpu')
    for i, actor in enumerate(mappo.actors):
        actor.load_state_dict(checkpoint['actors'][i])
    mappo.critic.load_state_dict(checkpoint['critic'])

    print("✓ Loaded checkpoint")
    print(f"✓ Trained for {checkpoint['timesteps']:,} timesteps")

    # Run episode and collect trajectory
    obs_dict, _ = env.reset()
    trajectory = []
    done = False
    episode_reward = 0
    step = 0

    print("\nRunning episode...")

    while not done and step < max_steps:
        # Get current state
        agent_positions = {
            'agent_0': env.agents_dict['agent_0'].pos,
            'agent_1': env.agents_dict['agent_1'].pos
        }
        goal_positions = env.goals.copy()

        # Get actions from MAPPO
        observations = [obs_dict[f'agent_{i}'].flatten() for i in range(2)]
        actions, _, _ = mappo.select_actions(observations)

        # Store state
        trajectory.append({
            'step': step,
            'agent_positions': agent_positions.copy(),
            'goal_positions': goal_positions.copy(),
            'actions': actions.copy(),
            'goals_reached': env.goals_reached.copy()
        })

        # Step environment
        action_dict = {f'agent_{i}': actions[i] for i in range(2)}
        obs_dict, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

        episode_reward += sum(reward_dict.values())
        step += 1
        done = all(done_dict.values()) or all(trunc_dict.values())

    # Final state
    trajectory.append({
        'step': step,
        'agent_positions': {
            'agent_0': env.agents_dict['agent_0'].pos,
            'agent_1': env.agents_dict['agent_1'].pos
        },
        'goal_positions': env.goals.copy(),
        'actions': [None, None],
        'goals_reached': env.goals_reached.copy()
    })

    print(f"✓ Episode finished in {step} steps")
    print(f"✓ Total reward: {episode_reward:.2f}")
    print(f"✓ Agent 0 reached goal: {env.goals_reached['agent_0']}")
    print(f"✓ Agent 1 reached goal: {env.goals_reached['agent_1']}")

    # Visualize
    print("\nCreating visualization...")
    visualize_trajectory(trajectory, grid_size, save_gif=save_gif)

    env.close()


def visualize_trajectory(trajectory, grid_size, save_gif=False):
    """Create animated visualization of trajectory"""

    action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY']

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        state = trajectory[frame]

        # Draw grid
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

        # Draw walls (perimeter)
        for i in range(grid_size):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color='gray', alpha=0.5))
            ax.add_patch(plt.Rectangle((i, grid_size-1), 1, 1, color='gray', alpha=0.5))
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color='gray', alpha=0.5))
            ax.add_patch(plt.Rectangle((grid_size-1, i), 1, 1, color='gray', alpha=0.5))

        # Draw goals
        goal_0 = state['goal_positions']['agent_0']
        goal_1 = state['goal_positions']['agent_1']

        ax.add_patch(plt.Rectangle(goal_0, 1, 1, color='green', alpha=0.3, label='Goal 0'))
        ax.add_patch(plt.Rectangle(goal_1, 1, 1, color='yellow', alpha=0.3, label='Goal 1'))

        # Draw agents
        agent_0_pos = state['agent_positions']['agent_0']
        agent_1_pos = state['agent_positions']['agent_1']

        # Agent 0 (Red)
        agent_0_reached = state['goals_reached']['agent_0']
        agent_0_color = 'darkgreen' if agent_0_reached else 'red'
        ax.add_patch(plt.Circle(
            (agent_0_pos[0] + 0.5, agent_0_pos[1] + 0.5),
            0.35,
            color=agent_0_color,
            label='Agent 0',
            edgecolor='black',
            linewidth=2
        ))
        ax.text(agent_0_pos[0] + 0.5, agent_0_pos[1] + 0.5, '0',
                ha='center', va='center', fontsize=12, color='white', weight='bold')

        # Agent 1 (Blue)
        agent_1_reached = state['goals_reached']['agent_1']
        agent_1_color = 'darkgreen' if agent_1_reached else 'blue'
        ax.add_patch(plt.Circle(
            (agent_1_pos[0] + 0.5, agent_1_pos[1] + 0.5),
            0.35,
            color=agent_1_color,
            label='Agent 1',
            edgecolor='black',
            linewidth=2
        ))
        ax.text(agent_1_pos[0] + 0.5, agent_1_pos[1] + 0.5, '1',
                ha='center', va='center', fontsize=12, color='white', weight='bold')

        # Title with actions
        actions = state['actions']
        if actions[0] is not None:
            title = f"MAPPO - Step {state['step']}/{len(trajectory)-1}\n"
            title += f"Agent 0: {action_names[actions[0]]} | Agent 1: {action_names[actions[1]]}"
        else:
            title = f"MAPPO - Step {state['step']}/{len(trajectory)-1} - FINISHED"
            if all(state['goals_reached'].values()):
                title += "\n✅ SUCCESS - Both agents reached goals!"
            else:
                reached = sum(state['goals_reached'].values())
                title += f"\n⚠️ PARTIAL - {reached}/2 agents reached goals"

        ax.set_title(title, fontsize=14, weight='bold')

        # Legend
        handles = [
            mpatches.Patch(color='red', label='Agent 0 (RED → GREEN goal)'),
            mpatches.Patch(color='blue', label='Agent 1 (BLUE → YELLOW goal)'),
            mpatches.Patch(color='darkgreen', label='Agent reached goal'),
            mpatches.Patch(color='green', alpha=0.3, label='Goal 0'),
            mpatches.Patch(color='yellow', alpha=0.3, label='Goal 1'),
        ]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=500, repeat=True)

    if save_gif:
        print("Saving animation as mappo_crl5_trajectory.gif...")
        anim.save('mappo_crl5_trajectory.gif', writer='pillow', fps=2)
        print("✓ Saved!")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize MAPPO agents (CRL5)')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/mappo/crl5/final_model.pt',
                        help='Path to checkpoint')
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--save-gif', action='store_true', help='Save as GIF')

    args = parser.parse_args()

    print("=" * 60)
    print("MAPPO Visualization (CRL5)")
    print("=" * 60)

    visualize_episode(
        args.checkpoint,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        save_gif=args.save_gif
    )
