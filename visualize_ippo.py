"""
Visualize trained IPPO agents in the MiniGrid environment
Watch the agents move around and try to reach their goals
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from stable_baselines3 import PPO
from multiagent_minigrid_env import MultiAgentGoalReachingEnv


def visualize_episode(models, grid_size=8, max_steps=100, delay=0.3):
    """
    Run one episode and visualize it step by step
    
    Args:
        models: Dictionary of trained PPO models {agent_id: model}
        grid_size: Size of the grid
        max_steps: Maximum steps per episode
        delay: Delay between frames (seconds)
    """
    
    # Create environment
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=len(models),
        max_steps=max_steps,
        shared_reward=True
    )
    
    # Setup matplotlib
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Reset environment
    observations, _ = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    
    episode_reward = {agent: 0 for agent in env.agents}
    steps = 0
    
    print("\n" + "="*60)
    print("Starting Visualization")
    print("="*60)
    print(f"Agent 0 (RED) → Goal (GREEN)")
    print(f"Agent 1 (BLUE) → Goal (YELLOW)")
    print("="*60 + "\n")
    
    # Run episode
    while not (all(terminated.values()) or all(truncated.values())):
        # Clear plot
        ax.clear()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Flip y-axis to match grid coordinates
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Step {steps}/{max_steps} | Total Reward: {sum(episode_reward.values()):.2f}', 
                     fontsize=14, fontweight='bold')
        
        # Draw grid cells
        for x in range(grid_size):
            for y in range(grid_size):
                cell = env.grid.get(x, y)
                
                # Draw walls
                if cell is not None and cell.type == 'wall':
                    rect = Rectangle((x, y), 1, 1, facecolor='gray', edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
        
        # Draw goals
        for agent_name, goal_pos in env.goals.items():
            goal_color = 'lightgreen' if agent_name == 'agent_0' else 'yellow'
            edge_color = 'darkgreen' if agent_name == 'agent_0' else 'orange'
            goal_x, goal_y = goal_pos
            
            # Draw goal as a star/target
            rect = Rectangle((goal_x, goal_y), 1, 1, 
                           facecolor=goal_color, 
                           edgecolor=edge_color, 
                           linewidth=3,
                           alpha=0.7)
            ax.add_patch(rect)
            
            # Add label
            ax.text(goal_x + 0.5, goal_y + 0.5, '★', 
                   ha='center', va='center', fontsize=20, color=edge_color)
        
        # Draw agents
        for agent_name, agent_obj in env.agents_dict.items():
            agent_x, agent_y = agent_obj.pos
            agent_color = 'red' if agent_name == 'agent_0' else 'blue'
            
            # Draw agent as circle
            circle = plt.Circle((agent_x + 0.5, agent_y + 0.5), 0.35, 
                              color=agent_color, 
                              edgecolor='black', 
                              linewidth=2,
                              zorder=10)
            ax.add_patch(circle)
            
            # Add agent number
            agent_num = agent_name.split('_')[1]
            ax.text(agent_x + 0.5, agent_y + 0.5, agent_num, 
                   ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
            
            # Show if goal is reached
            if env.goals_reached[agent_name]:
                ax.text(agent_x + 0.5, agent_y - 0.3, '✓', 
                       ha='center', va='center', 
                       fontsize=16, color='green', fontweight='bold')
        
        # Add legend
        legend_text = []
        for agent_name in env.agents:
            agent_id = agent_name.split('_')[1]
            color_name = 'Red' if agent_id == '0' else 'Blue'
            goal_status = '✓' if env.goals_reached[agent_name] else '✗'
            reward = episode_reward[agent_name]
            legend_text.append(f"Agent {agent_id} ({color_name}): {goal_status} | R={reward:.2f}")
        
        ax.text(0.02, 0.98, '\n'.join(legend_text), 
               transform=ax.transAxes, 
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
        plt.pause(delay)
        
        # Get actions from trained policies
        actions = {}
        for agent in env.agents:
            agent_id = int(agent.split('_')[1])
            obs = observations[agent]
            action, _ = models[agent_id].predict(obs, deterministic=True)
            actions[agent] = action
        
        # Print actions
        action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY']
        action_str = ' | '.join([f"Agent {i}: {action_names[actions[f'agent_{i}']]}" 
                                  for i in range(len(models))])
        print(f"Step {steps:3d}: {action_str}")
        
        # Step environment
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        # Update rewards
        for agent in env.agents:
            episode_reward[agent] += rewards[agent]
        
        steps += 1
    
    # Final display
    print("\n" + "="*60)
    print("Episode Complete!")
    print("="*60)
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {sum(episode_reward.values()):.2f}")
    for agent in env.agents:
        status = "✓ REACHED" if env.goals_reached[agent] else "✗ NOT REACHED"
        print(f"{agent}: {status} | Reward: {episode_reward[agent]:.2f}")
    
    if all(env.goals_reached.values()):
        print("\n🎉 SUCCESS! Both agents reached their goals!")
    else:
        print("\n❌ Failed - Not all goals reached")
    
    print("="*60)
    
    plt.ioff()
    plt.show()
    
    env.close()


def visualize_multiple_episodes(models, n_episodes=5, grid_size=8, max_steps=100, delay=0.2):
    """Run and visualize multiple episodes"""
    
    print("\n" + "="*60)
    print(f"Visualizing {n_episodes} Episodes")
    print("="*60 + "\n")
    
    for ep in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        visualize_episode(models, grid_size, max_steps, delay)
        
        if ep < n_episodes - 1:
            input("\nPress Enter to continue to next episode...")


def main():
    """Load models and visualize"""
    
    print("Loading trained IPPO models...")
    
    models = {}
    for agent_id in range(2):
        try:
            model = PPO.load(f"ippo_agent_{agent_id}_final")
            models[agent_id] = model
            print(f"  ✓ Loaded ippo_agent_{agent_id}_final.zip")
        except FileNotFoundError:
            print(f"  ✗ Could not find ippo_agent_{agent_id}_final.zip")
            print("\nPlease train the models first:")
            print("  python train_ippo.py --mode train")
            return
    
    print("\nModels loaded successfully!\n")
    
    # Choose visualization mode
    print("Visualization Options:")
    print("  1. Single episode (detailed)")
    print("  2. Multiple episodes (5 episodes)")
    print("  3. Fast demo (short delay)")
    
    choice = input("\nEnter choice (1/2/3) [default=1]: ").strip()
    
    if choice == '2':
        visualize_multiple_episodes(models, n_episodes=5, delay=0.2)
    elif choice == '3':
        visualize_episode(models, delay=0.05)
    else:
        visualize_episode(models, delay=0.3)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize trained IPPO agents')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to visualize')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between steps (seconds)')
    parser.add_argument('--grid-size', type=int, default=8,
                        help='Size of the grid')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # Load models
    print("Loading trained IPPO models...")
    models = {}
    for agent_id in range(2):
        try:
            model = PPO.load(f"ippo_agent_{agent_id}_final")
            models[agent_id] = model
            print(f"  ✓ Loaded ippo_agent_{agent_id}_final.zip")
        except FileNotFoundError:
            print(f"  ✗ Could not find ippo_agent_{agent_id}_final.zip")
            print("\nPlease train the models first:")
            print("  python train_ippo.py --mode train")
            exit(1)
    
    if args.episodes == 1:
        visualize_episode(models, args.grid_size, args.max_steps, args.delay)
    else:
        visualize_multiple_episodes(models, args.episodes, args.grid_size, args.max_steps, args.delay)