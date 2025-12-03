"""
Quick evaluation script for Lagrangian MAPPO checkpoint
Fixes PyTorch 2.6 weights_only issue
"""

import torch
import numpy as np
from multiagent_minigrid_env_safety_2b_saflag import MultiAgentGoalReachingEnv

# Import MAPPO class from the training file
import sys
sys.path.append('/Users/nithyasri/Documents/Penn/fa_25/MiniGrid')
from train_mappo_lagrangian_best import MAPPO_Lagrangian

def evaluate(checkpoint_path, n_episodes=20):
    """Evaluate a trained Lagrangian MAPPO checkpoint"""

    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint with weights_only=False (safe - we trust our own checkpoint)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Loaded lambda: {ckpt.get('lambda', 0.0):.4f}")
    print(f"{'='*60}\n")

    # Create environment (match training settings)
    env = MultiAgentGoalReachingEnv(
        grid_size=8,
        num_agents=2,
        max_steps=100,
        shared_reward=True,
        reward_shaping=True,
        safety_cfg={
            "enabled": True,
            "n_hazards": 4,
            "use_lagrangian": True,
            "lambda_coeff": ckpt.get("lambda", 0.0),
            "safety_budget": 15.0,
        }
    )

    obs_dim = int(np.prod(env.observation_space("agent_0").shape))
    act_dim = env.action_space("agent_0").n

    # Create agent and load weights
    agent = MAPPO_Lagrangian(
        n_agents=2,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device='cpu'
    )

    for i, actor in enumerate(agent.actors):
        actor.load_state_dict(ckpt['actors'][i])
    agent.critic.load_state_dict(ckpt['critic'])

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    episode_costs = []
    success_count = 0
    individual_goals = {"agent_0": 0, "agent_1": 0}

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        reward_sum = 0.0
        cost_sum = 0.0
        steps = 0

        while not done:
            obs_list_np = [obs_dict[f"agent_{i}"].flatten() for i in range(2)]
            actions, _, _ = agent.select_actions(obs_list_np)

            action_dict = {f"agent_{i}": actions[i] for i in range(2)}
            obs_dict, reward_dict, done_dict, trunc_dict, info_dict = env.step(action_dict)

            reward_sum += sum(reward_dict.values())
            cost_sum += sum(info["cost"] for info in info_dict.values())
            steps += 1
            done = all(done_dict.values()) or all(trunc_dict.values())

        episode_rewards.append(reward_sum)
        episode_lengths.append(steps)
        episode_costs.append(cost_sum)

        # Check success
        final_info = list(info_dict.values())[0]
        if hasattr(env, 'goals_reached'):
            all_reached = all(env.goals_reached.values())
            if all_reached:
                success_count += 1
            for agent_name in ["agent_0", "agent_1"]:
                if env.goals_reached.get(agent_name, False):
                    individual_goals[agent_name] += 1

        print(f"Episode {ep+1:2d}: Reward = {reward_sum:7.2f}, Steps = {steps:3d}, Cost = {cost_sum:4.1f}, "
              f"Goals: {sum(1 for v in env.goals_reached.values() if v)}/2")

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL EVAL RESULTS")
    print(f"{'='*60}")
    print(f"Average Reward: {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):7.2f}")
    print(f"Average Length: {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):7.2f}")
    print(f"Average Cost:   {np.mean(episode_costs):7.2f} ± {np.std(episode_costs):7.2f}")
    print(f"Success Rate:  {success_count/n_episodes*100:5.1f}%")
    print(f"\nIndividual Goal Success:")
    for agent, count in individual_goals.items():
        print(f"  {agent}: {count/n_episodes*100:5.1f}%")
    print(f"{'='*60}\n")

    env.close()

    return {
        'avg_reward': np.mean(episode_rewards),
        'success_rate': success_count/n_episodes*100,
        'avg_cost': np.mean(episode_costs)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/lagrangian/model_500000.pt',
                        help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    results = evaluate(args.checkpoint, args.episodes)

    print(f"\n📊 Summary:")
    print(f"   Avg Reward: {results['avg_reward']:.2f}")
    print(f"   Success Rate: {results['success_rate']:.1f}%")
    print(f"   Avg Cost: {results['avg_cost']:.2f}")
