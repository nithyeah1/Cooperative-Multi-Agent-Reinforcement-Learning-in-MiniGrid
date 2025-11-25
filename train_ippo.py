"""
Working IPPO Training Script for Multi-Agent MiniGrid
Simplified version that actually works with the custom environment
IMPROVED VERSION with custom CNN for small grids
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from multiagent_minigrid_env import MultiAgentGoalReachingEnv


class SmallGridCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for small grids (6x6)
    The default CnnPolicy expects 8x8 minimum, so we need a custom one
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # RGB = 3 channels

        # Simple CNN that works with 6x6 images
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            # Transpose to channels-first (NCHW) format for PyTorch
            sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


class SingleAgentWrapper(gym.Env):
    """
    Wrapper to convert multi-agent environment to single-agent for one specific agent
    This allows us to train each agent independently using SB3
    """
    
    def __init__(self, env, agent_id):
        super().__init__()
        self.env = env
        self.agent_id = agent_id
        self.agent_name = f"agent_{agent_id}"
        
        # Set observation and action spaces for this specific agent
        self.observation_space = env.observation_space(self.agent_name)
        self.action_space = env.action_space(self.agent_name)
        
        # Store other agents' policies (for evaluation)
        self.other_agent_models = {}
    
    def reset(self, seed=None, options=None):
        """Reset environment and return observation for our agent"""
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations[self.agent_name], infos[self.agent_name]
    
    def step(self, action):
        """
        Take a step in the environment
        For training, other agents take random actions
        For evaluation, other agents use their trained policies
        """
        # Collect actions from all agents
        actions = {}
        
        # Our agent's action
        actions[self.agent_name] = action
        
        # Other agents take random actions during training
        # (You can make them use trained policies during evaluation)
        for agent in self.env.agents:
            if agent != self.agent_name:
                if agent in self.other_agent_models:
                    # Use trained policy if available
                    obs = self.env._get_obs(agent)
                    other_action, _ = self.other_agent_models[agent].predict(obs, deterministic=False)
                    actions[agent] = other_action
                else:
                    # Random action
                    actions[agent] = self.env.action_space(agent).sample()
        
        # Step environment
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Return data for our specific agent
        obs = observations[self.agent_name]
        reward = rewards[self.agent_name]
        terminated = terminations[self.agent_name]
        truncated = truncations[self.agent_name]
        info = infos[self.agent_name]
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def make_env(agent_id, grid_size=8, max_steps=100, reward_shaping=True):
    """Factory function to create wrapped environment"""
    def _init():
        base_env = MultiAgentGoalReachingEnv(
            grid_size=grid_size,
            num_agents=2,
            max_steps=max_steps,
            shared_reward=True,
            reward_shaping=reward_shaping
        )
        return SingleAgentWrapper(base_env, agent_id)
    return _init


def train_ippo(
    num_agents=2,
    total_timesteps=1_000_000,
    grid_size=6,
    max_steps=150,
    save_freq=50_000,
    reward_shaping=True
):
    """
    Train IPPO: Independent PPO for each agent
    IMPROVED VERSION with CnnPolicy and better hyperparameters
    """

    models = {}

    print("="*60)
    print("IMPROVED IPPO Training")
    print("="*60)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Max Steps: {max_steps}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Reward Shaping: {reward_shaping}")
    print(f"Policy: CnnPolicy (optimized for image observations)")
    print("="*60)

    # Train each agent independently
    for agent_id in range(num_agents):
        print(f"\n{'='*60}")
        print(f"Training Agent {agent_id}")
        print(f"{'='*60}\n")

        # Create environment for this agent
        env = DummyVecEnv([make_env(agent_id, grid_size, max_steps, reward_shaping)])

        # Create PPO model with custom CNN for small grids
        # Using custom SmallGridCNN that works with 6x6 observations
        policy_kwargs = dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=128),
        )

        model = PPO(
            "CnnPolicy",  # Use CnnPolicy with our custom feature extractor
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,  # INCREASED: More exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"./tensorboard/ippo_baseline/agent_{agent_id}/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"./checkpoints/ippo_baseline/agent_{agent_id}/",
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=10
        )
        
        # Save final model as baseline
        model.save(f"ippo_baseline_agent_{agent_id}_final")
        models[agent_id] = model

        print(f"\nAgent {agent_id} training complete!")
        print(f"Model saved as: ippo_baseline_agent_{agent_id}_final.zip")
        
        env.close()
    
    return models


def evaluate_ippo(models, n_episodes=20, grid_size=6, max_steps=150, render=False, reward_shaping=True):
    """
    Evaluate trained IPPO agents together in the environment
    """

    print("\n" + "="*60)
    print("Evaluating IPPO Agents")
    print("="*60 + "\n")

    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=len(models),
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=reward_shaping
    )
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    individual_goals = {f"agent_{i}": 0 for i in range(len(models))}
    
    for episode in range(n_episodes):
        observations, _ = env.reset()
        terminated = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        
        episode_reward = {agent: 0 for agent in env.agents}
        steps = 0
        
        while not (all(terminated.values()) or all(truncated.values())):
            actions = {}
            
            # Get action from each agent's trained policy
            for agent in env.agents:
                agent_id = int(agent.split('_')[1])
                obs = observations[agent]
                action, _ = models[agent_id].predict(obs, deterministic=True)
                actions[agent] = action
            
            # Step environment
            observations, rewards, terminated, truncated, infos = env.step(actions)
            
            # Accumulate rewards
            for agent in env.agents:
                episode_reward[agent] += rewards[agent]
            
            steps += 1
            
            if render:
                env.render()
        
        # Track statistics
        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Count individual goal completions
        for agent in env.agents:
            if env.goals_reached[agent]:
                individual_goals[agent] += 1
        
        # Check if all goals were reached
        if all(env.goals_reached.values()):
            success_count += 1
        
        print(f"Episode {episode + 1:2d}: Total Reward = {total_reward:6.2f}, "
              f"Steps = {steps:3d}, Goals: {sum(env.goals_reached.values())}/{len(env.agents)}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average Total Reward:  {np.mean(episode_rewards):6.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):6.2f} ± {np.std(episode_lengths):6.2f}")
    print(f"Success Rate (all goals): {success_count / n_episodes * 100:5.1f}%")
    print(f"\nIndividual Goal Success Rates:")
    for agent, count in individual_goals.items():
        print(f"  {agent}: {count / n_episodes * 100:5.1f}%")
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes,
        'individual_success': {agent: count / n_episodes for agent, count in individual_goals.items()}
    }


def load_and_evaluate(n_episodes=20, improved=True):
    """Load saved models and evaluate them"""

    print("Loading saved IPPO models...")

    models = {}
    prefix = "ippo_improved" if improved else "ippo"

    for agent_id in range(2):
        try:
            model = PPO.load(f"{prefix}_agent_{agent_id}_final")
            models[agent_id] = model
            print(f"  ✓ Loaded {prefix}_agent_{agent_id}")
        except FileNotFoundError:
            print(f"  ✗ Could not find model for agent_{agent_id}")
            return None

    return evaluate_ippo(models, n_episodes=n_episodes)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate IPPO agents (IMPROVED VERSION)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total timesteps for training (default: 1M)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes for evaluation')
    parser.add_argument('--grid-size', type=int, default=6,
                        help='Size of the grid (default: 6x6 - easier than 8x8)')
    parser.add_argument('--max-steps', type=int, default=150,
                        help='Maximum steps per episode (default: 150)')
    parser.add_argument('--no-reward-shaping', action='store_true',
                        help='Disable reward shaping (distance-based rewards)')
    
    args = parser.parse_args()
    
    reward_shaping = not args.no_reward_shaping

    if args.mode == 'train':
        # Train IPPO
        models = train_ippo(
            num_agents=2,
            total_timesteps=args.timesteps,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            reward_shaping=reward_shaping
        )

        # Evaluate trained models
        print("\n" + "="*60)
        print("Training Complete! Running evaluation...")
        print("="*60)
        results = evaluate_ippo(
            models,
            n_episodes=args.episodes,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            reward_shaping=reward_shaping
        )

    else:  # eval mode
        # Load and evaluate existing models
        results = load_and_evaluate(n_episodes=args.episodes)

        if results is None:
            print("\nNo saved models found. Please train first with --mode train")