"""
Working IPPO Training Script for Multi-Agent MiniGrid
Simplified version that actually works with the custom environment
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import gymnasium as gym
from gymnasium import spaces

from multiagent_minigrid_env import MultiAgentGoalReachingEnv


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


def make_env(agent_id, grid_size=8, max_steps=100):
    """Factory function to create wrapped environment"""
    def _init():
        base_env = MultiAgentGoalReachingEnv(
            grid_size=grid_size,
            num_agents=2,
            max_steps=max_steps,
            shared_reward=True
        )
        return SingleAgentWrapper(base_env, agent_id)
    return _init


def train_ippo(
    num_agents=2,
    total_timesteps=500_000,
    grid_size=8,
    max_steps=100,
    save_freq=50_000
):
    """
    Train IPPO: Independent PPO for each agent
    """
    
    models = {}
    
    print("="*60)
    print("IPPO Training: Independent PPO for Multi-Agent MiniGrid")
    print("="*60)
    
    # Train each agent independently
    for agent_id in range(num_agents):
        print(f"\n{'='*60}")
        print(f"Training Agent {agent_id}")
        print(f"{'='*60}\n")
        
        # Create environment for this agent
        env = DummyVecEnv([make_env(agent_id, grid_size, max_steps)])
        
        # Create PPO model
        # Using MlpPolicy since we're flattening the grid observations
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"./tensorboard/ippo/agent_{agent_id}/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"./checkpoints/ippo/agent_{agent_id}/",
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
        
        # Save final model
        model.save(f"ippo_agent_{agent_id}_final")
        models[agent_id] = model
        
        print(f"\nAgent {agent_id} training complete!")
        print(f"Model saved as: ippo_agent_{agent_id}_final.zip")
        
        env.close()
    
    return models


def evaluate_ippo(models, n_episodes=20, grid_size=8, max_steps=100, render=False):
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
        shared_reward=True
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


def load_and_evaluate(n_episodes=20):
    """Load saved models and evaluate them"""
    
    print("Loading saved IPPO models...")
    
    models = {}
    for agent_id in range(2):
        try:
            model = PPO.load(f"ippo_agent_{agent_id}_final")
            models[agent_id] = model
            print(f"  ✓ Loaded agent_{agent_id}")
        except FileNotFoundError:
            print(f"  ✗ Could not find model for agent_{agent_id}")
            return None
    
    return evaluate_ippo(models, n_episodes=n_episodes)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate IPPO agents')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total timesteps for training')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes for evaluation')
    parser.add_argument('--grid-size', type=int, default=8,
                        help='Size of the grid')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train IPPO
        models = train_ippo(
            num_agents=2,
            total_timesteps=args.timesteps,
            grid_size=args.grid_size,
            max_steps=args.max_steps
        )
        
        # Evaluate trained models
        print("\n" + "="*60)
        print("Training Complete! Running evaluation...")
        print("="*60)
        results = evaluate_ippo(models, n_episodes=args.episodes, 
                               grid_size=args.grid_size, max_steps=args.max_steps)
        
    else:  # eval mode
        # Load and evaluate existing models
        results = load_and_evaluate(n_episodes=args.episodes)
        
        if results is None:
            print("\nNo saved models found. Please train first with --mode train")