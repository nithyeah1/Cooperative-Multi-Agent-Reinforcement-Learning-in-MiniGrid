"""
Independent PPO (IPPO) for Multi-Agent MiniGrid with PettingZoo
Each agent trains independently using its own PPO policy
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import supersuit as ss
from pettingzoo.utils import parallel_to_aec
import torch

# Import your custom multi-agent minigrid environment
# You'll need to create this - I'll show an example structure below
# from custom_multiagent_minigrid import MultiAgentGoalReachingEnv


class IPPOTrainer:
    """
    Independent PPO trainer for multi-agent environments
    Each agent has its own PPO policy trained independently
    """
    
    def __init__(self, env_fn, num_agents, total_timesteps=1_000_000):
        self.env_fn = env_fn
        self.num_agents = num_agents
        self.total_timesteps = total_timesteps
        self.models = {}
        
    def make_env(self, agent_id):
        """Create environment wrapper for a specific agent"""
        def _init():
            # Create the parallel environment
            env = self.env_fn()
            
            # Convert to AEC (turn-based) for easier single-agent training
            env = parallel_to_aec(env)
            
            # Wrap with supersuit for single-agent compatibility
            # This extracts observations for the specific agent
            env = ss.agent_indicator_v0(env, type_only=False)
            
            # Flatten observations if needed
            env = ss.flatten_v0(env)
            
            # Convert to gymnasium for SB3
            from pettingzoo.utils.conversions import aec_to_parallel
            env = aec_to_parallel(env)
            
            # You might need additional wrappers here depending on your env
            return env
        
        return _init
    
    def train_all_agents(self):
        """Train all agents independently"""
        
        for agent_id in range(self.num_agents):
            print(f"\n{'='*50}")
            print(f"Training Agent {agent_id}")
            print(f"{'='*50}\n")
            
            # Create vectorized environment for this agent
            env = DummyVecEnv([self.make_env(agent_id)])
            env = VecMonitor(env)
            
            # Create PPO model for this agent
            model = PPO(
                "MlpPolicy",  # Use CnnPolicy if using image observations
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=f"./tensorboard/ippo_agent_{agent_id}/"
            )
            
            # Setup callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=f"./checkpoints/agent_{agent_id}/",
                name_prefix="ippo_model"
            )
            
            # Train the agent
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=checkpoint_callback,
                progress_bar=True
            )
            
            # Save final model
            model.save(f"ippo_agent_{agent_id}_final")
            self.models[agent_id] = model
            
            env.close()
    
    def evaluate(self, n_episodes=10):
        """Evaluate all agents together in the environment"""
        env = self.env_fn()
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(n_episodes):
            observations = env.reset()
            done = {agent: False for agent in env.agents}
            episode_reward = {agent: 0 for agent in env.agents}
            steps = 0
            
            while not all(done.values()):
                actions = {}
                
                # Get action from each agent's policy
                for agent in env.agents:
                    if not done[agent]:
                        obs = observations[agent]
                        action, _ = self.models[int(agent[-1])].predict(obs, deterministic=True)
                        actions[agent] = action
                
                # Step environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Update tracking
                for agent in env.agents:
                    episode_reward[agent] += rewards.get(agent, 0)
                    done[agent] = terminations.get(agent, False) or truncations.get(agent, False)
                
                steps += 1
            
            total_reward = sum(episode_reward.values())
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Check if goals were reached (environment-specific)
            if hasattr(env, 'goals_reached') and all(env.goals_reached.values()):
                success_count += 1
            
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {steps}")
        
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({n_episodes} episodes)")
        print(f"{'='*50}")
        print(f"Average Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Success Rate: {success_count / n_episodes * 100:.1f}%")
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_count / n_episodes
        }


# Example custom multi-agent MiniGrid environment
# You'll need to implement this based on your specific task

"""
from pettingzoo import ParallelEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall

class MultiAgentGoalReachingEnv(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array'], 'name': 'multi_agent_goal_v0'}
    
    def __init__(self, grid_size=10, num_agents=2):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        # Define observation and action spaces for each agent
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=0, high=255, 
                shape=(self.grid_size, self.grid_size, 3), 
                dtype=np.uint8
            ) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Discrete(7)  # MiniGrid actions
            for agent in self.possible_agents
        }
        
    def reset(self, seed=None, options=None):
        # Initialize grid, agent positions, and goals
        # Return observations dict
        pass
    
    def step(self, actions):
        # Execute actions for all agents
        # Return observations, rewards, terminations, truncations, infos
        pass
    
    def render(self):
        # Render the environment
        pass
"""


def main():
    """Main training loop"""
    
    # Define your environment creation function
    # This should return a new instance of your parallel environment
    def env_fn():
        # Replace with your actual environment
        # return MultiAgentGoalReachingEnv(grid_size=10, num_agents=2)
        
        # For now, using a placeholder - you need to implement your env
        raise NotImplementedError(
            "You need to implement your multi-agent MiniGrid environment. "
            "See the commented example class above for structure."
        )
    
    # Initialize trainer
    trainer = IPPOTrainer(
        env_fn=env_fn,
        num_agents=2,
        total_timesteps=1_000_000
    )
    
    # Train all agents
    print("Starting IPPO Training...")
    trainer.train_all_agents()
    
    # Evaluate trained agents
    print("\nEvaluating trained agents...")
    results = trainer.evaluate(n_episodes=20)
    
    print("\nTraining complete!")
    print(f"Models saved as 'ippo_agent_X_final.zip'")


if __name__ == "__main__":
    main()