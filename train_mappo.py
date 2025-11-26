"""
MAPPO Training Script for Multi-Agent MiniGrid using RLlib
Multi-Agent PPO with centralized critic and parameter sharing
OUTPUT: Single shared policy saved as mappo_model_final.pkl
"""

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
import gymnasium as gym
from gymnasium import spaces
import argparse
import pickle
import os

from multiagent_minigrid_env import MultiAgentGoalReachingEnv


class RLlibMultiAgentWrapper(MultiAgentEnv):
    """
    Wrapper to make our MultiAgentGoalReachingEnv compatible with RLlib.

    - Underlying env returns uint8 images in [0, 255].
    - We expose flattened float32 obs in [0, 1] so RLlib uses an MLP
      (FullyConnectedNetwork) instead of VisionNet/CNN.
    """

    def __init__(self, config):
        super().__init__()
        self.grid_size = config.get("grid_size", 8)
        self._num_agents = config.get("num_agents", 2)
        self.max_steps = config.get("max_steps", 100)
        self.shared_reward = config.get("shared_reward", True)

        # Base multi-agent MiniGrid env
        self.env = MultiAgentGoalReachingEnv(
            grid_size=self.grid_size,
            num_agents=self._num_agents,
            max_steps=self.max_steps,
            shared_reward=self.shared_reward,
        )

        # Agent IDs
        self._agent_ids = list(self.env.possible_agents)
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self._agent_ids)  # updated at reset()

        # --- Observation and action spaces exposed to RLlib ---
        base_space = self.env.observation_space(self._agent_ids[0])
        obs_shape = base_space.shape  # e.g. (8, 8, 3)
        flat_dim = int(np.prod(obs_shape))

        # Expose a 1D float32 vector in [0, 1] -> forces MLP in RLlib
        float_obs_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(flat_dim,),
            dtype=np.float32,
        )

        self.observation_spaces = {
            agent_id: float_obs_space for agent_id in self._agent_ids
        }
        self.action_spaces = {
            agent_id: self.env.action_space(agent_id) for agent_id in self._agent_ids
        }

        # Single-agent views
        self.single_observation_space = float_obs_space
        self.single_action_space = self.action_spaces[self._agent_ids[0]]

        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    # ------------------------------------------------------------------
    # Helper: convert env observations (uint8 0–255) -> float32 0–1, flatten
    # ------------------------------------------------------------------
    def _convert_obs(self, observations):
        return {
            agent_id: (obs.astype(np.float32) / 255.0).ravel()
            for agent_id, obs in observations.items()
        }

    def reset(self, *, seed=None, options=None):
        """Reset environment - RLlib (Gymnasium-style) format."""
        observations, infos = self.env.reset(seed=seed, options=options)

        current_agent_ids = list(observations.keys())
        self.agents = current_agent_ids
        self.env.agents = current_agent_ids

        observations = self._convert_obs(observations)
        return observations, infos

    def step(self, action_dict):
        """
        Step environment with actions from all agents.
        Underlying env returns: obs, rewards, terminations, truncations, infos
        We adapt that to RLlib's multi-agent format.
        """
        observations, rewards, terminations, truncations, infos = self.env.step(
            action_dict
        )

        # Update live agents
        current_agent_ids = list(observations.keys())
        self.agents = current_agent_ids
        self.env.agents = current_agent_ids

        observations = self._convert_obs(observations)

        # RLlib expects dones/truncated for active agents + "__all__"
        dones = {}
        truncated = {}

        for agent in current_agent_ids:
            dones[agent] = bool(terminations.get(agent, False))
            truncated[agent] = bool(truncations.get(agent, False))

        # Episode done when all agents terminated or truncated
        all_terminated = all(terminations.get(a, False) for a in self._agent_ids)
        all_truncated = all(truncations.get(a, False) for a in self._agent_ids)

        dones["__all__"] = all_terminated or all_truncated
        truncated["__all__"] = all_truncated

        return observations, rewards, dones, truncated, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def train_mappo(
    num_agents=2,
    total_timesteps=500_000,
    grid_size=8,
    max_steps=100,
    save_freq=50_000,
    num_workers=2,
    num_gpus=0,
    run_name="mappo_1",
):
    """
    Train MAPPO: Multi-Agent PPO with parameter sharing and centralized critic.

    - Single shared policy network for all agents
    - Uses MLP (no CNN) via flattened observations
    """
    # Create directories with run name for version tracking
    checkpoint_dir = os.path.join("checkpoints", "mappo", run_name)
    tensorboard_dir = os.path.join("tensorboard", "mappo", run_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print("✓ Created directories:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Tensorboard: {tensorboard_dir}")
    print("=" * 60)
    print("MAPPO Training: Multi-Agent PPO with Centralized Critic")
    print("=" * 60)
    print(f"Run Name: {run_name}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Number of Agents: {num_agents}")
    print(f"Max Steps per Episode: {max_steps}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print("=" * 60 + "\n")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register environment
    def env_creator(config):
        return RLlibMultiAgentWrapper(config)

    tune.register_env("multiagent_minigrid", env_creator)

    # Environment configuration
    env_config = {
        "grid_size": grid_size,
        "num_agents": num_agents,
        "max_steps": max_steps,
        "shared_reward": True,
    }

    # Dummy env to get spaces
    dummy_env = RLlibMultiAgentWrapper(env_config)
    agent_ids = list(dummy_env._agent_ids)
    obs_space = dummy_env.observation_space   # flattened Box(...)
    act_space = dummy_env.action_space
    dummy_env.close()

    # --- MLP model config (no conv_filters, so no VisionNet) ---
    model_config = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    # PPO config using env_runners (new name) but OLD algo stack (api_stack off)
    config = (
        PPOConfig()
        .environment(env="multiagent_minigrid", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(
            num_env_runners=num_workers,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=100,
            batch_mode="truncate_episodes",
        )
        .training(
            train_batch_size=2048,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model=model_config,
        )
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    policy_class=None,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .reporting(
            min_sample_timesteps_per_iteration=1000,
            metrics_num_episodes_for_smoothing=10,
        )
        .debugging(log_level="WARN")
        .api_stack(  # turn OFF RLModule + env_runner_v2
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    # Set minibatch size explicitly
    config.sgd_minibatch_size = 256

    # Build the algorithm
    algo = config.build()
    print(f"Training MAPPO with {num_agents} agents (shared policy, MLP)")
    print(f"Workers: {num_workers}, GPUs: {num_gpus}")
    print(f"Checkpoints: {checkpoint_dir}/")
    print(f"Save frequency: every {save_freq:,} timesteps")
    print(f"Tensorboard: {tensorboard_dir}/")
    print("\nStarting training...\n")

    # Training loop with checkpoint saving
    total_timesteps_completed = 0
    next_save_timesteps = save_freq

    while total_timesteps_completed < total_timesteps:
        result = algo.train()
        ts_this_iter = result.get("timesteps_this_iter", 0)
        total_timesteps_completed += ts_this_iter

        episodes_this_iter = result.get("episodes_this_iter", 0)
        episode_reward_mean = result.get("episode_reward_mean", 0.0)
        episode_len_mean = result.get("episode_len_mean", 0.0)

        info = result.get("info", {})
        learner_stats = info.get("learner", {}).get("default_policy", {})
        if "learner_stats" in learner_stats:
            learner_stats = learner_stats["learner_stats"]

        policy_loss = learner_stats.get("policy_loss", 0.0)
        vf_loss = learner_stats.get("vf_loss", 0.0)

        print(
            f"Timesteps {total_timesteps_completed:8,} | "
            f"Episodes: {episodes_this_iter:4d} | "
            f"Reward: {episode_reward_mean:7.2f} | "
            f"Len: {episode_len_mean:6.1f} | "
            f"Policy Loss: {policy_loss:8.4f} | "
            f"VF Loss: {vf_loss:8.4f}"
        )

        # Save checkpoint
        if total_timesteps_completed >= next_save_timesteps:
            checkpoint_name = f"model_{total_timesteps_completed}_steps"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            algo.save(checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
            next_save_timesteps += save_freq

        if total_timesteps_completed >= total_timesteps:
            break

    # Final checkpoint
    final_checkpoint_name = f"model_{total_timesteps_completed}_steps"
    final_checkpoint = os.path.join(checkpoint_dir, final_checkpoint_name)
    algo.save(final_checkpoint)
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total timesteps: {total_timesteps_completed:,}")
    print(f"Final checkpoint: {final_checkpoint}")
    print("=" * 60 + "\n")

    return algo, final_checkpoint


def save_mappo_model(algo, checkpoint_path, run_name="mappo_1"):
    """
    Save MAPPO model as single pickle file.
    Saves the checkpoint path and policy info.
    """
    save_dir = os.path.join("checkpoints", "mappo", run_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "mappo_model_final")

    model_data = {
        "checkpoint_path": checkpoint_path,
        "policy_id": "shared_policy",
        "algorithm_type": "MAPPO",
        "framework": "RLlib",
        "run_name": run_name,
    }

    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("\n" + "=" * 60)
    print("Model Saved:")
    print(f"  {save_path}.pkl")
    print("\nThis contains the shared policy for all agents")
    print("=" * 60)


def load_mappo_model(run_name="mappo_1"):
    """
    Load MAPPO model from pickle file.
    Returns the RLlib algorithm instance.
    """
    load_path = os.path.join("checkpoints", "mappo", run_name, "mappo_model_final")
    print(f"Loading MAPPO model from {load_path}.pkl...")

    with open(f"{load_path}.pkl", "rb") as f:
        model_data = pickle.load(f)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    def env_creator(config):
        return RLlibMultiAgentWrapper(config)

    tune.register_env("multiagent_minigrid", env_creator)

    from ray.rllib.algorithms.ppo import PPO

    algo = PPO.from_checkpoint(model_data["checkpoint_path"])

    print("✓ Loaded MAPPO model successfully")
    print(f"  Run: {model_data.get('run_name', 'N/A')}")
    print(f"  Policy: {model_data['policy_id']}")
    print(f"  Checkpoint: {model_data['checkpoint_path']}\n")

    return algo


def evaluate_mappo(algo, n_episodes=20, grid_size=8, max_steps=100, render=False):
    """
    Evaluate trained MAPPO agents.
    """
    print("\n" + "=" * 60)
    print("Evaluating MAPPO Agents")
    print("=" * 60 + "\n")

    env_config = {
        "grid_size": grid_size,
        "num_agents": 2,
        "max_steps": max_steps,
        "shared_reward": True,
    }

    env = RLlibMultiAgentWrapper(env_config)

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    individual_goals = {agent: 0 for agent in env._agent_ids}

    for episode in range(n_episodes):
        observations, _ = env.reset()
        done = {"__all__": False}
        episode_reward = {agent: 0.0 for agent in env._agent_ids}
        steps = 0

        while not done["__all__"]:
            actions = {}
            for agent_id in env._agent_ids:
                action = algo.compute_single_action(
                    observations[agent_id],
                    policy_id="shared_policy",
                    explore=False,
                )
                actions[agent_id] = action

            observations, rewards, done, truncated, infos = env.step(actions)

            for agent_id in env._agent_ids:
                episode_reward[agent_id] += rewards.get(agent_id, 0.0)

            steps += 1

            if render:
                env.render()

        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Goal statistics from underlying env
        for agent in env._agent_ids:
            if env.env.goals_reached[agent]:
                individual_goals[agent] += 1

        if all(env.env.goals_reached.values()):
            success_count += 1

        print(
            f"Episode {episode + 1:2d}: Total Reward = {total_reward:6.2f}, "
            f"Steps = {steps:3d}, Goals: {sum(env.env.goals_reached.values())}/{len(env._agent_ids)}"
        )

    print("\n" + "=" * 60)
    print(f"Evaluation Results ({n_episodes} episodes)")
    print("=" * 60)
    print(
        f"Average Total Reward:  {np.mean(episode_rewards):6.2f} ± {np.std(episode_rewards):6.2f}"
    )
    print(
        f"Average Episode Length: {np.mean(episode_lengths):6.2f} ± {np.std(episode_lengths):6.2f}"
    )
    print(f"Success Rate (all goals): {success_count / n_episodes * 100:5.1f}%")
    print("\nIndividual Goal Success Rates:")
    for agent, count in individual_goals.items():
        print(f"  {agent}: {count / n_episodes * 100:5.1f}%")

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": success_count / n_episodes,
        "individual_success": {
            agent: count / n_episodes for agent, count in individual_goals.items()
        },
    }


def load_and_evaluate(run_name="mappo_1", n_episodes=20):
    algo = load_mappo_model(run_name)
    results = evaluate_mappo(algo, n_episodes=n_episodes)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate MAPPO agents")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--grid-size", type=int, default=8, help="Size of the grid"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="mappo_1",
        help="Run name for version tracking (e.g., mappo_1, mappo_2)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N timesteps",
    )

    args = parser.parse_args()

    if args.mode == "train":
        algo, checkpoint_path = train_mappo(
            num_agents=2,
            total_timesteps=args.timesteps,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            save_freq=args.save_freq,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            run_name=args.run_name,
        )

        save_mappo_model(algo, checkpoint_path, args.run_name)

        print("\n" + "=" * 60)
        print("Training Complete! Running evaluation...")
        print("=" * 60)
        _ = evaluate_mappo(
            algo,
            n_episodes=args.episodes,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
        )

        algo.stop()
        ray.shutdown()
    else:
        _ = load_and_evaluate(args.run_name, n_episodes=args.episodes)
        ray.shutdown()
