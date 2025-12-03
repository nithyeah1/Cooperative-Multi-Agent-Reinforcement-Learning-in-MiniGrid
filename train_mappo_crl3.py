"""
Fixed and Stable MAPPO Trainer for Multi-Agent MiniGrid
- Correct centralized critic usage
- Proper PPO ratio & advantage computation
- Advantage normalization and reward shaping
- Gradient clipping and entropy tuning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiagent_minigrid_env import MultiAgentGoalReachingEnv


# =====================================
# 1️⃣ MLP Networks for Actor & Critic
# =====================================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# =====================================
# 2️⃣ MAPPO Actor-Critic Model
# =====================================
class MAPPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.actor = MLP(obs_dim, act_dim)
        self.critic = MLP(obs_dim * n_agents, 1)

    def act(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, obs, actions):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy


# =====================================
# 3️⃣ MAPPO Trainer
# =====================================
class MAPPOTrainer:
    def __init__(self, grid_size=8, n_agents=2, total_steps=500_000, rollout_len=2048):
        self.env = MultiAgentGoalReachingEnv(grid_size=grid_size, num_agents=n_agents, shared_reward=True, reward_shaping=True)
        obs_dim = np.prod(self.env.observation_space('agent_0').shape)
        act_dim = self.env.action_space('agent_0').n

        self.n_agents = n_agents
        self.model = MAPPOActorCritic(obs_dim, act_dim, n_agents)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.total_steps = total_steps
        self.rollout_len = rollout_len
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2

        os.makedirs("./checkpoints/mappo/mappo_3_b/", exist_ok=True)
        os.makedirs("./tensorboard/mappo/mappo_3_b/", exist_ok=True)
        self.writer = SummaryWriter("./tensorboard/mappo/mappo_3_b/")

    # -------------------------------
    # Collect rollouts (joint obs)
    # -------------------------------
    def collect_rollout(self):
        obs, _ = self.env.reset()
        obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in obs.items()}

        storage = []
        for _ in range(self.rollout_len):
            actions, log_probs = {}, {}
            individual_obs = []  # Store individual observations
            joint_obs = torch.cat([obs[f'agent_{i}'] for i in range(self.n_agents)])

            for i in range(self.n_agents):
                agent_obs = obs[f'agent_{i}']
                individual_obs.append(agent_obs)
                action, log_prob, _ = self.model.act(agent_obs)
                actions[f'agent_{i}'] = action.item()
                log_probs[f'agent_{i}'] = log_prob

            next_obs, reward, done, trunc, _ = self.env.step(actions)

            # ✅ Reward shaping (encourages moving toward goal)
            rewards = sum(list(reward.values()))  # team reward

            # Store: joint_obs for critic, individual_obs for actors, rewards, log_probs, actions
            storage.append((joint_obs, individual_obs, torch.tensor(rewards, dtype=torch.float32), log_probs, actions))

            obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in next_obs.items()}
            if all(done.values()) or all(trunc.values()):
                obs, _ = self.env.reset()
                obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in obs.items()}

        return storage

    # -------------------------------
    # Compute GAE (Generalized Advantage Estimation)
    # -------------------------------
    def compute_gae(self, rewards, values, next_value):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[step])
            next_value = values[step]
        return returns

    # -------------------------------
    # Training Loop
    # -------------------------------
    def train(self):
        total_updates = self.total_steps // self.rollout_len
        print("=" * 60)
        print("MAPPO Training: Cooperative Multi-Agent PPO")
        print("=" * 60)

        all_returns = []

        for update in range(total_updates):
            # === Collect Rollout ===
            rollout = self.collect_rollout()
            joint_obs_batch = torch.stack([r[0] for r in rollout])  # For critic
            rewards = torch.stack([r[2] for r in rollout])  # Rewards are now at index 2

            # === Compute Critic Values ===
            with torch.no_grad():
                values = self.model.critic(joint_obs_batch).squeeze()

            # === GAE and Returns ===
            next_value = torch.tensor(0.0)
            returns = torch.stack(self.compute_gae(rewards, values, next_value)).detach()
            advantages = (returns - values).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            avg_return = returns.mean().item()
            all_returns.append(avg_return)

            # === PPO Update ===
            for epoch in range(4):  # multiple PPO epochs
                new_values = self.model.critic(joint_obs_batch).squeeze()
                log_probs_all, entropy_all = [], []

                # Log probs for each agent using INDIVIDUAL observations
                for i in range(self.n_agents):
                    # Extract individual agent observations from storage
                    agent_obs_batch = torch.stack([r[1][i] for r in rollout])  # r[1] contains individual_obs
                    actions = torch.tensor([r[4][f'agent_{i}'] for r in rollout])  # r[4] contains actions
                    log_p, ent = self.model.evaluate_actions(agent_obs_batch, actions)
                    log_probs_all.append(log_p)
                    entropy_all.append(ent)

                log_probs = torch.stack(log_probs_all).mean(0)
                entropy = torch.stack(entropy_all).mean()

                old_log_probs = torch.stack([
                    torch.stack(list(r[3].values())).mean()  # r[3] contains old log_probs
                    for r in rollout
                ]).detach()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns - new_values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            # === Logging and Printing ===
            if (update + 1) % 1 == 0:
                # Track moving average for returns
                avg_100_return = np.mean(all_returns[-10:]) if len(all_returns) > 10 else avg_return

                print(f"[Update {update+1:04d}/{total_updates:04d}] "
                      f"Loss: {loss.item():7.4f} | "
                      f"Actor: {actor_loss.item():7.4f} | "
                      f"Critic: {critic_loss.item():7.4f} | "
                      f"Entropy: {entropy.item():6.3f} | "
                      f"Adv: {advantages.mean().item():7.3f} | "
                      f"AvgReturn: {avg_100_return:7.3f}")

                # Save checkpoint periodically
                if (update + 1) % 10 == 0:
                    torch.save(self.model.state_dict(), f"./checkpoints/mappo/fixed/mappo_update_{update+1}.pt")

                # TensorBoard logging
                self.writer.add_scalar("Loss/Total", loss.item(), update)
                self.writer.add_scalar("Loss/Actor", actor_loss.item(), update)
                self.writer.add_scalar("Loss/Critic", critic_loss.item(), update)
                self.writer.add_scalar("Entropy", entropy.item(), update)
                self.writer.add_scalar("Advantage/Mean", advantages.mean().item(), update)
                self.writer.add_scalar("Return/Avg", avg_return, update)

        self.writer.close()
        print("\nTraining Complete ✅")
        print(f"Final Avg Return: {np.mean(all_returns[-50:]):.2f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    def evaluate(self, n_episodes=20, grid_size=8, max_steps=100, render=False):
        print("\n" + "="*60)
        print("Evaluating MAPPO Agents")
        print("="*60 + "\n")

        env = MultiAgentGoalReachingEnv(
            grid_size=grid_size,
            num_agents=self.n_agents,
            max_steps=max_steps,
            shared_reward=True,
            reward_shaping=True
        )

        episode_rewards, episode_lengths = [], []
        success_count = 0
        individual_goals = {f"agent_{i}": 0 for i in range(self.n_agents)}

        for episode in range(n_episodes):
            observations, _ = env.reset()
            terminated = {agent: False for agent in env.agents}
            truncated = {agent: False for agent in env.agents}
            episode_reward = {agent: 0 for agent in env.agents}
            steps = 0

            while not (all(terminated.values()) or all(truncated.values())):
                actions = {}
                for agent in env.agents:
                    obs = torch.tensor(observations[agent].flatten(), dtype=torch.float32)
                    action, _, _ = self.model.act(obs)
                    actions[agent] = action.item()

                observations, rewards, terminated, truncated, infos = env.step(actions)
                for agent in env.agents:
                    episode_reward[agent] += rewards[agent]

                steps += 1
                if render:
                    env.render()

            total_reward = sum(episode_reward.values())
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            for agent in env.agents:
                if env.goals_reached[agent]:
                    individual_goals[agent] += 1

            if all(env.goals_reached.values()):
                success_count += 1

            print(f"Episode {episode + 1:2d}: Total Reward = {total_reward:6.2f}, "
                  f"Steps = {steps:3d}, Goals: {sum(env.goals_reached.values())}/{len(env.agents)}")

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


# =====================================
# 4️⃣ Run from CLI
# =====================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate MAPPO agents')
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

    trainer = MAPPOTrainer(grid_size=args.grid_size, total_steps=args.timesteps)

    if args.mode == 'train':
        trainer.train()
        print("\n" + "="*60)
        print("Training Complete! Running evaluation...")
        print("="*60)
        trainer.evaluate(n_episodes=args.episodes, grid_size=args.grid_size, max_steps=args.max_steps)
    else:
        print("\nLoading model and evaluating...")
        model_path = "./checkpoints/mappo/mappo_3_b/mappo_update_240.pt"
        trainer.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        trainer.evaluate(n_episodes=args.episodes, grid_size=args.grid_size, max_steps=args.max_steps, render=False)
