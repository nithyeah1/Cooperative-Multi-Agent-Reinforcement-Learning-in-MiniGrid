"""
MAPPO Implementation - Clean & Working
Built from scratch with proper centralized critic
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiagent_minigrid_env_safety_2b import MultiAgentGoalReachingEnv


# =====================================
# Actor Network (Decentralized)
# =====================================
class Actor(nn.Module):
    """Individual policy for each agent"""
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs):
        """Sample action from policy"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, obs, actions):
        """Evaluate actions (for PPO update)"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


# =====================================
# Critic Network (Centralized)
# =====================================
class Critic(nn.Module):
    """Centralized value function - sees all agents"""
    def __init__(self, total_obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)


# =====================================
# Experience Buffer
# =====================================
class RolloutBuffer:
    """Store experience for PPO update"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = []      # Individual obs for each agent
        self.joint_observations = []  # Concatenated obs for critic
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, joint_obs, actions, log_probs, rewards, dones, values):
        self.observations.append(obs)
        self.joint_observations.append(joint_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)

    def get(self):
        """Convert lists to tensors"""
        return {
            'observations': self.observations,
            'joint_observations': torch.stack(self.joint_observations),
            'actions': self.actions,
            'log_probs': self.log_probs,
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.float32),
            'values': torch.stack(self.values)
        }


# =====================================
# MAPPO Agent
# =====================================
class MAPPO:
    def __init__(
        self,
        n_agents=2,
        obs_dim=192,  # 8*8*3 flattened
        act_dim=5,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        device='cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device

        # Create one actor per agent (decentralized execution)
        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]

        # Create one centralized critic (sees all agents)
        self.critic = Critic(obs_dim * n_agents).to(device)

        # Optimizers
        actor_params = []
        for actor in self.actors:
            actor_params.extend(actor.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Buffer
        self.buffer = RolloutBuffer()

    def select_actions(self, observations):
        """Get actions from all agents"""
        actions = []
        log_probs = []
        entropies = []

        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action, log_prob, entropy = self.actors[i].get_action(obs_tensor)
                actions.append(action.item())
                log_probs.append(log_prob)
                entropies.append(entropy)

        return actions, log_probs, entropies

    def get_value(self, joint_obs):
        """Get value estimate from centralized critic"""
        with torch.no_grad():
            value = self.critic(joint_obs)
        return value

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        values = values.tolist()
        rewards = rewards.tolist()
        dones = dones.tolist()

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)

        return advantages, returns

    def update(self):
        """PPO update using collected experience"""
        data = self.buffer.get()

        # Compute advantages
        next_value = 0.0  # Assuming episodes end
        advantages, returns = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones'],
            next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        for _ in range(self.ppo_epochs):
            # Update each agent's actor
            actor_losses = []
            entropy_losses = []

            for agent_id in range(self.n_agents):
                # Get this agent's observations and actions
                agent_obs = torch.stack([data['observations'][t][agent_id] for t in range(len(data['observations']))])
                agent_actions = torch.tensor([data['actions'][t][agent_id] for t in range(len(data['actions']))], dtype=torch.long).to(self.device)
                old_log_probs = torch.stack([data['log_probs'][t][agent_id] for t in range(len(data['log_probs']))]).to(self.device)

                # Evaluate actions with current policy
                new_log_probs, entropy = self.actors[agent_id].evaluate(agent_obs, agent_actions)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                actor_losses.append(actor_loss)
                entropy_losses.append(entropy.mean())

            # Combined actor loss
            total_actor_loss = sum(actor_losses) - self.entropy_coef * sum(entropy_losses)

            # Update actors
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(actor_params := [p for actor in self.actors for p in actor.parameters()], self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            joint_obs_batch = data['joint_observations']
            new_values = self.critic(joint_obs_batch)
            critic_loss = nn.MSELoss()(new_values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Clear buffer
        self.buffer.reset()

        return {
            'actor_loss': sum(actor_losses).item() / self.n_agents,
            'critic_loss': critic_loss.item(),
            'entropy': sum(entropy_losses).item() / self.n_agents,
        }


# =====================================
# Training Loop
# =====================================
def train_mappo(
    total_timesteps=500_000,
    n_rollout_steps=2048,
    grid_size=8,
    max_steps=100,
    n_agents=2,
    reward_shaping=False,
    device='cpu'
):
    """Train MAPPO agents"""

    # Create environment
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=n_agents,
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=reward_shaping,
        #modified to include safety configuration - lagrangian method    
        safety_cfg= {
             "enabled": True,
            "n_hazards": 6,
            # "hazard_cost": 1.0, #fixed penalty during evaluation
            "use_lagrangian": True,
            "lambda_coeff": 2.0,
            "safety_budget": 3.0,
        }
    )

    obs_dim = np.prod(env.observation_space('agent_0').shape)
    act_dim = env.action_space('agent_0').n

    # Create MAPPO agent
    mappo = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device
    )

    # Setup logging
    os.makedirs("./checkpoints/mappo/crl5/a1/", exist_ok=True)
    os.makedirs("./tensorboard/mappo/crl5/a1/", exist_ok=True)
    writer = SummaryWriter("./tensorboard/mappo/crl5/a1/")

    print("=" * 60)
    print("MAPPO Training - Clean Implementation")
    print("=" * 60)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Max Steps: {max_steps}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Reward Shaping: {reward_shaping}")
    print(f"Rollout Steps: {n_rollout_steps}")
    print("=" * 60)

    # Training loop
    timesteps = 0
    episode = 0
    episode_rewards = []

    obs_dict, _ = env.reset()
    episode_reward = 0

    while timesteps < total_timesteps:
        # Collect rollout
        for step in range(n_rollout_steps):
            # Prepare observations
            observations = [
                torch.FloatTensor(obs_dict[f'agent_{i}'].flatten()).to(device)
                for i in range(n_agents)
            ]
            joint_obs = torch.cat(observations)

            # Get actions
            actions, log_probs, _ = mappo.select_actions([obs.cpu().numpy() for obs in observations])

            # Get value estimate
            value = mappo.get_value(joint_obs)

            # Step environment
            action_dict = {f'agent_{i}': actions[i] for i in range(n_agents)}
            next_obs_dict, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

            # Sum rewards for team
            total_reward = sum(reward_dict.values())
            done = all(done_dict.values()) or all(trunc_dict.values())

            # Store in buffer
            mappo.buffer.add(
                observations,
                joint_obs,
                actions,
                log_probs,
                total_reward,
                done,
                value
            )

            episode_reward += total_reward
            timesteps += 1

            # Update observation
            obs_dict = next_obs_dict

            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode += 1
                obs_dict, _ = env.reset()
                episode_reward = 0

        # PPO Update
        stats = mappo.update()

        # Logging
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

            print(f"Timesteps: {timesteps:7d} | "
                  f"Episodes: {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Actor Loss: {stats['actor_loss']:7.4f} | "
                  f"Critic Loss: {stats['critic_loss']:7.4f} | "
                  f"Entropy: {stats['entropy']:6.3f}")

            writer.add_scalar("Reward/Average", avg_reward, timesteps)
            writer.add_scalar("Loss/Actor", stats['actor_loss'], timesteps)
            writer.add_scalar("Loss/Critic", stats['critic_loss'], timesteps)
            writer.add_scalar("Entropy", stats['entropy'], timesteps)

        # Save checkpoint
        if timesteps % 50000 == 0:
            checkpoint = {
                'actors': [actor.state_dict() for actor in mappo.actors],
                'critic': mappo.critic.state_dict(),
                'timesteps': timesteps
            }
            torch.save(checkpoint, f"./checkpoints/mappo/crl5/a1/checkpoint_{timesteps}.pt")
            print(f"  💾 Checkpoint saved at {timesteps} timesteps")

    # Save final model
    final_checkpoint = {
        'actors': [actor.state_dict() for actor in mappo.actors],
        'critic': mappo.critic.state_dict(),
        'timesteps': timesteps
    }
    torch.save(final_checkpoint, "./checkpoints/mappo/crl5/a1/final_model.pt")

    writer.close()
    env.close()

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"Final Average Reward: {np.mean(episode_rewards[-50:]):.2f}")
    print("=" * 60)

    return mappo


# =====================================
# Evaluation
# =====================================
def evaluate_mappo(checkpoint_path, n_episodes=20, grid_size=8, max_steps=100, render=False):
    """Evaluate trained MAPPO"""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create environment
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=2,
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=True,
        #modified to include safety configuration - lagrangian method    
        safety_cfg= {
             "enabled": True,
            "n_hazards": 6,
            # "hazard_cost": 1.0, #fixed penalty during evaluation
            "use_lagrangian": True,
            "lambda_coeff": 2.0,
            "safety_budget": 3.0,
        }

    )

    obs_dim = np.prod(env.observation_space('agent_0').shape)
    act_dim = env.action_space('agent_0').n

    # Create MAPPO and load weights
    mappo = MAPPO(n_agents=2, obs_dim=obs_dim, act_dim=act_dim, device='cpu')
    for i, actor in enumerate(mappo.actors):
        actor.load_state_dict(checkpoint['actors'][i])
    mappo.critic.load_state_dict(checkpoint['critic'])

    print("\n" + "=" * 60)
    print("Evaluating MAPPO")
    print("=" * 60)

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    individual_goals = {'agent_0': 0, 'agent_1': 0}

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Get actions
            observations = [obs_dict[f'agent_{i}'].flatten() for i in range(2)]
            actions, _, _ = mappo.select_actions(observations)

            # Step
            action_dict = {f'agent_{i}': actions[i] for i in range(2)}
            obs_dict, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

            episode_reward += sum(reward_dict.values())
            steps += 1
            done = all(done_dict.values()) or all(trunc_dict.values())

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Track goals
        for agent in env.agents:
            if env.goals_reached[agent]:
                individual_goals[agent] += 1

        if all(env.goals_reached.values()):
            success_count += 1

        print(f"Episode {ep+1:2d}: Reward = {episode_reward:6.2f}, "
              f"Steps = {steps:3d}, Goals: {sum(env.goals_reached.values())}/2")

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average Reward: {np.mean(episode_rewards):6.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Average Length: {np.mean(episode_lengths):6.2f} ± {np.std(episode_lengths):6.2f}")
    print(f"Success Rate: {success_count/n_episodes*100:5.1f}%")
    print(f"\nIndividual Goal Success:")
    for agent, count in individual_goals.items():
        print(f"  {agent}: {count/n_episodes*100:5.1f}%")

    env.close()


# =====================================
# Main
# ===================================== 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate MAPPO (CRL5)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/mappo/crl5/a1/final_model.pt')

    args = parser.parse_args()

    if args.mode == 'train':
        mappo = train_mappo(total_timesteps=args.timesteps)
        print("\nRunning evaluation...")
        evaluate_mappo('./checkpoints/mappo/crl5/a1/final_model.pt', n_episodes=args.episodes)
    else:
        evaluate_mappo(args.checkpoint, n_episodes=args.episodes)
