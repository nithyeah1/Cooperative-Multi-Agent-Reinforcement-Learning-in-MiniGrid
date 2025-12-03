"""
LAGRANGIAN MAPPO (TRAIN + EVALUATE)
Compatible with MultiAgentGoalReachingEnv (safety version)

Includes:
 - Lagrangian dual update (lambda)
 - Safety cost tracking from env.infos['cost']
 - PPO training with centralized critic
 - Integrated evaluation function
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from multiagent_minigrid_env_safety_2b_saflag import MultiAgentGoalReachingEnv


# ============================================================
#  Actor (Decentralized Policy)
# ============================================================

class Actor(nn.Module):
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
        """
        obs: tensor [obs_dim]
        returns: action (scalar tensor), log_prob, entropy
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, obs, actions):
        """
        obs: [T, obs_dim]
        actions: [T]
        returns: log_probs [T], entropy [T]
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


# ============================================================
#  Critic (Centralized)
# ============================================================

class Critic(nn.Module):
    def __init__(self, joint_obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
#  Rollout Buffer
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = []   # list of [n_agents][obs_tensor]
        self.joint_obs = []      # list of joint obs tensors [joint_obs_dim]
        self.actions = []        # list of [n_agents] ints
        self.log_probs = []      # list of [n_agents] log_prob tensors
        self.rewards = []        # scalar team reward per step
        self.costs = []          # scalar team safety cost per step
        self.dones = []          # bool per step
        self.values = []         # scalar value per step

    def add(self, obs, joint_obs, actions, log_probs, reward, cost, done, value):
        self.observations.append(obs)
        self.joint_obs.append(joint_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.values.append(value)

    def get(self):
        """
        Returns tensors / lists ready for training.
        """
        return {
            "observations": self.observations,
            "joint_obs": torch.stack(self.joint_obs),  # [T, joint_obs_dim]
            "actions": self.actions,
            "log_probs": self.log_probs,
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "costs": torch.tensor(self.costs, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
            "values": torch.stack(self.values),        # [T]
        }


# ============================================================
#  MAPPO Lagrangian Agent
# ============================================================

class MAPPO_Lagrangian:
    def __init__(
        self,
        n_agents,
        obs_dim,
        act_dim,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        device="cpu",
        safety_budget=15.0, 
        lambda_lr=0.001,     
        lambda_init=0.0,
        lambda_max=0.1,      
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lambda_max = lambda_max
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device

        # Lagrangian parameters
        self.lambda_coeff = lambda_init
        self.lambda_lr = lambda_lr
        self.safety_budget = safety_budget

        # Actors (parameter-shared or separate – here separate per agent)
        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.actor_optimizer = optim.Adam(
            [p for actor in self.actors for p in actor.parameters()],
            lr=lr
        )

        # Centralized critic
        self.critic = Critic(obs_dim * n_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Buffer
        self.buffer = RolloutBuffer()

    # -------------------------------
    def select_actions(self, obs_list_np):
        """
        obs_list_np: list of numpy arrays [obs_dim] for each agent
        returns: actions (ints), log_probs (tensors), entropies (tensors)
        """
        actions, logps, entropies = [], [], []
        with torch.no_grad():
            for i, obs_np in enumerate(obs_list_np):
                obs_t = torch.FloatTensor(obs_np).to(self.device)
                a, lp, ent = self.actors[i].get_action(obs_t)
                actions.append(a.item())
                logps.append(lp)
                entropies.append(ent)
        return actions, logps, entropies

    # -------------------------------
    def get_value(self, joint_obs):
        with torch.no_grad():
            return self.critic(joint_obs)

    # -------------------------------
    def compute_gae(self, rewards, values, dones, next_value=0.0):
        """
        Standard GAE for scalar team reward.
        rewards, values, dones: 1D tensors of length T
        """
        advantages = []
        gae = 0.0

        rewards = rewards.tolist()
        values = values.tolist()
        dones = dones.tolist()

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        return advantages, returns

    # -------------------------------
    def update(self):
        """
        PPO update using collected experience in self.buffer.
        """
        data = self.buffer.get()
        rewards = data["rewards"].to(self.device)
        dones = data["dones"].to(self.device)
        values = data["values"].to(self.device)

        # GAE
        advantages, returns = self.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=0.0
        )

        # Safe advantage normalization
        if advantages.numel() > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / std
            else:
                advantages = advantages - advantages.mean()

        # PPO updates
        for _ in range(self.ppo_epochs):
            actor_losses = []
            entropies = []

            T = len(data["observations"])

            for agent_id in range(self.n_agents):
                # Build batch for this agent
                obs_batch = torch.stack(
                    [data["observations"][t][agent_id] for t in range(T)]
                ).to(self.device)  # [T, obs_dim]

                act_batch = torch.tensor(
                    [data["actions"][t][agent_id] for t in range(T)],
                    dtype=torch.long,
                    device=self.device
                )

                old_logp_batch = torch.stack(
                    [data["log_probs"][t][agent_id] for t in range(T)]
                ).to(self.device)

                new_logp, entropy = self.actors[agent_id].evaluate(obs_batch, act_batch)

                ratio = torch.exp(new_logp - old_logp_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(actor_loss)
                entropies.append(entropy.mean())

            total_actor_loss = sum(actor_losses) - self.entropy_coef * sum(entropies)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for actor in self.actors for p in actor.parameters()],
                self.max_grad_norm
            )
            self.actor_optimizer.step()

            # Critic update
            joint_obs_batch = data["joint_obs"].to(self.device)  # [T, joint_obs_dim]
            new_values = self.critic(joint_obs_batch)
            critic_loss = nn.MSELoss()(new_values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Clear buffer
        self.buffer.reset()

        return {
            "actor_loss": total_actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": sum(entropies).item() / self.n_agents,  # Average entropy
        }


# ============================================================
#  TRAINING LOOP WITH LAMBDA UPDATE (NO EARLY BREAK)
# ============================================================

def train_lagrangian(
    total_timesteps=300_000,
    rollout_steps=1024,
    grid_size=8,
    max_steps=100,
    n_agents=2,
    device="cpu"
):
    # Environment with Lagrangian shaping enabled
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=n_agents,
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=True,  # ENABLED: helps agents learn paths
        safety_cfg={
            "enabled": True,
            "n_hazards": 4,  # REDUCED: from 6 to 4 for easier navigation
            "use_lagrangian": True,
            "lambda_coeff": 0.0,   # will be updated dynamically
            "safety_budget": 15.0,  # Balanced: between 12-25 cost range
        }
    )

    obs_dim = int(np.prod(env.observation_space("agent_0").shape))
    act_dim = env.action_space("agent_0").n

    agent = MAPPO_Lagrangian(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        safety_budget=15.0,  # Balanced: matches env
        lambda_lr=0.001,  # Small LR with normalization (from papers)
        lambda_init=0.0,
        lambda_max=0.5  # Lower max - keeps penalty reasonable vs goal reward (1.0)
    )

    writer = SummaryWriter("./tensorboard/lagrangian/")
    os.makedirs("./checkpoints/lagrangian/", exist_ok=True)

    obs_dict, _ = env.reset()
    timesteps = 0
    episode_reward = 0.0
    episode_cost = 0.0
    episode_len = 0
    episode_idx = 0
    episode_rewards = []
    rollout_costs = []  # Track costs within current rollout
    cost_ema = 0.0  # EMA of costs for smooth lambda updates
    update_counter = 0  # Count episodes for periodic lambda updates

    print("\n=========== TRAINING LAGRANGIAN MAPPO (BEST) ===========\n")
    print(f"Initial Lambda: {agent.lambda_coeff:.4f}")
    print(f"Lambda Max: {agent.lambda_max:.2f}")
    print(f"Safety Budget: {agent.safety_budget:.1f}")
    print(f"Lambda LR: {agent.lambda_lr:.4f}")
    print("=" * 60)

    while timesteps < total_timesteps:

        # Collect one rollout (possible multiple episodes)
        for step in range(rollout_steps):

            # Prepare observations
            obs_list_np = [
                obs_dict[f"agent_{i}"].flatten()
                for i in range(n_agents)
            ]
            obs_tensors = [torch.FloatTensor(o).to(device) for o in obs_list_np]
            joint_obs = torch.cat(obs_tensors)

            # Select actions
            actions, logps, _ = agent.select_actions(obs_list_np)

            # Environment step
            action_dict = {f"agent_{i}": actions[i] for i in range(n_agents)}
            next_obs_dict, reward_dict, done_dict, trunc_dict, info_dict = env.step(action_dict)

            total_reward = sum(reward_dict.values())
            total_cost = sum(info["cost"] for info in info_dict.values())
            done = all(done_dict.values()) or all(trunc_dict.values())

            # Critic value estimate
            value = agent.get_value(joint_obs)

            # Store transition
            agent.buffer.add(
                obs_tensors,
                joint_obs,
                actions,
                logps,
                total_reward,
                total_cost,
                done,
                value
            )

            timesteps += 1
            episode_reward += total_reward
            episode_cost += total_cost
            episode_len += 1
            obs_dict = next_obs_dict

            # Handle episode termination
            if done:
                # Track costs for this rollout
                rollout_costs.append(episode_cost)
                episode_rewards.append(episode_reward)

                # Logging
                writer.add_scalar("Safety/EpisodeCost", episode_cost, timesteps)
                writer.add_scalar("Reward/EpisodeReward", episode_reward, timesteps)
                writer.add_scalar("Reward/EpisodeLength", episode_len, timesteps)

                episode_idx += 1
                episode_reward = 0.0
                episode_cost = 0.0
                episode_len = 0

                # Reset env and continue rollout (NO break)
                obs_dict, _ = env.reset()

            # If we reached total timesteps, stop collecting
            if timesteps >= total_timesteps:
                break

        # PPO update at end of rollout
        stats = agent.update()

        # Lambda update AFTER PPO update with EMA + normalization
        # Combine best of both approaches
        if len(rollout_costs) > 0:
            avg_rollout_cost = np.mean(rollout_costs)

            # Update EMA (exponential moving average)
            if cost_ema == 0.0:
                cost_ema = avg_rollout_cost
            else:
                cost_ema = 0.9 * cost_ema + 0.1 * avg_rollout_cost

            update_counter += len(rollout_costs)

            # Update lambda every 10 episodes (using EMA)
            if update_counter >= 10:
                old_lambda = agent.lambda_coeff
                violation = (cost_ema - agent.safety_budget) / agent.safety_budget  # Normalize!

                # Dual ascent with clipping
                agent.lambda_coeff = max(
                    0.0,
                    min(agent.lambda_max, agent.lambda_coeff + agent.lambda_lr * violation)
                )

                # Push updated lambda to environment for NEXT rollout
                env.lambda_coeff = agent.lambda_coeff

                # Debug print occasionally
                if episode_idx % 50 == 0:
                    print(f"  [Lambda Update] EMA Cost: {cost_ema:.2f}, Budget: {agent.safety_budget:.0f}, "
                          f"Violation: {violation:.3f}, Lambda: {old_lambda:.4f} → {agent.lambda_coeff:.4f}")

                update_counter = 0

            # Clear rollout costs
            rollout_costs = []

            # Log metrics
            writer.add_scalar("Lagrangian/Lambda", agent.lambda_coeff, timesteps)
            writer.add_scalar("Safety/RolloutAvgCost", avg_rollout_cost, timesteps)
            writer.add_scalar("Safety/CostEMA", cost_ema, timesteps)

        # Print progress logging
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

            print(f"Timesteps: {timesteps:7d} | "
                  f"Episodes: {episode_idx:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Lambda: {agent.lambda_coeff:6.4f} | "
                  f"Cost EMA: {cost_ema:5.2f} | "
                  f"Actor Loss: {stats['actor_loss']:7.4f} | "
                  f"Critic Loss: {stats['critic_loss']:7.4f} | "
                  f"Entropy: {stats['entropy']:6.3f}")

            writer.add_scalar("Reward/Average", avg_reward, timesteps)

        writer.add_scalar("Loss/Actor", stats["actor_loss"], timesteps)
        writer.add_scalar("Loss/Critic", stats["critic_loss"], timesteps)
        writer.add_scalar("Entropy", stats["entropy"], timesteps)

        # Save intermediate checkpoints
        if timesteps % 50_000 == 0 or timesteps >= total_timesteps:
            ckpt_path = f"./checkpoints/lagrangian/lagrangian_best_{timesteps}.pt"
            torch.save(
                {
                    "actors": [a.state_dict() for a in agent.actors],
                    "critic": agent.critic.state_dict(),
                    "lambda": float(agent.lambda_coeff),  # Convert to pure Python float
                    "timesteps": int(timesteps),          # Convert to pure Python int
                },
                ckpt_path
            )
            print(f"  💾 Saved checkpoint at {ckpt_path}")

    writer.close()
    env.close()

    # Save final checkpoint with clear name
    final_path = "./checkpoints/lagrangian/lagrangian_best_final.pt"
    torch.save(
        {
            "actors": [a.state_dict() for a in agent.actors],
            "critic": agent.critic.state_dict(),
            "lambda": float(agent.lambda_coeff),
            "timesteps": int(timesteps),
        },
        final_path
    )

    print("\n=========== TRAINING COMPLETE ===========")
    print(f"Final timesteps: {timesteps}")
    print(f"Final checkpoint: {final_path}")
    print("=========================================\n")

    return final_path


# ============================================================
#  EVALUATION
# ============================================================

def evaluate_lagrangian(
    checkpoint_path,
    n_episodes=50,
    grid_size=8,
    max_steps=100,
    device="cpu"
):
    # Load checkpoint (weights_only=False is safe for our own checkpoints)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create environment for evaluation (match training settings)
    env = MultiAgentGoalReachingEnv(
        grid_size=grid_size,
        num_agents=2,
        max_steps=max_steps,
        shared_reward=True,
        reward_shaping=True,  # MATCH TRAINING
        safety_cfg={
            "enabled": True,
            "n_hazards": 4,  # MATCH TRAINING
            # Evaluate under same Lagrangian shaping (can change if you want plain penalty)
            "use_lagrangian": True,
            "lambda_coeff": ckpt.get("lambda", 0.0),
            "safety_budget": 15.0,  # MATCH TRAINING
        }
    )

    obs_dim = int(np.prod(env.observation_space("agent_0").shape))
    act_dim = env.action_space("agent_0").n

    # Reconstruct agent and load parameters
    agent = MAPPO_Lagrangian(
        n_agents=2,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device
    )
    for i, actor in enumerate(agent.actors):
        actor.load_state_dict(ckpt["actors"][i])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.lambda_coeff = ckpt.get("lambda", 0.0)

    print("\n=========== LAGRANGIAN MAPPO EVALUATION ===========")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Loaded lambda: {agent.lambda_coeff:.4f}")
    print("===================================================\n")

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

        # track goal success
        for agent_name in env.agents:
            if env.goals_reached[agent_name]:
                individual_goals[agent_name] += 1

        if all(env.goals_reached.values()):
            success_count += 1

        print(f"Episode {ep+1:2d}: Reward = {reward_sum:7.2f}, "
              f"Steps = {steps:3d}, Cost = {cost_sum:4.1f}, "
              f"Goals: {sum(env.goals_reached.values())}/2")

    print("\n============== FINAL EVAL RESULTS ==============")
    print(f"Average Reward: {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):7.2f}")
    print(f"Average Length: {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):7.2f}")
    print(f"Average Cost:   {np.mean(episode_costs):7.2f} ± {np.std(episode_costs):7.2f}")
    print(f"Success Rate:   {success_count / n_episodes * 100:5.1f}%")
    print("\nIndividual Goal Success:")
    for agent_name, cnt in individual_goals.items():
        print(f"  {agent_name}: {cnt / n_episodes * 100:5.1f}%")
    print("=================================================\n")

    env.close()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lagrangian MAPPO Training + Evaluation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--timesteps", type=int, default=300_000, help="total training timesteps")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/lagrangian/lagrangian_best_final.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    device = args.device

    if args.mode == "train":
        ckpt_path = train_lagrangian(
            total_timesteps=args.timesteps,
            device=device
        )
        print("\nRunning evaluation on final checkpoint...")
        evaluate_lagrangian(
            checkpoint_path=ckpt_path,
            n_episodes=args.episodes,
            device=device
        )

    else:  # eval mode
        evaluate_lagrangian(
            checkpoint_path=args.checkpoint,
            n_episodes=args.episodes,
            device=device
        )
