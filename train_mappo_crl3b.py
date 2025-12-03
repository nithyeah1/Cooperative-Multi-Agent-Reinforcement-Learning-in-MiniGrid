import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiagent_minigrid_env import MultiAgentGoalReachingEnv


# =====================================
# 1️⃣ MLP Network
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
# 2️⃣ MAPPO Actor–Critic
# =====================================
class MAPPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents

        # Shared actor across agents
        self.actor = MLP(obs_dim, act_dim)

        # Centralized critic observes all agents
        self.critic = MLP(obs_dim * n_agents, 1)

    def act(self, obs):
        """obs: [obs_dim]"""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()

    def evaluate_actions(self, obs, actions):
        """obs: [N, obs_dim], actions: [N]"""
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


# =====================================
# 3️⃣ MAPPO TRAINER — FIXED VERSION
# =====================================
class MAPPOTrainer:
    def __init__(self, grid_size=8, n_agents=2, total_steps=500_000, rollout_len=2048):

        self.env = MultiAgentGoalReachingEnv(grid_size=grid_size, num_agents=n_agents, shared_reward=True)

        obs_dim = int(np.prod(self.env.observation_space('agent_0').shape))
        act_dim = self.env.action_space('agent_0').n

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.model = MAPPOActorCritic(obs_dim, act_dim, n_agents)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.total_steps = total_steps
        self.rollout_len = rollout_len
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.ppo_epochs = 4
        self.entropy_coef = 0.02
        self.critic_coef = 0.5

        os.makedirs("./checkpoints/mappo/mappo_3b/", exist_ok=True)
        os.makedirs("./tensorboard/mappo/mappo_3b/", exist_ok=True)
        self.ckpt_dir = "./checkpoints/mappo/mappo_3b/"

        self.writer = SummaryWriter("./tensorboard/mappo/mappo_3b/")

    # ---------------------------------------------------------------------
    #  Collect Rollouts (stores joint obs + per-agent actions + logprobs)
    # ---------------------------------------------------------------------
    def collect_rollout(self):

        obs, _ = self.env.reset()
        obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in obs.items()}

        joint_obs_list = []
        actions_list = []
        logprobs_list = []
        rewards_list = []
        dones_list = []

        steps = 0

        while steps < self.rollout_len:

            # joint obs for centralized critic
            joint_obs = torch.cat([obs[f'agent_{i}'] for i in range(self.n_agents)], dim=0)

            # per-agent actions and logprobs
            acts = []
            lps = []

            for i in range(self.n_agents):
                a, lp, _ = self.model.act(obs[f'agent_{i}'])
                acts.append(a.item())
                lps.append(lp)

            act_dict = {f'agent_{i}': acts[i] for i in range(self.n_agents)}
            next_obs, reward, done, trunc, _ = self.env.step(act_dict)

            team_reward = sum(reward.values())
            episode_done = float(all(done.values()) or all(trunc.values()))

            joint_obs_list.append(joint_obs)
            actions_list.append(torch.tensor(acts, dtype=torch.long))
            logprobs_list.append(torch.stack(lps))
            rewards_list.append(torch.tensor(team_reward, dtype=torch.float32))
            dones_list.append(torch.tensor(episode_done, dtype=torch.float32))

            if episode_done:
                next_obs, _ = self.env.reset()

            obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in next_obs.items()}

            steps += 1

        return (
            torch.stack(joint_obs_list),
            torch.stack(actions_list),
            torch.stack(logprobs_list),
            torch.stack(rewards_list),
            torch.stack(dones_list)
        )

    # ---------------------------------------------------------------------
    #  GAE
    # ---------------------------------------------------------------------
    def compute_gae(self, rewards, values, dones):
        T = rewards.size(0)
        adv = torch.zeros(T)
        gae = 0.0

        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            next_value = values[t + 1] if t < T - 1 else 0

            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.lam * non_terminal * gae
            adv[t] = gae

        return adv, adv + values

    # ---------------------------------------------------------------------
    #  TRAIN LOOP
    # ---------------------------------------------------------------------
    def train(self):
        total_updates = self.total_steps // self.rollout_len
        print("=" * 60)
        print("MAPPO Training (fixed version)")
        print("=" * 60)

        all_returns = []

        for update in range(total_updates):

            joint_obs, actions, old_lps, rewards, dones = self.collect_rollout()

            with torch.no_grad():
                values = self.model.critic(joint_obs).squeeze(-1)

            adv, ret = self.compute_gae(rewards, values, dones)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            T = joint_obs.size(0)
            n = self.n_agents
            d = self.obs_dim

            # reshape for per-agent PPO
            agent_obs = joint_obs.view(T, n, d).reshape(T * n, d)
            actions = actions.reshape(T * n)
            old_lps = old_lps.reshape(T * n).detach()
            adv_agents = adv.unsqueeze(1).expand(T, n).reshape(T * n)

            # PPO update
            for _ in range(self.ppo_epochs):

                new_lps, ent = self.model.evaluate_actions(agent_obs, actions)
                ratio = torch.exp(new_lps - old_lps)

                s1 = ratio * adv_agents
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_agents

                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = (ret - self.model.critic(joint_obs).squeeze(-1)).pow(2).mean()
                entropy_loss = ent.mean()

                loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            avg_return = ret.mean().item()
            all_returns.append(avg_return)

            print(
                f"[Update {update+1:04d}/{total_updates:04d}] "
                f"Loss: {loss.item():.4f} | "
                f"Actor: {actor_loss.item():.4f} | "
                f"Critic: {critic_loss.item():.4f} | "
                f"Entropy: {entropy_loss.item():.4f} | "
                f"Return: {avg_return:.3f}"
            )

            if (update + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.ckpt_dir + f"mappo_{update+1}.pt")

            self.writer.add_scalar("Loss/Total", loss.item(), update)
            self.writer.add_scalar("Loss/Actor", actor_loss.item(), update)
            self.writer.add_scalar("Loss/Critic", critic_loss.item(), update)
            self.writer.add_scalar("Entropy", entropy_loss.item(), update)
            self.writer.add_scalar("Return", avg_return, update)

        print("\nTraining Done!")

    # ---------------------------------------------------------------------
    #  EVALUATION
    # ---------------------------------------------------------------------
    def evaluate(self, n_episodes=20, grid_size=8, max_steps=100, render=False):

        env = MultiAgentGoalReachingEnv(grid_size=grid_size, num_agents=self.n_agents, max_steps=max_steps, shared_reward=True)
        print("Evaluating...")

        success = 0
        indiv = {f"agent_{i}": 0 for i in range(self.n_agents)}

        for ep in range(n_episodes):

            obs, _ = env.reset()
            obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in obs.items()}

            terminated = {a: False for a in env.agents}
            truncated = {a: False for a in env.agents}

            total = 0
            steps = 0

            while not (all(terminated.values()) or all(truncated.values())):

                actions = {}
                for a in env.agents:
                    o = obs[a]
                    act, _, _ = self.model.act(o)
                    actions[a] = act.item()

                next_obs, reward, terminated, truncated, _ = env.step(actions)
                total += sum(reward.values())

                obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in next_obs.items()}
                steps += 1

                if render:
                    env.render()

            if all(env.goals_reached.values()):
                success += 1

            for a in env.agents:
                if env.goals_reached[a]:
                    indiv[a] += 1

            print(f"Episode {ep+1}: Reward={total:.2f}, Steps={steps}, Goals={sum(env.goals_reached.values())}")

        print("\nSuccess Rate:", success / n_episodes * 100, "%")
        for a in indiv:
            print(a, ":", indiv[a] / n_episodes * 100, "%")


# =====================================
# 4️⃣ MAIN ENTRY POINT
# =====================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=None)

    args = parser.parse_args()

    trainer = MAPPOTrainer(grid_size=args.grid_size, total_steps=args.timesteps)

    if args.mode == "train":
        trainer.train()
        trainer.evaluate(
            n_episodes=args.episodes,
            grid_size=args.grid_size if hasattr(args, "grid_size") else 8,
            max_steps=args.max_steps,
        )

    else:
        if args.model_path is None:
            raise ValueError("Must pass --model-path for eval mode!")
        trainer.model.load_state_dict(torch.load(args.model_path))
        trainer.evaluate(
            n_episodes=args.episodes,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            render=False,
        )