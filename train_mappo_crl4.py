"""
Stable & Learnable MAPPO Training Script
----------------------------------------
✅ Uses vector-based environment
✅ Proper PPO clipping, advantage normalization
✅ Cooperative centralized critic
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiagent_minigrid_env_mappo import MultiAgentGoalReachingEnv


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x): return self.net(x)


class MAPPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim)
        self.critic = MLP(obs_dim * n_agents, 1)

    def act(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()

    def evaluate_actions(self, obs, actions):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        return logp, dist.entropy().mean()


class MAPPOTrainer:
    def __init__(self, grid=8, agents=2, steps=500_000, rollout=1024):
        self.env = MultiAgentGoalReachingEnv(grid_size=grid, num_agents=agents)
        obs_dim = self.env.observation_space('agent_0').shape[0]
        act_dim = self.env.action_space('agent_0').n
        self.model = MAPPOActorCritic(obs_dim, act_dim, agents)
        self.optim = optim.Adam(self.model.parameters(), lr=3e-4)

        self.n_agents = agents
        self.rollout = rollout
        self.total_steps = steps
        self.gamma, self.lam, self.clip = 0.99, 0.95, 0.2
        os.makedirs("./checkpoints/mappo/mappo_4", exist_ok=True)
        os.makedirs("./tensorboard/mappo/mappo_4", exist_ok=True)
        self.writer = SummaryWriter("./tensorboard/mappo/mappo_4")

    def collect(self):
        obs, _ = self.env.reset()
        obs = {a: torch.tensor(v, dtype=torch.float32) for a, v in obs.items()}
        data = []
        for _ in range(self.rollout):
            acts, logps = {}, {}
            joint = torch.cat([obs[f"agent_{i}"] for i in range(self.n_agents)])
            for i in range(self.n_agents):
                o = obs[f"agent_{i}"]
                a, lp, _ = self.model.act(o)
                acts[f"agent_{i}"] = a.item()
                logps[f"agent_{i}"] = lp
            nxt, r, d, t, _ = self.env.step(acts)
            R = sum(r.values())
            data.append((joint, torch.tensor(R, dtype=torch.float32), logps, acts))
            obs = {a: torch.tensor(v, dtype=torch.float32) for a, v in nxt.items()}
            if all(d.values()) or all(t.values()):
                obs, _ = self.env.reset()
                obs = {a: torch.tensor(v, dtype=torch.float32) for a, v in obs.items()}
        return data

    def compute_gae(self, rewards, values):
        adv, ret, gae = [], [], 0
        next_val = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lam * gae
            ret.insert(0, gae + values[i])
            next_val = values[i]
        adv = torch.stack(ret) - values
        return torch.stack(ret).detach(), (adv - adv.mean()) / (adv.std() + 1e-8)

    def train(self):
        updates = self.total_steps // self.rollout
        for u in range(updates):
            roll = self.collect()
            obs = torch.stack([r[0] for r in roll])
            rews = torch.stack([r[1] for r in roll])
            with torch.no_grad():
                vals = self.model.critic(obs).squeeze()
            rets, adv = self.compute_gae(rews, vals)

            for _ in range(4):
                new_vals = self.model.critic(obs).squeeze()
                logps_all, ents_all = [], []
                for i in range(self.n_agents):
                    idx = slice(i * obs.shape[1] // self.n_agents, (i + 1) * obs.shape[1] // self.n_agents)
                    o_i = obs[:, idx]
                    a_i = torch.tensor([r[3][f"agent_{i}"] for r in roll])
                    lp, ent = self.model.evaluate_actions(o_i, a_i)
                    logps_all.append(lp)
                    ents_all.append(ent)
                logp_new = torch.stack(logps_all).mean(0)
                ent = torch.stack(ents_all).mean()
                old = torch.stack([torch.stack(list(r[2].values())).mean() for r in roll]).detach()
                ratio = torch.exp(logp_new - old)
                s1, s2 = ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = (rets - new_vals).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * ent
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optim.step()

            if (u + 1) % 5 == 0:
                print(f"[{u+1}/{updates}] Loss={loss.item():.3f} | Return={rets.mean():.2f}")
                torch.save(self.model.state_dict(), f"./checkpoints/mappo/mappo_4/mappo_{u+1}.pt")
                self.writer.add_scalar("loss/total", loss.item(), u)
                self.writer.add_scalar("return/mean", rets.mean().item(), u)
        self.writer.close()

    def evaluate(self, n_episodes=10):
        env = MultiAgentGoalReachingEnv()
        obs_dim = env.observation_space('agent_0').shape[0]
        act_dim = env.action_space('agent_0').n
        rewards = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = {a: False for a in env.agents}
            total = 0
            while not all(done.values()):
                acts = {}
                for a in env.agents:
                    o = torch.tensor(obs[a], dtype=torch.float32)
                    act, _, _ = self.model.act(o)
                    acts[a] = act.item()
                obs, r, done, trunc, _ = env.step(acts)
                total += sum(r.values())
            rewards.append(total)
            print(f"Episode {ep+1} | Reward={total:.2f}")
        print(f"Avg Reward={np.mean(rewards):.2f}")
        env.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    t = MAPPOTrainer()
    if args.mode == "train": t.train()
    else:
        t.model.load_state_dict(torch.load("./checkpoints/mappo/mappo_4/mappo_485.pt", map_location="cpu"))
        t.evaluate(args.episodes)
