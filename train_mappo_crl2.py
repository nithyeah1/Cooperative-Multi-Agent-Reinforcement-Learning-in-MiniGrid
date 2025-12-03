import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiagent_minigrid_env import MultiAgentGoalReachingEnv


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


class MAPPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents):
        super().__init__()
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


class MAPPOTrainer:
    def __init__(self, grid_size=8, n_agents=2, total_steps=500_000, rollout_len=2048):
        self.env = MultiAgentGoalReachingEnv(grid_size=grid_size, num_agents=n_agents)
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

        os.makedirs("./checkpoints/mappo/mappo_2/", exist_ok=True)
        os.makedirs("./tensorboard/mappo/mappo_2/", exist_ok=True)
        self.writer = SummaryWriter("./tensorboard/mappo/mappo_2/")

    def collect_rollout(self):
        obs, _ = self.env.reset()
        obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in obs.items()}

        storage = []
        for _ in range(self.rollout_len):
            actions, log_probs, rewards = {}, {}, {}
            joint_obs = torch.cat([obs[f'agent_{i}'] for i in range(self.n_agents)])

            for i in range(self.n_agents):
                agent_obs = obs[f'agent_{i}']
                action, log_prob, _ = self.model.act(agent_obs)
                actions[f'agent_{i}'] = action.item()
                log_probs[f'agent_{i}'] = log_prob

            next_obs, reward, done, trunc, _ = self.env.step(actions)
            rewards = np.mean(list(reward.values()))  # shared reward

            storage.append((joint_obs, torch.tensor(rewards, dtype=torch.float32), log_probs, actions))

            obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in next_obs.items()}

            if all(done.values()) or all(trunc.values()):
                obs, _ = self.env.reset()
                obs = {k: torch.tensor(v.flatten(), dtype=torch.float32) for k, v in obs.items()}

        return storage

    def compute_gae(self, rewards, values, next_value):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[step])
            next_value = values[step]
        return returns

    def train(self):
        total_updates = self.total_steps // self.rollout_len

        for update in range(total_updates):
            rollout = self.collect_rollout()

            obs_batch = torch.stack([r[0] for r in rollout])
            rewards = torch.stack([r[1] for r in rollout])

            with torch.no_grad():
                values = self.model.critic(obs_batch).squeeze()

            next_value = torch.tensor(0.0)
            returns = self.compute_gae(rewards, values, next_value)
            returns = torch.stack(returns).detach()

            advantages = (returns - values).detach()

            for epoch in range(4):
                log_probs, entropy = [], []
                new_values = self.model.critic(obs_batch).squeeze()

                for i in range(self.n_agents):
                    agent_obs = obs_batch[:, i * len(obs_batch[0]) // self.n_agents:(i + 1) * len(obs_batch[0]) // self.n_agents]
                    actions = torch.tensor([r[3][f'agent_{i}'] for r in rollout])
                    log_p, ent = self.model.evaluate_actions(agent_obs, actions)
                    log_probs.append(log_p)
                    entropy.append(ent)

                log_probs = torch.stack(log_probs).mean(0)
                entropy = torch.stack(entropy).mean()

                old_log_probs = torch.stack([r[2]['agent_0'] for r in rollout]).detach()

                ratio = torch.exp(log_probs - old_log_probs)
                ratio_clamped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)

                surr1 = ratio * advantages
                surr2 = ratio_clamped * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns - new_values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if update % 10 == 0:
                print(f"Update {update}/{total_updates}, Loss: {loss.item():.4f}")
                torch.save(self.model.state_dict(), f"./checkpoints/mappo/mappo_2/mappo_update_{update}.pt")

                self.writer.add_scalar("Loss/Total", loss.item(), update)
                self.writer.add_scalar("Loss/Actor", actor_loss.item(), update)
                self.writer.add_scalar("Loss/Critic", critic_loss.item(), update)
                self.writer.add_scalar("Entropy", entropy.item(), update)
                self.writer.add_scalar("Advantage/Mean", advantages.mean().item(), update)

        self.writer.close()


if __name__ == "__main__":
    trainer = MAPPOTrainer()
    trainer.train()
