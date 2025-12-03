"""
Fixed Multi-Agent Goal Reaching Environment (Vector-Based)
----------------------------------------------------------
✅ Learnable structured observations
✅ Dense reward shaping for faster convergence
✅ Shared cooperative reward for MAPPO
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class MultiAgentGoalReachingEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_agent_goal_v1"}

    def __init__(self, grid_size=8, num_agents=2, max_steps=100, shared_reward=True, render_mode=None):
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.max_steps = max_steps
        self.shared_reward = shared_reward
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # Observation: [agent_x, agent_y, goal_x, goal_y, other_x, other_y]
        obs_shape = (6,)
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=grid_size - 1, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }

        # Actions: 0=left, 1=right, 2=up, 3=down, 4=stay
        self._action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

    @property
    def num_agents(self):
        return self._num_agents

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents.copy()
        self.step_count = 0

        # Random initial positions and goals
        positions, goals = [], []
        self.agents_dict, self.goals, self.goals_reached = {}, {}, {}
        for agent in self.agents:
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in positions:
                    positions.append(pos)
                    self.agents_dict[agent] = {"pos": pos}
                    break
        for agent in self.agents:
            while True:
                goal = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if goal not in positions + goals:
                    goals.append(goal)
                    self.goals[agent] = goal
                    break
        self.goals_reached = {agent: False for agent in self.agents}
        self.prev_dists = {agent: self._dist(self.agents_dict[agent]["pos"], self.goals[agent]) for agent in self.agents}
        obs = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        self.step_count += 1
        rewards, obs, term, trunc, info = {}, {}, {}, {}, {}

        # Move agents
        for agent, act in actions.items():
            if self.goals_reached[agent]:
                continue
            x, y = self.agents_dict[agent]["pos"]
            if act == 0: x -= 1
            elif act == 1: x += 1
            elif act == 2: y -= 1
            elif act == 3: y += 1
            x, y = np.clip(x, 0, self.grid_size - 1), np.clip(y, 0, self.grid_size - 1)
            other_positions = [self.agents_dict[a]["pos"] for a in self.agents if a != agent]
            if (x, y) not in other_positions:
                self.agents_dict[agent]["pos"] = (x, y)

        # Compute rewards
        for agent in self.agents:
            pos, goal = self.agents_dict[agent]["pos"], self.goals[agent]
            dist = self._dist(pos, goal)
            r = -0.01
            if pos == goal:
                self.goals_reached[agent] = True
                r += 1.0
            if dist < self.prev_dists[agent]:
                r += 0.05
            elif dist > self.prev_dists[agent]:
                r -= 0.02
            rewards[agent] = r
            self.prev_dists[agent] = dist

        # Cooperative bonus
        if self.shared_reward:
            reached = sum(self.goals_reached.values())
            for agent in self.agents:
                rewards[agent] += 0.25 * reached

        # Collect transitions
        for agent in self.agents:
            term[agent] = self.goals_reached[agent]
            trunc[agent] = self.step_count >= self.max_steps
            obs[agent] = self._get_obs(agent)
            info[agent] = {}
        return obs, rewards, term, trunc, info

    def _get_obs(self, agent):
        pos, goal = self.agents_dict[agent]["pos"], self.goals[agent]
        others = [a for a in self.agents if a != agent]
        other_pos = self.agents_dict[others[0]]["pos"] if others else (0, 0)
        return np.array([*pos, *goal, *other_pos], dtype=np.float32)

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def render(self):
        print(f"\nStep {self.step_count}")
        for agent in self.agents:
            pos, goal = self.agents_dict[agent]["pos"], self.goals[agent]
            print(f"{agent}: pos={pos}, goal={goal}, reached={self.goals_reached[agent]}")

    def close(self): pass


if __name__ == "__main__":
    env = MultiAgentGoalReachingEnv(grid_size=6)
    obs, _ = env.reset()
    for i in range(3):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, _ = env.step(actions)
        print(i, rew)
