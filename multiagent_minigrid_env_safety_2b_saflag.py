"""
Safety-Aware Multi-Agent Goal Reaching Environment using MiniGrid and PettingZoo

- Original baseline: same interface as before
- Added:
  * Hazard cells (unsafe regions)
  * Safety cost signal
  * Optional Lagrangian reward shaping: r' = r - lambda * cost
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, WorldObj
import matplotlib.pyplot as plt

# --------------------------------------------------
# Helper classes & functions
# --------------------------------------------------


class Agent(WorldObj):
    """Agent object for the grid"""
    def __init__(self, agent_id, color='red'):
        super().__init__('agent', color)
        self.agent_id = agent_id
        self.pos = None
        self.dir = 0  # Facing direction (0=right, 1=down, 2=left, 3=up)

    def can_overlap(self):
        return False

    def render(self, img):
        # Render agent as a colored square
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


def fill_coords(img, fn, color):
    """Fill coordinates in image"""
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color


def point_in_rect(xmin, xmax, ymin, ymax):
    """Check if point is in rectangle"""
    def fn(x, y):
        return xmin <= x <= xmax and ymin <= y <= ymax
    return fn


# Color mapping
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'yellow': np.array([255, 255, 0]),
    'purple': np.array([112, 39, 195]),  # used for hazards
    'grey': np.array([100, 100, 100]),
}


# --------------------------------------------------
# Environment
# --------------------------------------------------

class MultiAgentGoalReachingEnv(ParallelEnv):
    """
    Simple multi-agent goal reaching environment with optional safety constraints.

    - 2 agents start at different positions
    - Each agent has its own goal
    - Agents get reward when reaching their goal
    - Optional shared reward bonus when both reach goals
    - OPTIONAL SAFETY:
        * Hazard cells: stepping on them incurs a safety cost
        * Optionally: Lagrangian shaping r' = r - lambda * cost

    Safety config (safety_cfg dict):
        enabled: bool (default True if dict provided)
        n_hazards: int, number of hazard cells to sample
        hazard_cost: float, cost per violation (if not using Lagrangian)
        use_lagrangian: bool, if True, reward -= lambda_coeff * cost
        lambda_coeff: float, Lagrange multiplier λ
        terminate_on_violation: bool, if True, can end episode on violation
        max_violations_per_episode: int or None, cap per-episode violations
        safety_budget: float or None, logged in infos for analysis
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'multi_agent_goal_safety_v0'
    }

    def __init__(
        self,
        grid_size=10,
        num_agents=2,
        max_steps=100,
        shared_reward=True,
        reward_shaping=False,   # kept for compatibility with your trainer
        safety_cfg=None,        # NEW: dict for safety / Lagrangian config
        render_mode=None,
    ):
        self.grid_size = grid_size
        self._num_agents = num_agents
        self.max_steps = max_steps
        self.shared_reward = shared_reward
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        # Agent setup
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        super().__init__()
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # Observation and action spaces
        obs_shape = (grid_size, grid_size, 3)
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            for agent in self.possible_agents
        }
        # Actions: 0=left, 1=right, 2=up, 3=down, 4=stay
        self._action_spaces = {
            agent: spaces.Discrete(5)
            for agent in self.possible_agents
        }

        # Core state
        self.grid = None
        self.agents_dict = {}
        self.goals = {}
        self.goals_reached = {}
        self.step_count = 0

        # -----------------------------
        # Safety / Lagrangian settings
        # -----------------------------
        safety_cfg = safety_cfg or {}
        self.safety_enabled = bool(safety_cfg) and safety_cfg.get("enabled", True)
        self.n_hazards = safety_cfg.get("n_hazards", 4)
        self.hazard_cost = safety_cfg.get("hazard_cost", 1.0)

        self.use_lagrangian = safety_cfg.get("use_lagrangian", False)
        self.lambda_coeff = safety_cfg.get("lambda_coeff", 1.0)

        self.terminate_on_violation = safety_cfg.get("terminate_on_violation", False)
        self.max_violations_per_episode = safety_cfg.get("max_violations_per_episode", None)
        self.safety_budget = safety_cfg.get("safety_budget", None)  # purely informational

        # per-episode safety tracking
        self.hazard_positions = set()
        self.episode_costs = {}        # per-agent cumulative cost
        self.episode_violations = {}   # per-agent violation counts

    # ------------- PettingZoo API -------------

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ------------- Reset -------------

    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self.step_count = 0
        self.goals_reached = {agent: False for agent in self.agents}

        # Safety tracking
        self.episode_costs = {agent: 0.0 for agent in self.agents}
        self.episode_violations = {agent: 0 for agent in self.agents}
        self.hazard_positions = set()

        # Create grid
        self.grid = Grid(self.grid_size, self.grid_size)

        # Add walls around the perimeter
        for i in range(self.grid_size):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, self.grid_size - 1, Wall())
            self.grid.set(0, i, Wall())
            self.grid.set(self.grid_size - 1, i, Wall())

        # Place agents at random positions
        self.agents_dict = {}
        agent_colors = ['red', 'blue']

        for i, agent_name in enumerate(self.agents):
            while True:
                x = np.random.randint(1, self.grid_size - 1)
                y = np.random.randint(1, self.grid_size - 1)
                if self.grid.get(x, y) is None:
                    agent = Agent(i, color=agent_colors[i])
                    agent.pos = (x, y)
                    self.agents_dict[agent_name] = agent
                    self.grid.set(x, y, agent)
                    break

        # Place goals at random positions
        self.goals = {}
        goal_colors = ['green', 'yellow']

        for i, agent_name in enumerate(self.agents):
            while True:
                x = np.random.randint(1, self.grid_size - 1)
                y = np.random.randint(1, self.grid_size - 1)
                if self.grid.get(x, y) is None:
                    goal = Goal()
                    goal.color = goal_colors[i]
                    self.goals[agent_name] = (x, y)
                    self.grid.set(x, y, goal)
                    break

        # Place hazard cells (safety-critical regions)
        if self.safety_enabled and self.n_hazards > 0:
            self._place_hazards()

        # Get initial observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {
            "cost": 0.0,
            "episode_cost": 0.0,
            "hazard_violations": 0,
            "safety_budget": self.safety_budget,
        } for agent in self.agents}

        return observations, infos

    def _place_hazards(self):
        """Randomly place hazard cells that incur safety cost when stepped on."""
        available_positions = []

        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                # avoid walls (already excluded), agents, goals
                cell = self.grid.get(x, y)
                if cell is None:
                    # clear cell: candidate for hazard
                    available_positions.append((x, y))

        np.random.shuffle(available_positions)
        n = min(self.n_hazards, len(available_positions))
        self.hazard_positions = set(available_positions[:n])

    # ------------- Step -------------

    def step(self, actions):
        """Execute actions for all agents"""
        self.step_count += 1

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Execute actions for each agent
        for agent_name, action in actions.items():
            if agent_name in self.agents:
                self._execute_action(agent_name, action)

        # Check if goals are reached and compute rewards
        for agent_name in self.agents:
            agent = self.agents_dict[agent_name]
            goal_pos = self.goals[agent_name]

            # Base reward: goal / time step penalty
            if agent.pos == goal_pos:
                self.goals_reached[agent_name] = True
                reward = 1.0
            else:
                reward = -0.01  # time penalty

            # ---------------------------
            # SAFETY COST + LAGRANGIAN
            # ---------------------------
            cost = 0.0
            # Stepping on a hazard cell is a safety violation
            if self.safety_enabled and agent.pos in self.hazard_positions:
                cost = 1.0
                self.episode_costs[agent_name] += cost
                self.episode_violations[agent_name] += 1

                # Optional: immediate termination if violation is not allowed
                if self.terminate_on_violation:
                    terminations[agent_name] = True

                # Optional: cap number of violations per episode
                if (self.max_violations_per_episode is not None and
                        self.episode_violations[agent_name] >= self.max_violations_per_episode):
                    terminations[agent_name] = True

            # Lagrangian shaping vs fixed penalty
            if self.safety_enabled and cost > 0.0:
                if self.use_lagrangian:
                    reward -= self.lambda_coeff * cost  # r' = r - λ * cost
                else:
                    reward -= self.hazard_cost * cost   # plain penalty

            # (Optional) simple reward shaping based on distance to goal
            if self.reward_shaping:
                gx, gy = goal_pos
                ax, ay = agent.pos
                dist = abs(gx - ax) + abs(gy - ay)
                # small shaping: closer → slightly higher reward
                reward += -0.001 * dist

            rewards[agent_name] = reward
            observations[agent_name] = self._get_obs(agent_name)

            # termination: goal reached OR safety termination logic (if set above)
            if agent_name not in terminations:
                terminations[agent_name] = self.goals_reached[agent_name]

            truncations[agent_name] = self.step_count >= self.max_steps

            infos[agent_name] = {
                'goal_reached': self.goals_reached[agent_name],
                'cost': cost,
                'episode_cost': self.episode_costs[agent_name],
                'hazard_violations': self.episode_violations[agent_name],
                'safety_budget': self.safety_budget,
            }

        # Shared bonus reward if both agents reach goals
        if self.shared_reward and all(self.goals_reached.values()):
            for agent_name in self.agents:
                rewards[agent_name] += 5.0  # Bonus for coordination

        return observations, rewards, terminations, truncations, infos

    # ------------- Dynamics helpers -------------

    def _execute_action(self, agent_name, action):
        """Execute a single agent's action"""
        agent = self.agents_dict[agent_name]
        old_pos = agent.pos

        # Action mapping: 0=left, 1=right, 2=up, 3=down, 4=stay
        if action == 0:      # Left
            new_pos = (old_pos[0] - 1, old_pos[1])
        elif action == 1:    # Right
            new_pos = (old_pos[0] + 1, old_pos[1])
        elif action == 2:    # Up
            new_pos = (old_pos[0], old_pos[1] - 1)
        elif action == 3:    # Down
            new_pos = (old_pos[0], old_pos[1] + 1)
        else:                # Stay
            new_pos = old_pos

        # Check if new position is valid
        if self._is_valid_position(new_pos, agent_name):
            # Remove agent from old position
            if self.grid.get(*old_pos) == agent:
                self.grid.set(*old_pos, None)

            # Check if there's a goal at new position
            goal_obj = self.grid.get(*new_pos)
            if goal_obj is not None and isinstance(goal_obj, Goal):
                # Keep the goal, just update agent position
                pass

            # Place agent at new position
            agent.pos = new_pos
            # Only set agent in grid if not on goal
            if not isinstance(self.grid.get(*new_pos), Goal):
                self.grid.set(*new_pos, agent)

    def _is_valid_position(self, pos, agent_name):
        """Check if position is valid (within bounds and not blocked)"""
        x, y = pos

        # Check bounds
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        # Check for walls
        cell = self.grid.get(x, y)
        if cell is not None and isinstance(cell, Wall):
            return False

        # Check for other agents
        if cell is not None and isinstance(cell, Agent):
            return False

        # Hazards are traversable: they are not obstacles, just unsafe
        return True

    # ------------- Observations & Rendering -------------

    def _get_obs(self, agent_name):
        """Get observation for an agent (full grid view for now)"""
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Fill in grid (walls, goals, agents)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid.get(x, y)
                if cell is not None:
                    if isinstance(cell, Wall):
                        img[y, x] = COLORS['grey']
                    elif isinstance(cell, Goal):
                        img[y, x] = COLORS[cell.color]
                    elif isinstance(cell, Agent):
                        img[y, x] = COLORS[cell.color]
                else:
                    img[y, x] = [255, 255, 255]  # White for empty

        # Draw hazard cells (purple), under goals/agents
        for (hx, hy) in self.hazard_positions:
            # don't overwrite walls
            cell = self.grid.get(hx, hy)
            if not isinstance(cell, Wall):
                img[hy, hx] = COLORS['purple']

        # Draw goals explicitly (so they are visible if overlapping hazard)
        for other_agent, goal_pos in self.goals.items():
            goal_color = 'green' if other_agent == 'agent_0' else 'yellow'
            gx, gy = goal_pos
            img[gy, gx] = COLORS[goal_color]

        # Draw agents on top
        for other_agent, agent_obj in self.agents_dict.items():
            x, y = agent_obj.pos
            img[y, x] = COLORS[agent_obj.color]

        return img

    def render(self):
        """Render the environment"""
        if self.render_mode == 'human' or self.render_mode == 'rgb_array':
            img = self._render_grid()

            if self.render_mode == 'human':
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'Step: {self.step_count}')
                plt.pause(0.01)

            return img

    def _render_grid(self):
        """Render grid as RGB image with hazards"""
        cell_size = 32
        img = np.ones((self.grid_size * cell_size,
                       self.grid_size * cell_size, 3),
                      dtype=np.uint8) * 255

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                x_pixel = x * cell_size
                y_pixel = y * cell_size
                cell = self.grid.get(x, y)

                if cell is not None:
                    if isinstance(cell, Wall):
                        img[y_pixel:y_pixel+cell_size,
                            x_pixel:x_pixel+cell_size] = COLORS['grey']
                    elif isinstance(cell, Goal):
                        img[y_pixel:y_pixel+cell_size,
                            x_pixel:x_pixel+cell_size] = COLORS[cell.color]
                    elif isinstance(cell, Agent):
                        img[y_pixel:y_pixel+cell_size,
                            x_pixel:x_pixel+cell_size] = COLORS[cell.color]

        # Draw hazards as purple overlay
        for (hx, hy) in self.hazard_positions:
            x_pixel = hx * cell_size
            y_pixel = hy * cell_size
            img[y_pixel:y_pixel+cell_size,
                x_pixel:x_pixel+cell_size] = COLORS['purple']

        return img

    def close(self):
        """Close environment"""
        pass


# --------------------------------------------------
# Quick manual test
# --------------------------------------------------
if __name__ == "__main__":
    # Baseline behaviour (no safety, no Lagrangian)
    env = MultiAgentGoalReachingEnv(grid_size=8, num_agents=2)
    obs, infos = env.reset(seed=42)

    print("Baseline env created successfully!")
    print(f"Agents: {env.agents}")
    print(f"Observation shape: {obs['agent_0'].shape}")
    print(f"Action space: {env.action_space('agent_0')}")

    # Safety + Lagrangian example (this is what you'll use for constrained MAPPO):
    safety_env = MultiAgentGoalReachingEnv(
        grid_size=8,
        num_agents=2,
        safety_cfg={
            "enabled": True,
            "n_hazards": 6,
            "hazard_cost": 1.0,
            "use_lagrangian": True,
            "lambda_coeff": 2.0,          # λ
            "terminate_on_violation": False,
            "max_violations_per_episode": None,
            "safety_budget": 3.0,         # just logged in info
        }
    )

    obs, infos = safety_env.reset(seed=0)
    print("\nSafety env created successfully!")
    print(f"Hazard positions: {safety_env.hazard_positions}")

    for i in range(5):
        actions = {agent: safety_env.action_space(agent).sample()
                   for agent in safety_env.agents}
        obs, rewards, terms, truncs, infos = safety_env.step(actions)
        print(f"\nStep {i+1}:")
        print(f"  Rewards: {rewards}")
        print(f"  Costs:   {{a: infos[a]['cost'] for a in infos}}")
        if all(terms.values()) or all(truncs.values()):
            print("Episode finished!")
            break
