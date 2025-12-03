"""
Simple Multi-Agent Goal Reaching Environment using MiniGrid and PettingZoo
Two agents need to reach their respective goals
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, WorldObj
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


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
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn


# Color mapping
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'yellow': np.array([255, 255, 0]),
    'purple': np.array([112, 39, 195]),
    'grey': np.array([100, 100, 100]),
}


class MultiAgentGoalReachingEnv(ParallelEnv):
    """
    Simple multi-agent goal reaching environment
    - 2 agents start at different positions
    - Each agent has its own goal
    - Agents get reward when reaching their goal
    - Optional: Shared reward when both reach goals simultaneously
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'multi_agent_goal_v0'
    }
    
    def __init__(
        self,
        grid_size=10,
        num_agents=2,
        max_steps=100,
        shared_reward=True,
        render_mode=None,
        reward_shaping=True
    ):
        # Handle RLlib's EnvContext (config dict) or direct parameters
        if isinstance(grid_size, dict):
            # RLlib passes config as a dict
            config = grid_size
            self.grid_size = config.get("grid_size", 10)
            self._num_agents = config.get("num_agents", 2)
            self.max_steps = config.get("max_steps", 100)
            self.shared_reward = config.get("shared_reward", True)
            self.render_mode = config.get("render_mode", None)
            self.reward_shaping = config.get("reward_shaping", True)
        else:
            # Direct parameter passing
            self.grid_size = grid_size
            self._num_agents = num_agents
            self.max_steps = max_steps
            self.shared_reward = shared_reward
            self.render_mode = render_mode
            self.reward_shaping = reward_shaping

        # Agent setup
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]

        super().__init__()
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # Define observation and action spaces
        # Using simple grid observations (can modify to partial observability)
        obs_shape = (self.grid_size, self.grid_size, 3)
        
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            for agent in self.possible_agents
        }
        
        # Actions: 0=left, 1=right, 2=up, 3=down, 4=stay, 5=toggle (unused), 6=done (unused)
        self._action_spaces = {
            agent: spaces.Discrete(5)  # Simplified: left, right, up, down, stay
            for agent in self.possible_agents
        }
        
        self.grid = None
        self.agents_dict = {}
        self.goals = {}
        self.goals_reached = {}
        self.step_count = 0
        self.prev_distances = {}  # Track previous distances for reward shaping

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
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents.copy()
        self.step_count = 0
        self.goals_reached = {agent: False for agent in self.agents}
        
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

        # Initialize distances for reward shaping
        for agent_name in self.agents:
            agent_pos = self.agents_dict[agent_name].pos
            goal_pos = self.goals[agent_name]
            self.prev_distances[agent_name] = np.linalg.norm(
                np.array(agent_pos) - np.array(goal_pos)
            )

        # Get initial observations
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
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
            
            # Check if agent reached its goal
            if agent.pos == goal_pos:
                self.goals_reached[agent_name] = True
                rewards[agent_name] = 1.0
            else:
                # Base time penalty
                rewards[agent_name] = -0.01

                # Add reward shaping based on distance to goal
                if self.reward_shaping:
                    current_dist = np.linalg.norm(
                        np.array(agent.pos) - np.array(goal_pos)
                    )
                    prev_dist = self.prev_distances[agent_name]

                    # Reward for getting closer, penalty for moving away
                    # Using 0.1 as baseline (0.3 caused reward hacking)
                    distance_reward = 0.1 * (prev_dist - current_dist)
                    rewards[agent_name] += distance_reward

                    # Update previous distance
                    self.prev_distances[agent_name] = current_dist
            
            observations[agent_name] = self._get_obs(agent_name)
            terminations[agent_name] = self.goals_reached[agent_name]
            truncations[agent_name] = self.step_count >= self.max_steps
            infos[agent_name] = {'goal_reached': self.goals_reached[agent_name]}
        
        # Add shared reward if both agents reached goals
        if self.shared_reward and all(self.goals_reached.values()):
            for agent_name in self.agents:
                rewards[agent_name] += 5.0  # Bonus for coordination
        
        return observations, rewards, terminations, truncations, infos
    
    def _execute_action(self, agent_name, action):
        """Execute a single agent's action"""
        agent = self.agents_dict[agent_name]
        old_pos = agent.pos
        
        # Action mapping: 0=left, 1=right, 2=up, 3=down, 4=stay
        if action == 0:  # Left
            new_pos = (old_pos[0] - 1, old_pos[1])
        elif action == 1:  # Right
            new_pos = (old_pos[0] + 1, old_pos[1])
        elif action == 2:  # Up
            new_pos = (old_pos[0], old_pos[1] - 1)
        elif action == 3:  # Down
            new_pos = (old_pos[0], old_pos[1] + 1)
        else:  # Stay
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
        
        return True
    
    def _get_obs(self, agent_name):
        """Get observation for an agent (full grid view for now)"""
        # Create RGB image of the grid
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Fill in grid
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
        
        # Draw goals even if agents are on them
        for other_agent, goal_pos in self.goals.items():
            goal_color = 'green' if other_agent == 'agent_0' else 'yellow'
            img[goal_pos[1], goal_pos[0]] = COLORS[goal_color]
        
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
        """Render grid as RGB image"""
        img = np.ones((self.grid_size * 32, self.grid_size * 32, 3), dtype=np.uint8) * 255
        
        cell_size = 32
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid.get(x, y)
                
                x_pixel = x * cell_size
                y_pixel = y * cell_size
                
                if cell is not None:
                    if isinstance(cell, Wall):
                        img[y_pixel:y_pixel+cell_size, x_pixel:x_pixel+cell_size] = COLORS['grey']
                    elif isinstance(cell, Goal):
                        img[y_pixel:y_pixel+cell_size, x_pixel:x_pixel+cell_size] = COLORS[cell.color]
                    elif isinstance(cell, Agent):
                        img[y_pixel:y_pixel+cell_size, x_pixel:x_pixel+cell_size] = COLORS[cell.color]
        
        return img
    
    def close(self):
        """Close environment"""
        pass


# Test the environment
if __name__ == "__main__":
    env = MultiAgentGoalReachingEnv(grid_size=8, num_agents=2)
    observations, infos = env.reset(seed=42)
    
    print("Environment created successfully!")
    print(f"Agents: {env.agents}")
    print(f"Observation shape: {observations['agent_0'].shape}")
    print(f"Action space: {env.action_space('agent_0')}")
    
    # Run a few random steps
    for i in range(10):
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"\nStep {i+1}:")
        print(f"  Rewards: {rewards}")
        print(f"  Goals reached: {env.goals_reached}")
        
        if all(terminations.values()) or all(truncations.values()):
            print("Episode finished!")
            break