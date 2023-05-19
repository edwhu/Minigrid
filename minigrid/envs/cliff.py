from __future__ import annotations

from enum import Enum
import itertools as itt
from operator import add

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Ball
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.spaces import Discrete

class MAP_TYPE(Enum):
    CROSS = 'CROSS',
    SLOT = 'SLOT',
    SPIRAL = 'SPIRAL'

class CliffEnv(MiniGridEnv):

    """
    ## Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Lava` :
        - `MiniGrid-LavaCrossingS9N1-v0`
        - `MiniGrid-LavaCrossingS9N2-v0`
        - `MiniGrid-LavaCrossingS9N3-v0`
        - `MiniGrid-LavaCrossingS11N5-v0`

    - otherwise :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`

    """

    def __init__(
        self,
        size=9,
        obstacle_type=Lava,
        n_obstacles=1,
        max_steps: int | None = None,
        map_type: MAP_TYPE = MAP_TYPE.CROSS,
        **kwargs,
    ):
        self.obstacle_type = obstacle_type
        self.map_type = map_type

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

        # Allow only 3 actions permitted: left, right, forward
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (0, 1)

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and moving obstacles and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "avoid the moving obstacles and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)

        # place lava walls
        if self.map_type == MAP_TYPE.CROSS:
            self.grid.vert_wall(width//2, 2, height-4, Lava)
            self.grid.horz_wall(2, height//2, width - 4, Lava)
        elif self.map_type == MAP_TYPE.SLOT:
            if width >= 7 and height >= 7:
                self.grid.horz_wall(2, 2, width-4, Lava)  # top wall
                self.grid.vert_wall(2, 2, height-4, Lava)  # left
                self.grid.vert_wall(width-3, 2, height-4, Lava)  # right
                self.grid.horz_wall(2, height-3, (width-4-1)//2, Lava)
                self.grid.horz_wall(2+(width-4-1)//2 + 1, height-3, (width-4-1)//2, Lava)  # bottom w/ slot
        elif self.map_type == MAP_TYPE.SPIRAL:
            l_idx, t_idx = 2, 0  # left, top
            r_idx, b_idx = width - 3, height - 3  # right, bottom
            while r_idx - l_idx >= 0 and b_idx - t_idx >= 0:
                self.grid.vert_wall(l_idx, t_idx, b_idx-t_idx+1, Lava)  # left
                t_idx += 2
                self.grid.horz_wall(l_idx, b_idx, r_idx-l_idx+1, Lava)  # bottom
                l_idx += 2
                self.grid.vert_wall(r_idx, t_idx, b_idx-t_idx+1, Lava)  # right
                b_idx -= 2
                self.grid.horz_wall(l_idx, t_idx, r_idx-l_idx+1, Lava)  # top
                r_idx -= 2
            self.grid.vert_wall(2, 0, 1)  # replace wall from first vertical lava placement
                

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)


    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
            except Exception:
                pass

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            # reward = -1
            # terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    env = CliffEnv(size=7, n_obstacles=1, render_mode="human")
    env.reset()
    while True:
        env.render()