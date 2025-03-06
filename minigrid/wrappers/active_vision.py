from __future__ import annotations

import numpy as np
from gymnasium import spaces
from minigrid.core.wrapper import ObservationWrapper
from minigrid.wrappers.symbolic_obs import SymbolicObsWrapper


class ActiveVisionWrapper(ObservationWrapper):
    """
    Wrapper that adds a movable camera with partial observability to the environment.
    The camera can move independently of the agent and only shows a partial view of the world.
    """
    def __init__(self, env, camera_size=2):
        # First wrap with SymbolicObsWrapper to get full grid observation
        env = SymbolicObsWrapper(env)
        super().__init__(env)

        self.camera_size = camera_size
        self.grid_size = env.width  # Assuming square grid

        # Initialize camera position at center of grid
        self.camera_pos = np.array([self.grid_size // 2, self.grid_size // 2])

        # Add camera movement actions: up, down, left, right
        self.camera_actions = 4
        self.base_actions = env.action_space.n
        self.action_space = spaces.Discrete(self.base_actions + self.camera_actions)

        # Update observation space to include camera position
        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            'camera_pos': spaces.Box(
                low=0,
                high=self.grid_size-1,
                shape=(2,),
                dtype=np.int32
            )
        })

    def observation(self, obs):
        """Apply camera masking to the observation."""
        image = obs['image']
        masked_image = np.zeros_like(image)

        # Calculate camera view boundaries
        x_start = max(0, self.camera_pos[0] - self.camera_size // 2)
        x_end = min(self.grid_size, x_start + self.camera_size)
        y_start = max(0, self.camera_pos[1] - self.camera_size // 2)
        y_end = min(self.grid_size, y_start + self.camera_size)

        # Only show the area within camera view
        masked_image[y_start:y_end, x_start:x_end] = image[y_start:y_end, x_start:x_end]

        return {
            **obs,
            'image': masked_image,
            'camera_pos': self.camera_pos.copy()
        }

    def step(self, action):
        """Handle both environment and camera actions."""
        # Camera movement actions
        if action >= self.base_actions:
            camera_action = action - self.base_actions
            # Update camera position based on action
            if camera_action == 0:  # up
                self.camera_pos[1] = max(0, self.camera_pos[1] - 1)
            elif camera_action == 1:  # down
                self.camera_pos[1] = min(self.grid_size - 1, self.camera_pos[1] + 1)
            elif camera_action == 2:  # left
                self.camera_pos[0] = max(0, self.camera_pos[0] - 1)
            elif camera_action == 3:  # right
                self.camera_pos[0] = min(self.grid_size - 1, self.camera_pos[0] + 1)

            # Camera movement doesn't change environment state
            obs = self.env._get_obs()
            return self.observation(obs), 0.0, False, False, {}

        # Regular environment actions
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset both environment and camera position."""
        obs, info = self.env.reset(**kwargs)
        # Reset camera to center
        self.camera_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        return self.observation(obs), info