from __future__ import annotations

from datetime import datetime
from typing import List

import gymnasium as gym
import imageio
import networkx as nx
import numpy as np
from dm_env_wrappers import GymnasiumWrapper
from gymnasium.wrappers import FilterObservation

from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC
from minigrid.envs.wfc.graphtransforms import GraphTransforms
from minigrid.envs.wfc.wfcenv import EDGE_CONFIG, FEATURE_DESCRIPTORS
from minigrid.wrappers import FullyObsWrapper, ReseedWrapper


def path_to_actions(path: list, current_direction_index: int) -> List[int]:
    actions = []
    dir_to_vec = [list(v) for v in DIR_TO_VEC]
    for i in range(len(path) - 1):
        direction_vector = np.array(path[i+1]) - np.array(path[i])
        desired_direction_index = dir_to_vec.index(list(direction_vector))

        while current_direction_index != desired_direction_index:
            # Calculate the difference between the current and desired direction indices
            diff = desired_direction_index - current_direction_index

            # If the difference is positive, the shortest path is to turn right
            if diff > 0 or (diff < 0 and abs(diff) > len(dir_to_vec) / 2):
                actions.append(Actions.right.value)  # Turn right
                current_direction_index = (current_direction_index + 1) % len(dir_to_vec)
            else:
                actions.append(Actions.left.value)  # Turn left
                current_direction_index = (current_direction_index - 1) % len(dir_to_vec)

        actions.append(Actions.forward.value)  # Move forward
    return actions

class MazePolicy(object):
    """
    Maze policy that computes shortest path with epsilon-greedy exploration.
    The policy first observes the maze through the reset, and computes the shortest path.
    """
    def __init__(self, epsilon, low, high, seed):
        self.epsilon = epsilon
        assert self.epsilon >= 0.0 and self.epsilon <= 1.0
        self.low = low
        self.high = high
        self.np_random = np.random.RandomState(seed)

    def reset(self, obs):
        if self.epsilon != 1.0:
            dir = obs["direction"]
            path = self._compute_path(obs)
            self.actions = path_to_actions(path, dir)
            self.action_index = 0
            self.prev_mode = "*"
        else:
            self.prev_mode = "e"

    def _compute_path(self, obs):
        img = obs["image"]
        dense_graph = GraphTransforms.minigrid_to_dense_graph([img], node_attr=FEATURE_DESCRIPTORS, edge_config=EDGE_CONFIG)[0]
        edge_layers = GraphTransforms.get_edge_layers(dense_graph, edge_config=EDGE_CONFIG, node_attr=FEATURE_DESCRIPTORS, dim_grid=img.shape[:2])
        graph = edge_layers["navigable"]
        # Find the start and goal nodes
        start_node = [n for n, d in graph.nodes(data=True) if d.get('start') == 1][0]
        goal_node = [n for n, d in graph.nodes(data=True) if d.get('goal') == 1][0]
        # Find the shortest path
        path = nx.shortest_path(graph, start_node, goal_node)
        return path 

    def step(self, obs):
        """
        decide if we should recompute the path or not based on previous mode.
                compute path?
        * -> *    N
        e -> *    Y
        * -> e    N
        e -> e    N
        """
        mode = 'e' if self.np_random.rand() < self.epsilon else '*'
        if self.prev_mode == '*' and mode == '*':
            pass
        elif self.prev_mode == 'e' and mode == '*':
            self.reset(obs)
        elif self.prev_mode == '*' and mode == 'e':
            pass
        elif self.prev_mode == 'e' and mode == 'e':
            pass

        self.prev_mode = mode

        if mode == 'e':
            action = self.np_random.randint(low=self.low, high=self.high)
            return action

        if self.action_index < len(self.actions):
            action = self.actions[self.action_index]
            self.action_index += 1
        else:
            action = self.np_random.randint(low=self.low, high=self.high)
        return action

def test_maze_policy():
    env = gym.make("MiniGrid-WFC-MazeSimple-v0", render_mode="rgb_array", tile_size=16, max_steps=100)
    env = FullyObsWrapper(env)
    env = FilterObservation(env, ["image", "direction"])
    env = ReseedWrapper(env, seeds=[0])
    env = GymnasiumWrapper(env)
    low = env.action_spec().minimum
    high = env.action_spec().maximum + 1

    video = []
    epsilon = 0.9
    policy = MazePolicy(epsilon=epsilon, low=low, high=high, seed=0)

    for _ in range(5):
        timestep = env.reset()
        policy.reset(timestep.observation)
        video.append(env.environment.render())
        while not timestep.last():
            action = policy.step(timestep.observation)
            timestep = env.step(action)
            video.append(env.environment.render())
        ending = "terminated" if timestep.discount == 0.0 else "timeout"
        now = datetime.now()
        datetime_str = now.strftime("%Y%m%d_%H%M%S")
        imageio.mimsave(f"policy_{epsilon}_{ending}_{datetime_str}.mp4", video, fps=20)

def test_path_to_actions():
    env = gym.make("MiniGrid-WFC-MazeSimple-v0", render_mode="rgb_array", tile_size=16)
    env = FullyObsWrapper(env)
    env = FilterObservation(env, ["image", "direction"])
    env = GymnasiumWrapper(env)
    low = env.action_spec().minimum
    high = env.action_spec().maximum + 1

    video = []

    timestep = env.reset()
    img = timestep.observation["image"]
    dir = timestep.observation["direction"]
    video.append(env.environment.render())
    # imageio.imwrite("maze.png", img)
    dense_graph = GraphTransforms.minigrid_to_dense_graph([img], node_attr=FEATURE_DESCRIPTORS, edge_config=EDGE_CONFIG)[0]

    edge_layers = GraphTransforms.get_edge_layers(dense_graph, edge_config=EDGE_CONFIG, node_attr=FEATURE_DESCRIPTORS, dim_grid=img.shape[:2])
    graph = edge_layers["navigable"]
    # Find the start and goal nodes
    start_node = [n for n, d in graph.nodes(data=True) if d.get('start') == 1][0]
    goal_node = [n for n, d in graph.nodes(data=True) if d.get('goal') == 1][0]
    # Find the shortest path
    path = nx.shortest_path(graph, start_node, goal_node)
    actions = path_to_actions(path, dir)
    print(actions)
    action_index = 0
    while not timestep.last():
        if action_index < len(actions):
            action = actions[action_index]
            action_index += 1
        else:
            action = np.random.randint(low=low, high=high)
            print('doing random actions now')
        timestep = env.step(action)
        video.append(env.environment.render())
    imageio.mimsave("maze.mp4", video, fps=10)

if __name__ == "__main__":
    test_maze_policy()