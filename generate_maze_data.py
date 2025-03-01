from __future__ import annotations

import os
import time
from multiprocessing import Process

import envlogger
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from dm_env_wrappers import GymnasiumWrapper
from envlogger.backends import rlds_utils, tfds_backend_writer
from gymnasium.wrappers import FilterObservation
from maze_solver import MazePolicy

from minigrid.wrappers import FullyObsWrapper, ReseedWrapper, RGBImgObsWrapper

"""
  Use envlogger to write trajectories into RLDS dataset format.
  Envlogger: https://github.com/google-deepmind/envlogger
  Some code taken from https://github.com/google-research/rlds/blob/main/rlds/examples/rlds_tfds_envlogger.ipynb
"""

def record_data(env, data_dir, ds_config, num_steps, max_episodes_per_shard, worker_idx, epsilon, split):
  # def step_fn(unused_timestep, unused_action, unused_env):
  #   return {'timestamp_ns': time.time_ns()}

  # def episode_fn(timestep, unused_action, unused_env):
  #   if timestep.first:
  #     return {'image': timestep.observation['image']}
  #   else:
  #     return None

  log_interval = 10

  with envlogger.EnvLogger(
      env,
      backend = tfds_backend_writer.TFDSBackendWriter(
        data_directory=data_dir,
        split_name=split,
        max_episodes_per_file=max_episodes_per_shard,
        ds_config=ds_config),
      ) as env:
    print(f'{worker_idx}: Logging an ε={epsilon} agent for {num_steps} steps...')
    low = env.action_spec().minimum
    high = env.action_spec().maximum + 1
    policy = MazePolicy(epsilon=epsilon, low=low, high=high, seed=worker_idx)
    step_count = 0
    ep_count = 0
    while step_count < num_steps:
      timestep = env.reset()
      policy.reset(timestep.observation)
      step_count += 1
      while not timestep.last():
        action = policy.step(timestep.observation)
        timestep = env.step(action)
        step_count += 1
      ep_count += 1
      if ep_count % log_interval == 0:
        ending = "Terminated" if timestep.discount == 0.0 else "Timed out"
        print(f'Worker {worker_idx}: {ending} episode {ep_count}.  {step_count} / {num_steps} steps')
    ending = "Terminated" if timestep.discount == 0.0 else "Timed out"
    print(f'Worker {worker_idx}: {ending} episode {ep_count}.  {step_count} / {num_steps} steps')

def worker(worker_idx, maze_seeds, epsilon, data_dir, split):
  num_steps = 1000 
  max_episodes_per_shard = 100 

  os.makedirs(data_dir, exist_ok=False)

  env = gym.make("MiniGrid-WFC-MazeSimple-v0", render_mode="rgb_array", tile_size=1, max_steps=100)
  env = ReseedWrapper(env, seeds=maze_seeds)
  # MazePolicy needs to see the full observation
  env = FullyObsWrapper(env)
  # Save RGB observations since RGB is easier to visualize
  env = RGBImgObsWrapper(env, key="rgb_image",tile_size=1)
  # Remove extraneous keys like mission.
  gym_env = FilterObservation(env, ["image", "rgb_image", "direction"])

  # convert to deepmind env.
  env = GymnasiumWrapper(gym_env)
  print(env.observation_space)
  ds_config = tfds.rlds.rlds_base.DatasetConfig(
        name='maze_env',
        observation_info={
          'rgb_image': tfds.features.Tensor(
                     shape=(gym_env.unwrapped.height, gym_env.unwrapped.width, 3),
                     dtype=np.uint8,
                     encoding=tfds.features.Encoding.ZLIB
          ),
          'image': tfds.features.Tensor(
                     shape=(gym_env.unwrapped.height, gym_env.unwrapped.width, 3),
                     dtype=np.uint8,
                     encoding=tfds.features.Encoding.ZLIB
          ),
          'direction':  np.uint8,
        },
        action_info=np.int64,
        reward_info=np.float64,
        discount_info=np.float64,
    )

  record_data(env, data_dir, ds_config, num_steps, max_episodes_per_shard, worker_idx, epsilon, split)

  print(f'Worker {worker_idx}: Finished, checking if last shard needs recovering.')
  builder = tfds.builder_from_directory(data_dir)
  builder = rlds_utils.maybe_recover_last_shard(builder)

_METADATA_FILENAME='features.json'
def get_ds_paths(pattern: str):
  """Returns the paths of tfds datasets under a (set of) directories.

  We assume that a sub-directory with features.json file contains the dataset
  files.

  Args:
    pattern: Root directory to search for dataset paths or a glob that matches
      a set of directories, e.g. /some/path or /some/path/prefix*. See
      tf.io.gfile.glob for the supported patterns.

  Returns:
    A list of paths that contain the environment logs.

  Raises:
    ValueError if the specified pattern matches a non-directory.
  """
  paths = set([])
  for root_dir in tf.io.gfile.glob(pattern):
    if not tf.io.gfile.isdir(root_dir):
      raise ValueError(f'{root_dir} is not a directory.')
    print(f'root: {root_dir}')
    for path, _, files in tf.io.gfile.walk(root_dir):
      if _METADATA_FILENAME in files:
        # print(f'path: {path}')
        paths.add(path)
  print(f'discovered {len(paths)} folders')
  return list(paths)

if __name__ == "__main__":
  GENERATE_DATA = True
  TEST_DATALOADING = False
  num_processes = 100

  maze_range = range(100, 200)
  maze_seeds = [i for i in maze_range]
  root_data_dir = './tensorflow_datasets/maze/'
  split = 'test'

  if GENERATE_DATA:
    if num_processes == 1:
      eps = 0
      data_dir = os.path.join(root_data_dir, f'worker0_e{eps}_maze{maze_range.start}-{maze_range.stop}') # @param
      worker(0, maze_seeds, eps, data_dir, split)
    else:
      epsilon_ranges = []
      for i in range(10):
        epsilon_ranges.extend([(i+1) * 0.1] * 10)
      assert len(epsilon_ranges) == num_processes
      processes = []
      for worker_idx in range(num_processes):
        eps = epsilon_ranges[worker_idx]
        data_dir = os.path.join(root_data_dir, f'worker{worker_idx}_e{eps:.1f}_maze{maze_range.start}-{maze_range.stop}') # @param
        p = Process(target=worker, args=(worker_idx, maze_seeds, eps, data_dir, split))
        p.start()
        processes.append(p)

      for p in processes:
        p.join()

  if TEST_DATALOADING:
    multiple_dataset_path = os.path.join(root_data_dir)
    all_subdirs = get_ds_paths(multiple_dataset_path)
    builder = tfds.builder_from_directories(all_subdirs)
    import ipdb; ipdb.set_trace()
    ds = builder.as_dataset(split='test')
    ds = ds.take(5).cache().repeat(1)
    for i, e in enumerate(ds):
      print(i)
    
