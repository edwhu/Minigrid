import os
import envlogger
from envlogger.backends import rlds_utils
from envlogger.backends import tfds_backend_writer
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time
from typing import List
import gymnasium as gym
from gymnasium.wrappers import FilterObservation 
from dm_env_wrappers import GymnasiumWrapper
from multiprocessing import Process


"""
  Use envlogger to write trajectories into RLDS dataset format.
  Envlogger: https://github.com/google-deepmind/envlogger
  Some code taken from https://github.com/google-research/rlds/blob/main/rlds/examples/rlds_tfds_envlogger.ipynb
  TODO: @Ed use a better policy than random policy. 
"""
_METADATA_FILENAME='features.json'

def record_data(env, data_dir, ds_config, num_episodes, max_episodes_per_shard, worker_idx):
  def step_fn(unused_timestep, unused_action, unused_env):
    return {'timestamp_ns': time.time_ns()}

  with envlogger.EnvLogger(
      env,
      backend = tfds_backend_writer.TFDSBackendWriter(
        data_directory=data_dir,
        split_name='train',
        max_episodes_per_file=max_episodes_per_shard,
        ds_config=ds_config),
      step_fn=step_fn) as env:
    print('Done wrapping environment with EnvironmentLogger.')

    print(f'{worker_idx}: Logging a random agent for {num_episodes} episodes...')
    low = env.action_spec().minimum
    high = env.action_spec().maximum + 1
    for i in range(num_episodes):
      print(worker_idx, f': episode {i}')
      timestep = env.reset()
      while not timestep.last():
        action = np.random.randint(low=low, high=high)
        timestep = env.step(action)
    print(f'{worker_idx}: Done logging a random agent for {num_episodes} episodes.')

def worker(worker_idx):
  generate_data_dir = os.path.join('./tensorflow_datasets/maze/', f'worker_{worker_idx}') # @param
  num_episodes = 125 # @param
  max_episodes_per_shard = 100 # @param
  os.makedirs(generate_data_dir, exist_ok=True)

  env = gym.make("MiniGrid-WFC-MazeSimple-v0")
  env = FilterObservation(env, ["image"])
  print(env.observation_space)
  env = GymnasiumWrapper(env)
  ds_config = tfds.rlds.rlds_base.DatasetConfig(
        name='maze_env',
        observation_info={
          'image': tfds.features.Tensor(
                     shape=(7, 7, 3),
                     dtype=np.uint8,
                     encoding=tfds.features.Encoding.ZLIB
                   )
        },
        action_info=np.int64,
        reward_info=np.float64,
        discount_info=np.float64,
        step_metadata_info={'timestamp_ns': np.int64})

  record_data(env, generate_data_dir, ds_config, num_episodes, max_episodes_per_shard, worker_idx)

  # recover_dataset_path = generate_data_dir # @param
  print(worker_idx, ': Recovering dataset.')
  builder = tfds.builder_from_directory(generate_data_dir)
  builder = rlds_utils.maybe_recover_last_shard(builder)

if __name__ == "__main__":
  GENERATE_DATA = False
  num_processes = 8 

  if GENERATE_DATA:
    if num_processes == 1:
      worker(0)
    else:
      processes = []
      for i in range(num_processes):
        p = Process(target=worker, args=(i,))
        p.start()
        processes.append(p)

      for p in processes:
        p.join()

  multiple_dataset_path = os.path.join('./tensorflow_datasets/maze/')
  all_subdirs = []
  for i in range(num_processes):
    all_subdirs.append(os.path.join(multiple_dataset_path, f'worker_{i}'))
  ds = tfds.builder_from_directories(all_subdirs).as_dataset(split='all')
  ds = ds.take(5).cache().repeat(1)
  for i, e in enumerate(ds):
    print(i)
