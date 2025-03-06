import gymnasium as gym
from minigrid.wrappers import ActiveVisionWrapper
import numpy as np

def main():
    # Create the base environment
    env = gym.make("MiniGrid-PutNear-8x8-N3-v0", render_mode="human")

    # Wrap it with the active vision wrapper
    env = ActiveVisionWrapper(env, camera_size=2)

    # Reset the environment
    obs, info = env.reset()
    print("Initial observation shape:", obs['image'].shape)
    print("Camera position:", obs['camera_pos'])

    # Run a few steps with sampled actions
    print("\n--- Using action_space.sample() ---")
    for i in range(5):
        # Sample from the dictionary action space
        action = env.action_space.sample()

        # Print the sampled action
        print(f"Step {i+1}: Sampled action: Sensory={action['sensory']}, Motor={action['motor']}")

        # Step the environment with the sampled action
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        print(f"Camera position: {obs['camera_pos']}")

    # Run a few steps with manually created actions
    print("\n--- Using manually created actions ---")
    for i in range(5):
        # Create a custom action dictionary
        # For sensory: 0=up, 1=down, 2=left, 3=right
        # For motor: depends on the environment (typically 0-6 for MiniGrid)
        sensory = i % 4  # Cycle through camera directions
        motor = np.random.randint(0, 7)  # Random agent action

        action = {
            'sensory': sensory,
            'motor': motor
        }

        print(f"Step {i+1}: Manual action: Sensory={action['sensory']}, Motor={action['motor']}")

        # Step the environment with the manual action
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        print(f"Camera position: {obs['camera_pos']}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()