import os
import torch
import numpy as np
import time
from es_framework.commons.control_rule import ControlRule
from environment.env_go2 import Go2Env

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to your saved model folder
MODEL_DIR = "models/cem/D_03100944"

# Filenames (ensure these exist in the folder)
MODEL_FILE = "overall_best_model.pth"
NORM_FILE = "cem_model_final_mean_normalizer.npz"
# OR use the overall best if you prefer:
# MODEL_FILE = "overall_best_model.pth"
# NORM_FILE = "overall_best_model_normalizer.npz"

# Network Hyperparameters (Must match training!)
FC1_DIM = 128
FC2_DIM = 128
IS_DISCRETE = True

# Simulation Settings
DIFFICULTY = 1.0  # Test at full difficulty
EPISODE_LENGTH = 10000
RENDER = True
# ==============================================================================


def load_normalization_stats(env, filepath):
    """
    Loads mean/var/count from .npz and injects them into the 
    environment's velocity normalizer.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Normalizer file not found: {filepath}")

    data = np.load(filepath)

    # Access the specific normalizer used for policy input (the "Actor")
    normalizer = env.normalizer.vel_normalizer

    # Inject stats
    normalizer.mean = data['mean']
    normalizer.var = data['var']
    normalizer.count = data['count']

    print(f"[-] Loaded Normalization Stats from {filepath}")
    print(f"    Count: {float(normalizer.count):.2f}")
    print(f"    Mean[0]: {normalizer.mean[0]:.4f} ...")


def main():
    # 1. Initialize Environment
    # We use ID 0 and rendering=True to see the simulation
    env = Go2Env(env_id=0, rendering=RENDER, max_step=EPISODE_LENGTH)

    # Set the difficulty (usually we want to evaluate the final difficulty)
    env.set_difficulty(DIFFICULTY)

    # 2. Initialize Model Architecture
    obs_dim = env.observation_space.shape[0]

    if IS_DISCRETE:
        out_dim = env.action_space.n
    else:
        out_dim = env.action_space.shape[0]

    policy = ControlRule(observation_dim=obs_dim,
                         output_dim=out_dim,
                         fc1_dim=FC1_DIM,
                         fc2_dim=FC2_DIM,
                         discrete=IS_DISCRETE)

    # 3. Load Model Weights
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, weights_only=False)
    policy.load_state_dict(state_dict)
    policy.eval()  # Set to evaluation mode
    print(f"[-] Loaded Model Weights from {model_path}")

    # 4. Load Normalization Statistics
    # CRITICAL: If you skip this, the robot will fail immediately because
    # the policy expects normalized inputs.
    norm_path = os.path.join(MODEL_DIR, NORM_FILE)
    load_normalization_stats(env, norm_path)

    print("\n=== Starting Evaluation Loop (Press Ctrl+C to stop) ===")

    try:

        for i in range(5):
            # Reset Env
            obs, info = env.reset()
            total_reward = 0
            tick = 0

            for step in range(EPISODE_LENGTH):
                # Predict Action
                # with torch.no_grad():
                #     action, _ = policy.predict(obs)
                if tick < 100:
                    action = 0
                elif tick >= 100 and tick < 300:
                    action = 1
                elif tick >= 300 and tick < 500:
                    action = 2
                elif tick >= 500 and tick < 7000:
                    action = 3
                else:
                    action = 4
                # Step Environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    print(f"Episode finished. Total Reward: {total_reward:.4f} | Steps: {step}")
                    break
                tick += 1

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
