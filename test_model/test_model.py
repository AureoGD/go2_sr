import os
import torch
import numpy as np
import time
from es_framework.commons.control_rule import ControlRule
from environment.env_go2 import Go2Env

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# CONTROL MODE: True = Use Neural Network | False = Use Hardcoded If/Else
USE_TRAINED_POLICY = True

# Path to your saved model folder
MODEL_DIR = "models/cem/D_cem_20251222_011002"
MODEL_FILE = "model_best_overall.pth"

# Network Hyperparameters (Must match training!)
FC1_DIM = 128
FC2_DIM = 128
IS_DISCRETE = True

# Simulation Settings
DIFFICULTY = 1.0  # Test at full difficulty
EPISODE_LENGTH = 1500
RENDER = True  # Set to True to verify behavior visually
# ==============================================================================


def main():
    # 1. Initialize Environment
    # We use ID 0 and rendering=True to see the simulation
    env = Go2Env(env_id=0, rendering=RENDER, max_step=EPISODE_LENGTH)
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

    # 3. Load Weights (Only if using Policy)
    if USE_TRAINED_POLICY:
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        print(f"[-] Loading checkpoint from {model_path}...")

        # Load the dictionary
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # A. Load Neural Network Weights
        if 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
            print("    [OK] Model weights loaded.")
        else:
            policy.load_state_dict(checkpoint)  # Legacy fallback
            print("    [Warning] Loaded raw state dict.")

        policy.eval()  # Set to evaluation mode

        # B. Load Normalization Statistics
        if 'normalizer_state' in checkpoint:
            norm_data = checkpoint['normalizer_state']

            # FIX: Convert dictionary to Object if necessary
            if isinstance(norm_data, dict):
                from types import SimpleNamespace
                # This creates an object where obj.key can be accessed as obj.key
                norm_stats = SimpleNamespace(**norm_data)
            else:
                norm_stats = norm_data

            env.normalizer.sync_global_stats(norm_stats)

            print("    [OK] Normalizer stats loaded.")
            print(f"         Count: {norm_stats.count:.1f}")
        else:
            print("    [CRITICAL WARNING] 'normalizer_state' key missing!")

        print("\n=== Starting Evaluation Loop (Press Ctrl+C to stop) ===")

    try:
        for i in range(5):
            print(f"\n--- Episode {i+1} ---")

            # Reset Env
            obs, info = env.reset()
            total_reward = 0
            tick = 0

            # Start Timer
            start_time = time.perf_counter()

            for step in range(EPISODE_LENGTH):

                # --- SELECT ACTION ---
                if USE_TRAINED_POLICY:
                    with torch.no_grad():
                        action, _ = policy.predict(obs)
                else:
                    # Hardcoded Logic
                    if tick < 120:
                        action = 0
                    elif tick >= 120 and tick < 350:
                        action = 1
                    elif tick >= 350 and tick < 500:
                        action = 2
                    elif tick >= 500 and tick < 1100:
                        action = 3
                    elif tick >= 1100 and tick < 1500:
                        action = 4
                    else:
                        action = -1

                # --- STEP ENVIRONMENT ---
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Render delay if needed
                # if RENDER: time.sleep(0.002)

                if terminated or truncated:
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time

                    reason = "Terminated" if terminated else "Truncated"
                    print(f"[{reason}] Reward: {total_reward:.4f} | Steps: {step+1} | Time: {elapsed_time:.4f}s")
                    break

                tick += 1

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
