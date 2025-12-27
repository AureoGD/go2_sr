import os
import json
import torch
import numpy as np
import time

from environment.go2_env import Go2Env
from environment.normalizer import NormalizerStats
from es_framework.components.policy import Policy

USE_TRAINED_POLICY = False

RESULTS_DIR = "results/go2_self_righting_CEM_20251225_091824"
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
MODEL_FILE = "gen_0140.pth"
CONFIG_FILE = os.path.join(RESULTS_DIR, "config.json")

DIFFICULTY = 1.0
EPISODE_LENGTH = 1500
RENDER = True


def main():
    env = Go2Env(env_id=0, rendering=RENDER, max_step=EPISODE_LENGTH)
    env.set_difficulty(DIFFICULTY)

    if USE_TRAINED_POLICY:
        obs_dim = env.observation_space.shape[0]
        out_dim = env.action_space.n

        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"config.json not found: {CONFIG_FILE}")

        with open(CONFIG_FILE, "r") as f:
            train_config = json.load(f)

        model_cfg = train_config.get("model_config", None)
        if model_cfg is None:
            raise RuntimeError("model_config missing from config.json")

        policy = Policy(observation_dim=obs_dim, output_dim=out_dim, **model_cfg)

        params = list(policy.parameters())
        assert len(params) > 0, "Policy has no parameters!"
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        print(f"[-] Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            policy.load_state_dict(checkpoint)

        policy.eval()

        if "normalizer_state" in checkpoint:
            norm_data = checkpoint["normalizer_state"]

            if isinstance(norm_data, dict):
                mean = norm_data.get("mean", None)
                var = norm_data.get("var", None)

                if mean is None or var is None:
                    raise RuntimeError(f"Invalid normalizer_state keys: {list(norm_data.keys())}")

                count = norm_data.get("count", 1.0)

                norm_stats = NormalizerStats(
                    count=float(count),
                    mean=np.asarray(mean, dtype=np.float64),
                    var=np.asarray(var, dtype=np.float64),
                )
            else:
                norm_stats = norm_data

            env.normalizer.sync_global_stats(norm_stats)
            print("[OK] Normalizer loaded.")
        else:
            print("[WARNING] No normalizer stats found in checkpoint.")

        print("\n=== Starting Evaluation Loop (Ctrl+C to stop) ===")

    try:
        for ep in range(5):
            print(f"\n--- Episode {ep + 1} ---")
            r0 = [np.pi, 0, 0.25]
            obs, info = env.reset(r0=r0)
            total_reward = 0.0
            tick = 0

            start_time = time.perf_counter()

            for step in range(EPISODE_LENGTH):

                if USE_TRAINED_POLICY:
                    with torch.no_grad():
                        action, _ = policy.predict(obs, deterministic=True)
                        action = int(action)
                else:
                    if tick < 120:
                        action = 1
                    elif tick < 350:
                        action = 2
                    elif tick < 1000:
                        action = 3
                    # elif tick < 620:
                    #     action = 4
                    # elif tick < 920:
                    #     action = 5
                    # elif tick < 1200:
                    #     action = 6
                    else:
                        action = 0

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    elapsed = time.perf_counter() - start_time
                    status = ("SUCCESS" if info.get("is_success", False) else
                              ("TERMINATED" if terminated else "TRUNCATED"))

                    print(f"[{status}] "
                          f"Reward: {total_reward:.3f} | "
                          f"Steps: {step + 1} | "
                          f"Time: {elapsed:.3f}s")
                    break

                tick += 1

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
