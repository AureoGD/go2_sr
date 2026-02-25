import os
import json
import torch
import numpy as np
import time

from environment.go2_env import Go2Env
from environment.normalizer import NormalizerStats
from es_framework.components.policy import Policy

from environment.phase_library import PhaseLibrary
from environment.phase_spawner import PhaseSpawner

from collections import deque
import pickle

USE_TRAINED_POLICY = False

RESULTS_DIR = "results/go2_self_righting_CEM_20260109_175836"
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
MODEL_FILE = "gen_0600.pth"
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

        # if "normalizer_state" in checkpoint:
        #     norm_data = checkpoint["normalizer_state"]

        #     if isinstance(norm_data, dict):
        #         mean = norm_data.get("mean", None)
        #         var = norm_data.get("var", None)

        #         if mean is None or var is None:
        #             raise RuntimeError(f"Invalid normalizer_state keys: {list(norm_data.keys())}")

        #         count = norm_data.get("count", 1.0)

        #         norm_stats = NormalizerStats(
        #             count=float(count),
        #             mean=np.asarray(mean, dtype=np.float64),
        #             var=np.asarray(var, dtype=np.float64),
        #         )
        #     else:
        #         norm_stats = norm_data

        #     env.normalizer.sync_global_stats(norm_stats)
        #     print("[OK] Normalizer loaded.")
        # else:
        #     print("[WARNING] No normalizer stats found in checkpoint.")

        # print("\n=== Starting Evaluation Loop (Ctrl+C to stop) ===")

    phase_lib = PhaseLibrary()
    phase_spawner = PhaseSpawner(phase_lib)

    scenarios = [phase_spawner.apply() for _ in range(10)]
    try:
        ep = 0
        for s in scenarios:
            print(f"\n--- Episode {ep + 1} ---")
            # q0, r0, b0, mode = s
            # obs, info = env.reset(b0=b0, r0=r0, q0=q0, mode=mode)

            r0 = [np.pi, 0, 0]
            q0 = [0.7, 1.0, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]
            obs, info = env.reset(r0=r0, q0=q0)

            total_reward = 0.0
            tick = 0

            start_time = time.perf_counter()

            for step in range(1500):

                if USE_TRAINED_POLICY:
                    with torch.no_grad():
                        action, _ = policy.predict(obs, deterministic=True)
                        action = int(action)
                else:
                    action = debug_tb(tick=tick)
                    # action = debg_rgc(tick=tick)
                    # action = 6

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


def debug_tb(tick):
    if tick < 25:
        action = 0
    elif tick < 25 + 120:
        action = 1
    elif tick < 25 + 120 + 310:
        action = 2
    elif tick < 25 + 120 + 310 + 370:
        action = 3
    elif tick < 25 + 120 + 310 + 370 + 300:
        action = 4
    elif tick < 25 + 120 + 310 + 370 + 300 + 210:
        action = 5
    elif tick < 25 + 120 + 310 + 370 + 300 + 210 + 210:
        action = 6
    else:
        action = 0

    return action


def debg_rgc(tick):

    if tick < 20:
        action = 0
    elif tick < 150:
        action = 1  # go_safe
    elif tick < 150 + 250:
        action = 2  # prepare_cw
    elif tick < 150 + 250 + 150:
        action = 3  # roll_cw
    elif tick < 150 + 250 + 150 + 250:
        action = 4  # landing_cw
    elif tick < 150 + 250 + 150 + 250 + 270:
        action = 5  # prone
    elif tick < 150 + 250 + 150 + 250 + 270 + 300:
        action = 6  # standing_up
    else:
        action = 0

    return action


if __name__ == "__main__":
    main()
