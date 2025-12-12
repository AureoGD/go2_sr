import os
import glob
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from environment.go2_normalize_states import SaveNormalizerCallback

# Import the discrete environment directly
from environment.env_go2 import Go2Env

# --- DQN Hyperparameters ---
# Number of parallel environments
N_ENVS = 1
# Size of the replay buffer
BUFFER_SIZE = 1_000_000
# Number of steps to collect before starting training
LEARNING_STARTS = 50_000
# Batch size for each training update
BATCH_SIZE = 128
# The soft update coefficient for the target network
TAU = 0.005
# Update the model every TRAIN_FREQ steps or episodes
TRAIN_FREQ = 2
# How many gradient steps to do after each update
GRADIENT_STEPS = 1
# Exploration fraction
EXPLORATION_FRACTION = 0.1
# Final value of epsilon
EXPLORATION_FINAL_EPS = 0.05

# --- Training Control ---
# Frequency to save intermediate models (in steps)
CHECKPOINT_FREQ = 10_000
# Frequency to run evaluation
EVAL_FREQ = 10_000
# Total timesteps for training
TOTAL_TIMESTEPS = 100_000_000

# Option to continue training from the latest checkpoint
CONTINUE_TRAINING = False
# Root folder for saved models
MODEL_ROOT = "models/dqn"


def get_base_env(env):
    """Helper function to extract base environment from wrappers"""
    while hasattr(env, 'env'):
        env = env.env
    return env


def main(args):
    continue_training = CONTINUE_TRAINING
    # Set prefix for discrete environment models
    prefix = 'D_'

    tensorboard_log = "logs/dqn"
    run_dir = None
    checkpoint_path = None

    # --- 0. Handle Continue Training ---
    if continue_training:
        # Search for the latest checkpoint inside the model folder
        checkpoints = glob.glob(os.path.join(MODEL_ROOT, prefix + "*/rl_model_*.zip"))
        if checkpoints:
            checkpoints.sort()
            checkpoint_path = checkpoints[-1]
            run_dir = os.path.dirname(checkpoint_path)
            run_name = os.path.basename(run_dir)
            print(f"[INFO] Continuing training from: {checkpoint_path}")
        else:
            print("[INFO] No checkpoints found. Starting a new training run.")
            continue_training = False

    if not continue_training:
        # Get the time string for a new run
        time_str = datetime.now().strftime("%d%H%M%S")

        # Combine them in the new order
        run_name = prefix + time_str
        run_dir = os.path.join(MODEL_ROOT, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run directory: {run_dir}")

    train_env = make_vec_env(Go2Env, n_envs=N_ENVS, seed=0)
    eval_env = make_vec_env(lambda: Go2Env(), n_envs=1, seed=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    # Save a model every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=run_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # Save normalizer weights with the same frequency as checkpoints
    normalizer_callback = SaveNormalizerCallback(save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
                                                 save_path=run_dir,
                                                 name_prefix="normalizer_weights",
                                                 verbose=1)

    # --- 4. Create or Load the Model ---
    if continue_training and checkpoint_path:
        # Load the last checkpoint and attach the new environment
        print(f"Loading model from {checkpoint_path}...")
        model = DQN.load(checkpoint_path, env=train_env)

        # Load the replay buffer if it exists
        replay_buffer_path = os.path.join(run_dir, "rl_model_replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            print(f"Loading replay buffer from {replay_buffer_path}...")
            model.load_replay_buffer(replay_buffer_path)

        # Try to load the latest normalizer weights with robust approach
        normalizer_files = glob.glob(os.path.join(run_dir, "normalizer_weights_*.npz"))
        if normalizer_files:
            normalizer_files.sort()
            latest_normalizer = normalizer_files[-1]
            print(f"Loading normalizer weights from: {latest_normalizer}")

            # Get the base environment using robust approach
            base_env = get_base_env(train_env)

            # Handle different environment structures
            if hasattr(base_env, 'load_normalizer'):
                # Single environment
                success = base_env.load_normalizer(latest_normalizer)
                if success:
                    print("Successfully loaded normalizer weights into base environment")
                else:
                    print("Failed to load normalizer weights into base environment")
            elif hasattr(base_env, 'envs') and len(base_env.envs) > 0:
                # Vectorized environment
                first_env = base_env.envs[0]
                if hasattr(first_env, 'load_normalizer'):
                    success = first_env.load_normalizer(latest_normalizer)
                    if success:
                        print("Successfully loaded normalizer weights into first sub-environment")
                    else:
                        print("Failed to load normalizer weights into first sub-environment")
            else:
                print("No suitable environment found for loading normalizer weights")
        else:
            print("No normalizer weight files found to load")
    else:
        print("Creating new model...")
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=1,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=0.99,
            learning_rate=1e-4,
            train_freq=(TRAIN_FREQ, "step"),
            gradient_steps=GRADIENT_STEPS,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            target_update_interval=1000,
        )

    print("\nStarting model training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[checkpoint_callback, normalizer_callback],
                tb_log_name=run_name,
                reset_num_timesteps=not continue_training)

    # --- 5. Final Save ---
    print("Training finished. Saving final model.")
    model.save(os.path.join(run_dir, "final_model"))
    model.save_replay_buffer(os.path.join(run_dir, "final_replay_buffer"))

    # Save final normalizer weights with robust approach
    base_env = get_base_env(train_env)
    final_normalizer_saved = False

    if hasattr(base_env, 'save_normalizer'):
        # Single environment
        final_normalizer_path = os.path.join(run_dir, "final_normalizer_weights.npz")
        success = base_env.save_normalizer(final_normalizer_path)
        if success:
            print(f"Saved final normalizer weights to: {final_normalizer_path}")
            final_normalizer_saved = True
        else:
            print("Failed to save final normalizer weights to base environment")
    elif hasattr(base_env, 'envs') and len(base_env.envs) > 0:
        # Vectorized environment
        first_env = base_env.envs[0]
        if hasattr(first_env, 'save_normalizer'):
            final_normalizer_path = os.path.join(run_dir, "final_normalizer_weights.npz")
            success = first_env.save_normalizer(final_normalizer_path)
            if success:
                print(f"Saved final normalizer weights to: {final_normalizer_path}")
                final_normalizer_saved = True
            else:
                print("Failed to save final normalizer weights to first sub-environment")

    if not final_normalizer_saved:
        print("Warning: Could not save final normalizer weights - no suitable environment found")

    # --- 6. Updated Training Summary ---
    print("\n--- Training Summary ---")
    print(f"Run directory: {run_dir}")
    print(f"Saved final model to: {os.path.join(run_dir, 'final_model.zip')}")
    if final_normalizer_saved:
        print(f"Saved final normalizer to: {os.path.join(run_dir, 'final_normalizer_weights.npz')}")
    else:
        print("Final normalizer weights were not saved")
    print("----------------------")


if __name__ == "__main__":
    main(None)
