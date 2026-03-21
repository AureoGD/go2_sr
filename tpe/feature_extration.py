import os
import json
import numpy as np

RAW_DIR = os.path.join("tpe", "data", "raw")
OUT_DIR = os.path.join("tpe", "data", "processed")

EPS = 1e-8

# --------------------------------------------
# JOINT LIMITS
# --------------------------------------------
q_min = np.array(
    [-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227])

q_max = np.array(
    [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])


# --------------------------------------------
# LOAD EPISODES
# --------------------------------------------
def load_episodes():

    files = sorted([os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".npz")])

    episodes = []

    for f in files:

        data = np.load(f)

        episodes.append({
            "com_vel": data["com_vel"],
            "ang_vel": data["ang_vel"],
            "rpy": data["rpy"],
            "q_pos": data["q_pos"],
            "q_vel": data["q_vel"]
        })

    print(f"Episodes loaded: {len(episodes)}")

    return episodes


# --------------------------------------------
# GLOBAL STATISTICS
# --------------------------------------------
def compute_global_stats(episodes):

    v_all = []
    dq_all = []
    wx_all = []

    for ep in episodes:

        v = ep["com_vel"].reshape(-1, 3)
        dq = ep["q_vel"].reshape(-1, 12)
        w = ep["ang_vel"].reshape(-1, 3)

        v_norm = np.linalg.norm(v, axis=1)
        dq_norm = np.linalg.norm(dq, axis=1)

        v_all.append(v_norm)
        dq_all.append(dq_norm)
        wx_all.append(np.abs(w[:, 0]))

    v_all = np.concatenate(v_all)
    dq_all = np.concatenate(dq_all)
    wx_all = np.concatenate(wx_all)

    stats = {"v_scale": np.std(v_all), "dq_scale": np.std(dq_all), "omega_scale": np.std(wx_all)}

    print("Statistics:", stats)

    return stats


# --------------------------------------------
# COMPUTE ALPHA
# --------------------------------------------
def compute_alpha(rpy):

    rpy = rpy.reshape(-1, 3)

    roll = rpy[:, 0]
    pitch = rpy[:, 1]

    alpha = np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0))
    alpha = 1 - alpha / np.pi

    return alpha


# --------------------------------------------
# NORMALIZE JOINT POSITIONS
# --------------------------------------------
def normalize_q(q):

    q = q.reshape(-1, 12)

    q_norm = 2 * (q - q_min) / (q_max - q_min) - 1

    return q_norm


# --------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------
def extract_features(episodes, stats):

    features = []
    episode_lengths = []

    v_scale = stats["v_scale"]
    dq_scale = stats["dq_scale"]
    omega_scale = stats["omega_scale"]

    for ep in episodes:

        v = ep["com_vel"].reshape(-1, 3)
        w = ep["ang_vel"].reshape(-1, 3)
        dq = ep["q_vel"].reshape(-1, 12)
        q = ep["q_pos"].reshape(-1, 12)
        rpy = ep["rpy"].reshape(-1, 3)

        # --------------------------------
        # ALPHA
        # --------------------------------
        alpha = compute_alpha(rpy)

        # --------------------------------
        # VELOCITY
        # --------------------------------
        v_norm = np.linalg.norm(v, axis=1)
        dir_v = v / (v_norm[:, None] + EPS)
        v_abs = np.tanh(v_norm / (v_scale + EPS))

        # --------------------------------
        # ANGULAR VELOCITY
        # --------------------------------
        wx = w[:, 0]
        wx_norm = np.tanh(wx / (omega_scale + EPS))

        # --------------------------------
        # JOINT VELOCITY
        # --------------------------------
        dq_mag = np.linalg.norm(dq, axis=1)
        dq_norm = np.tanh(dq_mag / (dq_scale + EPS))

        # --------------------------------
        # JOINT POSITIONS
        # --------------------------------
        q_norm = normalize_q(q)

        count = 0

        # --------------------------------
        # BUILD STATE VECTOR
        # --------------------------------
        for i in range(len(alpha)):

            state = np.concatenate([[alpha[i]], dir_v[i], [v_abs[i]], [wx_norm[i]], [dq_norm[i]], q_norm[i]])

            features.append(state)
            count += 1

        episode_lengths.append(count)

    features = np.array(features)
    episode_lengths = np.array(episode_lengths)

    print("Feature shape:", features.shape)
    print("Episode lengths:", episode_lengths.shape)

    return features, episode_lengths


# --------------------------------------------
# SAVE DATASET
# --------------------------------------------
def save_dataset(features, stats, episode_lengths):

    os.makedirs(OUT_DIR, exist_ok=True)

    np.savez(os.path.join(OUT_DIR, "dataset_features.npz"), states=features, episode_lengths=episode_lengths)

    with open(os.path.join(OUT_DIR, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print("Dataset saved.")


# --------------------------------------------
# MAIN
# --------------------------------------------
def main():

    print("Loading episodes...")
    episodes = load_episodes()

    print("Computing statistics...")
    stats = compute_global_stats(episodes)

    print("Extracting features...")
    features, episode_lengths = extract_features(episodes, stats)

    print("Saving dataset...")
    save_dataset(features, stats, episode_lengths)


if __name__ == "__main__":
    main()
