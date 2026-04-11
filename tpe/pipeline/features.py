from tqdm import tqdm
import numpy as np


def build_features(episodes, normalizer):

    features = []
    episode_lengths = []

    eps = 1e-8

    for ep in tqdm(episodes, desc="Building features"):

        if ep["robot_quat"] is None:
            raise ValueError("Quaternion required!")

        N = len(ep["robot_pos"])
        count = 0

        for i in range(N):

            pos = ep["robot_pos"][i]
            vel = ep["robot_vel"][i]
            omega = ep["robot_omega"][i]
            q = ep["robot_q"][i]
            dq = ep["robot_dq"][i]
            epsilon = ep["robot_quat"][i]

            alpha = normalizer.compute_alpha(epsilon)

            pos_norm = normalizer.normalize_position(pos)

            dir_v, v_abs = normalizer.normalize_velocity(vel)

            dir_omega, omega_abs = normalizer.normalize_omega(omega)

            q_norm = normalizer.normalize_q(q)
            dq_norm = normalizer.compute_dq_norm(dq)

            feat = np.concatenate([[alpha], dir_v, [v_abs], dir_omega, [omega_abs], q_norm, [dq_norm]])

            features.append(feat)
            count += 1

        episode_lengths.append(count)

    features = np.array(features, dtype=np.float32)
    episode_lengths = np.array(episode_lengths)

    print("\n Features:", features.shape)

    return features, episode_lengths
