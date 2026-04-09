import numpy as np


def compute_stats(episodes):

    v_all, dq_all, omega_all = [], [], []

    for ep in episodes:
        v_all.append(np.linalg.norm(ep["robot_vel"], axis=1))
        dq_all.append(np.linalg.norm(ep["robot_dq"], axis=1))
        omega_all.append(np.linalg.norm(ep["robot_omega"], axis=1))

    stats = {
        "v_scale": float(np.std(np.concatenate(v_all))),
        "dq_scale": float(np.std(np.concatenate(dq_all))),
        "omega_scale": float(np.std(np.concatenate(omega_all)))
    }

    print("\n Normalization stats:")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    return stats
