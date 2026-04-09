import os
import json
import numpy as np
from env.normalizer import StateNormalizer

OUT_DIR = "tpe/data/processed"


def create_normalizer(stats):

    os.makedirs(OUT_DIR, exist_ok=True)

    stats_path = os.path.join(OUT_DIR, "normalization_stats.json")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    q_min = np.array(
        [-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227])

    q_max = np.array(
        [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

    joint_limits = np.stack([q_min, q_max], axis=1)

    torque_limits = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43])

    return StateNormalizer(joint_limits, torque_limits, stats_path)
