import os
import numpy as np

RAW_DIR = "tpe/data/raw"


def load_raw_data():

    files = sorted([os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".npz")])

    episodes = []

    for f in files:
        with np.load(f) as data:
            episodes.append({
                "robot_pos": data["robot_pos"].copy(),
                "robot_vel": data["robot_vel"].copy(),
                "robot_omega": data["robot_omega"].copy(),
                "robot_rpy": data["robot_rpy"].copy(),
                "robot_q": data["robot_q"].copy(),
                "robot_dq": data["robot_dq"].copy(),
                "robot_quat": data["robot_quat"].copy() if "robot_quat" in data else None,
                "controller": data["task_controller_index"].copy()
            })

    print(f"Loaded {len(episodes)} episodes")
    return episodes
