import json
import numpy as np
from datetime import datetime
from pathlib import Path
from evaluation.self_colision_checker import SelfCollisionChecker

# --- configuration ---
SEED = 42
N_PER_BIN = 20
N_BINS = 24
JOINT_NOISE_STD = np.array([0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.5])
Q0_NOMINAL = np.array([-0.3, 1.0, -1.72, -0.5, 1.0, -1.72, 0.5, 1.40, -1.7, -0.5, 0.8, -1.5])
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "ics" / "ics_v1.json"
Q_MIN = np.array(
    [-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227])
Q_MAX = np.array(
    [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

PATH = "sim/assets/unitree_go2/scene.xml"

scc = SelfCollisionChecker(PATH)


def is_valid(q):
    """Joint limits + any collision/feasibility check you want."""
    return np.all(q >= Q_MIN) and np.all(q <= Q_MAX) and not scc.is_self_colliding(q)


def main():
    rng = np.random.default_rng(SEED)
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)

    ics = []
    ic_id = 0
    for b in range(N_BINS):
        for a in range(N_PER_BIN):
            yaw = rng.uniform(edges[b], edges[b + 1])
            while True:  # reject-and-resample
                q = Q0_NOMINAL + rng.normal(0, JOINT_NOISE_STD, Q0_NOMINAL.shape)
                if is_valid(q):
                    break
            ics.append({"ic_id": ic_id, "yaw_bin": b, "yaw_bin_number": a, "yaw": float(yaw), "q0": q.tolist()})
            ic_id += 1

    data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "seed": SEED,
            "n_per_bin": N_PER_BIN,
            "n_bins": N_BINS,
            "bin_edges_rad": edges.tolist(),
            "q0_nominal": Q0_NOMINAL.tolist(),
            "joint_noise_std_rad": JOINT_NOISE_STD.tolist(),
        },
        "ics": ics,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(ics)} ICs to {OUTPUT}")


if __name__ == "__main__":
    main()
