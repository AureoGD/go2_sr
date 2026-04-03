from sim.go2_sim import Go2ModelSimMuJoCo
from tpe.phases import Phase
from tpe.model import TPE

import numpy as np
import random
import torch
from collections import deque
import json

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

q0_base = np.array([0.7, 1.0, -2.1, 0, 1.4, -1.75, 0, 1.4, -1.8, 0, 1.4, -2.0])

render_sim = True
model_path = "tpe_model.pt"

WINDOW = 15
INPUT_DIM = 19

EPS = 1e-8

# -------------------------------------------------
# LOAD NORMALIZATION PARAMETERS
# -------------------------------------------------

with open("tpe/data/processed/normalization_stats.json") as f:
    stats = json.load(f)

v_scale = stats["v_scale"]
omega_scale = stats["omega_scale"]
dq_scale = stats["dq_scale"]

# -------------------------------------------------
# JOINT LIMITS
# -------------------------------------------------

q_min = np.array(
    [-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227])

q_max = np.array(
    [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------


def compute_alpha(rpy):

    roll = rpy[0]
    pitch = rpy[1]

    alpha = np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1))
    alpha = 1 - alpha / np.pi

    return alpha


def compute_features(state):

    v = state.r_vel.flatten()
    w = state.omega.flatten()
    rpy = state.rpy.flatten()
    q = state.q.flatten()
    dq = state.dq.flatten()

    # ----------------------------
    # ALPHA
    # ----------------------------

    alpha = compute_alpha(rpy)

    # ----------------------------
    # VELOCITY
    # ----------------------------

    v_norm = np.linalg.norm(v)

    dir_v = v / (v_norm + EPS)

    v_abs = np.tanh(v_norm / (v_scale + EPS))

    # ----------------------------
    # ANGULAR VELOCITY
    # ----------------------------

    wx = w[0]

    wx_norm = np.tanh(wx / (omega_scale + EPS))

    # ----------------------------
    # JOINT VELOCITY
    # ----------------------------

    dq_mag = np.linalg.norm(dq)

    dq_norm = np.tanh(dq_mag / (dq_scale + EPS))

    # ----------------------------
    # JOINT POSITION NORMALIZATION
    # ----------------------------

    q_norm = 2 * (q - q_min) / (q_max - q_min) - 1

    # ----------------------------
    # FEATURE VECTOR
    # ----------------------------

    features = np.concatenate([[alpha], dir_v, [v_abs], [wx_norm], [dq_norm], q_norm])

    return features, alpha, wx_norm, v_abs


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------


def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TPE(INPUT_DIM, WINDOW, num_classes=4)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    return model, device


# -------------------------------------------------
# SCHEDULER
# -------------------------------------------------


class TimeSchedulling():

    def __init__(self):

        self.tick = 0

        self.base_durations = np.array([25, 110, 310, 360, 260, 210, 210])
        self.delta = np.array([15, 20, 50, 30, 30, 30, 30])

        self.gen_new_ticks()

    def gen_new_ticks(self):

        self.tick = 0

        self.current_durations = (self.base_durations + np.random.uniform(-self.delta, self.delta))

        self.current_durations = np.maximum(self.current_durations, 1)
        self.current_durations = np.round(self.current_durations).astype(int)

        self.current_tick = np.cumsum(self.current_durations)

        self.direction = np.random.choice(["cw", "ccw"])

    def _map_action(self, logical_idx):

        if self.direction == "cw":

            mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

        else:

            mapping = {0: 0, 1: 1, 2: 7, 3: 8, 4: 9, 5: 10, 6: 6}

        return mapping[logical_idx]

    def action(self):

        self.tick += 1

        idx = np.searchsorted(self.current_tick, self.tick)

        if idx < len(self.current_tick):
            return self._map_action(idx)

        return 0


# -------------------------------------------------
# RANDOM START
# -------------------------------------------------


def random_start():

    noise_q0 = np.random.uniform(-0.5, 0.5, (12, 1))

    q0 = q0_base.reshape(12, 1) + noise_q0

    yaw = random.uniform(-np.pi, np.pi)

    rpy = np.array([np.pi, 0, yaw])

    x0 = random.uniform(-1, 2.75)

    b0 = np.array([x0, 0, 0.3])

    return q0, rpy, b0


# -------------------------------------------------
# MAIN
# -------------------------------------------------


def main():

    model, device = load_model()

    robot_sim = Go2ModelSimMuJoCo(render=render_sim, strategy_name="tb")

    scheduler = TimeSchedulling()

    window = deque(maxlen=WINDOW)

    q0, rpy, b0 = random_start()

    robot_sim.reset_robot_pose(b0=b0, q0=q0, r0=rpy)

    scheduler.gen_new_ticks()

    total_duration = scheduler.current_tick[-1]

    print("\nRunning simulation with TPE\n")

    log_data = {"alpha": [], "omega_x": [], "v_abs": [], "controller": [], "tpe_phase": []}

    for _ in range(total_duration):

        action = scheduler.action()

        robot_sim.control_loop(mode=action)

        state = robot_sim.robot_states

        features, alpha, wx_norm, v_abs = compute_features(state)

        window.append(features)

        if len(window) < WINDOW:
            continue

        x = np.array(window)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():

            logits = model(x)

            phase_idx = torch.argmax(logits, dim=1).item()

        phase = Phase(phase_idx)

        print(f"controller={action:2d}   TPE={phase.name}")

        log_data["alpha"].append(alpha)
        log_data["omega_x"].append(wx_norm)
        log_data["v_abs"].append(v_abs)
        log_data["controller"].append(action)
        log_data["tpe_phase"].append(phase_idx)

    np.savez("tpe_simulation_log.npz",
             alpha=np.array(log_data["alpha"]),
             omega_x=np.array(log_data["omega_x"]),
             v_abs=np.array(log_data["v_abs"]),
             controller=np.array(log_data["controller"]),
             tpe_phase=np.array(log_data["tpe_phase"]))

    print("\nSimulation log saved: tpe_simulation_log.npz")


if __name__ == "__main__":
    main()
