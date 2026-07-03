import json
import numpy as np
from pathlib import Path

from scripts.sim_tsm import SimTSM
from scripts.sim_fsm import SimFSM

IC_FILE = Path(__file__).resolve().parent / "ics" / "ics_v1.json"

with open(IC_FILE) as f:
    data = json.load(f)

yaws = np.array([ic["yaw"] for ic in data["ics"]])
joints = np.array([ic["q0"] for ic in data["ics"]])

sim = SimFSM()

for i in range(len(yaws)):
    yaw = yaws[i]
    b0 = [2.5, 0, 0.3]
    r0 = [np.pi, 0, yaw]
    q0 = joints[i].tolist()
    print(f"int: {i}, yaw: {yaw}")
    sim.sim_reset(r0, b0, q0)
    sim.run_sim()
