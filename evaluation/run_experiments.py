import json
import numpy as np
from pathlib import Path

from scripts.sim_tsm import SimTSM
from scripts.sim_fsm import SimFSM
from scripts.sim_default import SimDef

SCRIPT_DIR = Path(__file__).resolve().parent

IC_FILE = SCRIPT_DIR / "ics" / "ics_v1.json"
OUTPUT = SCRIPT_DIR / "controller_data" / "unitree_validation_data.npz"

N_SLOP = 6
SLOP_ANG_INC = 5
SLOP_LENGTH = 3
SLOP_WIDTH = 3

# -----------------------------------------------------------------------------
# Load initial conditions
# -----------------------------------------------------------------------------

with open(IC_FILE, "r") as f:
    data = json.load(f)

metadata = data["metadata"]
ics = data["ics"]

n_bins = metadata["n_bins"]
n_per_bin = metadata["n_per_bin"]

# -----------------------------------------------------------------------------
# Allocate results
# success[slope, yaw_bin, yaw_bin_number]
# -----------------------------------------------------------------------------

success = np.zeros((N_SLOP, n_bins, n_per_bin), dtype=bool)

# -----------------------------------------------------------------------------
# Select controller
# -----------------------------------------------------------------------------

# sim = SimFSM()
# sim = SimTSM()
sim = SimDef()

# -----------------------------------------------------------------------------
# Run experiments
# -----------------------------------------------------------------------------

for i_slope in range(N_SLOP):

    slope_angle = i_slope * SLOP_ANG_INC

    z = -0.05 + SLOP_WIDTH * np.sin(np.deg2rad(slope_angle)) / 2
    b0 = [0, SLOP_LENGTH * i_slope, 0.3 + z]

    for ic in ics:

        yaw = ic["yaw"]
        q0 = ic["q0"]

        r0 = [np.pi, 0.0, yaw]

        sim.sim_reset(
            r0=r0,
            b0=b0,
            q0=q0,
        )

        accomplish, _, _ = sim.run_sim()

        success[
            i_slope,
            ic["yaw_bin"],
            ic["yaw_bin_number"],
        ] = accomplish

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------

np.savez(
    OUTPUT,
    success=success,
    slope_angles=np.arange(N_SLOP) * SLOP_ANG_INC,
)

print(f"\nValidation data saved to:\n{OUTPUT}")
