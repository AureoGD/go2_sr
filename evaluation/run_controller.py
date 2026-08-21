import numpy as np
from pathlib import Path

from scripts.sim_tsm import SimTSM
from scripts.sim_fsm import SimFSM

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "controller_data" / "rgc_mpc_data.npz"


def main():
    sim = SimFSM(log=True)
    b0 = [0, 0, 0.3]
    r0 = [np.pi, 0, 0]
    q0 = [-0.3, 1.0, -1.72, -0.5, 1.0, -1.72, 0.5, 1.40, -1.7, -0.5, 0.8, -1.5]
    sim.sim_reset(r0, b0, q0)
    status, data, act_list = sim.run_sim()

    print(status)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, b0=b0, r0=r0, q0=q0, act_list=act_list, **data)
    print(f"status={status}, wrote {OUTPUT}")


if __name__ == "__main__":
    main()
