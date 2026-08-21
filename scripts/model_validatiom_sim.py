import argparse

import numpy as np
import mujoco
import mujoco.viewer

from pathlib import Path

from sim.go2_sim import Go2Sim
from sim.engine.pinocchio_engine import PinocchioEngine
import pinocchio as pin

from control.validation.model_validation import ModelValidation

np.set_printoptions(linewidth=250)

np.set_printoptions(precision=3)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        type=str,
                        default="class_validation",
                        help="output name (saved to evaluation/results/<out>.npz)")
    args = parser.parse_args()

    root_joint = pin.JointModelFreeFlyer()
    pin_model = pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", root_joint)
    pin_engine = PinocchioEngine(pin_model)

    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    conf_controller = {"pin_engine": pin_engine}

    control = ModelValidation(**conf_controller)

    sim = Go2Sim(mj_model=model, mj_data=data, controller=control, pin_engine=pin_engine, viewer=viewer)

    b0 = [0, 0, 0.55]
    r0 = [np.pi, 0, 0]
    q0 = [-0.9, 0, -2.8, 0, 1.26, -2.8, -0.5, 0, -2.8, -1.025, 4.15, -2.2]
    # q0 = [0.0, 1.4, -2.7, 0.5, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7]
    # q0 = [0.0, 2.4, -2.7, 0.5, 1.4, -2.7, 0.0, 2.4, -2.7, 0.0, 1.4, -2.7]
    # q0 = np.array([-0.5, 0, -2.8, 0.5, 1.26, -2.8, -0.5, 0, -2.8, 0.9, 3.75, -1.5])

    # dqe, dq = [], []
    dqe, dq = [], []
    sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)
    for tick in range(150):
        action = 0 if tick < 100 else 2
        sim.simulation_loop(action)

        dq.append(np.asarray(sim.dg.dq, np.float64).ravel())
        dqe.append(np.asarray(sim.dg.dqe, np.float64).ravel())

    out = Path("evaluation/results") / f"{args.out}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        dq=np.array(dq),  # shape (n_runs, 600, 12)
        dqe=np.array(dqe),  # shape (n_runs, 600, 12)
        tick=np.arange(150),  # shared time axis
    )


#     KP = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
#     Percents = [0.01, 0.02, 0.05, 0.1]
#     DAMPING = [0.0, 2.0, 3.0, 4.0]

#     # accumulators
#     Q, QR, QVEL = [], [], []
#     KPa, KDa, PCTa, DMPa = [], [], [], []

#     for damping in DAMPING:
#         model.dof_damping[6:] = damping
#         for kp in KP:
#             for percent in Percents:
#                 kd = kp * percent
#                 sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)
#                 sim.controller.reset_phase(kp=kp, kd=kd)

#                 q_run, qr_run, qv_run = [], [], []
#                 for tick in range(600):
#                     action = 0 if tick < 50 else (1 if tick < 200 else 2)
#                     sim.simulation_loop(action)
#                     q_run.append(np.asarray(sim.dg.q, np.float64).ravel())
#                     qr_run.append(np.asarray(sim.dg.qr, np.float64).ravel())
#                     qv_run.append(np.asarray(sim.dg.dq, np.float64).ravel())

#                 Q.append(q_run)
#                 QR.append(qr_run)
#                 QVEL.append(qv_run)
#                 PCTa.append(percent)
#                 DMPa.append(damping)
#                 KPa.append(float(sim.controller.Kp[0]))
#                 KDa.append(float(sim.controller.Kd[0]))
#                 print(f"done kp:{sim.controller.Kp[0]} kd:{sim.controller.Kd[0]} p:{percent} d:{damping} ")

#     out = Path("evaluation/results") / f"{args.out}.npz"
#     out.parent.mkdir(parents=True, exist_ok=True)
#     np.savez_compressed(
#         out,
#         q=np.array(Q),  # shape (n_runs, 600, 12)
#         qr=np.array(QR),  # shape (n_runs, 600, 12)
#         qvel=np.array(QVEL),  # shape (n_runs, 600, 12)
#         kp=np.array(KPa),  # (n_runs,)
#         kd=np.array(KDa),  # (n_runs,)
#         percent=np.array(PCTa),
#         damping=np.array(DMPa),
#         tick=np.arange(600),  # shared time axis
#     )
#     print(f"saved {out} | {len(KPa)} runs")

if __name__ == "__main__":
    main()
