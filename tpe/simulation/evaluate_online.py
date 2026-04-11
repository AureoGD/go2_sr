import numpy as np
import mujoco
import mujoco.viewer
import json

from sim.go2_sim import Go2Sim
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from tpe.simulation.scenario import TimeSchedulingScenario

from tpe.core.tpe_module import TPEModule
from tpe.core.phases import Phase
from env.normalizer import StateNormalizer


# ======================================================
# BUILD FEATURE (MESMA PIPELINE)
# ======================================================
def build_feature(state, normalizer):

    rs = state.robot

    pos = rs.r_pos
    vel = rs.r_vel
    omega = rs.omega
    q = rs.q
    dq = rs.dq
    epsilon = rs.epsilon

    eps = 1e-8

    alpha = normalizer.compute_alpha(epsilon)
    pos_norm = normalizer.normalize_position(pos)
    dir_v, v_abs = normalizer.normalize_velocity(vel)

    # omega_norm = np.linalg.norm(omega)
    # dir_omega = omega / (omega_norm + eps)
    # omega_abs = np.tanh(omega_norm / (normalizer.omega_scale + eps))

    dir_omega, omega_abs = normalizer.normalize_omega(omega)

    q_norm = normalizer.normalize_q(q)
    dq_norm = normalizer.compute_dq_norm(dq)

    feature = np.concatenate([[alpha], dir_v, [v_abs], dir_omega, [omega_abs], q_norm, [dq_norm]])

    return feature


# ======================================================
# MAIN
# ======================================================
def run_online_evaluation():

    # ------------------------
    # LOAD NORMALIZER
    # ------------------------
    stats_path = "tpe/data/processed/normalization_stats.json"

    # ------------------------
    # LOAD TPE MODEL
    # ------------------------
    tpe = TPEModule(model_path="tpe/models/tpe_cnn/best_model.pt", window=20)

    # ------------------------
    # MUJOCO
    # ------------------------
    urdf_path = "sim/assets/unitree_go2/go2.urdf"
    scene_path = "sim/assets/unitree_go2/scene.xml"

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch_passive(model, data)

    # ------------------------
    # SCENARIO + CONTROLLER
    # ------------------------
    scenario = TimeSchedulingScenario()
    controller = SchedulerTB(stochastic=True)

    sim = Go2Sim(urdf_path=urdf_path, mj_model=model, mj_data=data, controller=controller, viewer=viewer)

    normalizer = StateNormalizer(joint_limits=sim.joint_limits, torque_limits=sim.torque_limits, stats_path=stats_path)

    # ------------------------
    # RESET
    # ------------------------
    scenario.reset()
    tpe.reset()

    q0, rpy, b0 = scenario.get_initial_state()
    sim.reset_robot_pose(b0=b0, q0=q0, r0=rpy)

    total_steps = scenario.get_total_steps()

    print("\n🚀 Starting ONLINE evaluation...\n")

    # ======================================================
    # LOOP
    # ======================================================
    for step in range(total_steps):

        action = scenario.get_action()

        sim.simulation_loop(action=action)
        state = sim.state

        # ------------------------
        # FEATURE
        # ------------------------
        feature = build_feature(state, normalizer)

        # ------------------------
        # PREDICTION
        # ------------------------
        probs = tpe.predict(feature)

        if probs is not None:
            pred = np.argmax(probs)

            phase_name = Phase(pred).name
            print(f"Step {step:04d} | Ctrl: {action:2d} | TPE: {phase_name} | probs: {np.round(probs, 3)}")

        # opcional: desacelerar visual
        # time.sleep(0.01)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    run_online_evaluation()
