"""Per-mode equivalence: OSQP (Python) vs HPIPM (C++) on the SAME QPs.

The simulation is driven by the OSQP solution (exactly the Python controller), and at every
MPC call the identical problem is also solved by HPIPM. Reports, per mode, the first-input
difference |u0_osqp - u0_hpipm|, the objective difference, status agreement and solve times.

    python3 -m control.self_righting.rgc_mpc_cpp.tests.test_mode_equivalence [--ticks 450]
"""
import argparse
import time
from collections import defaultdict

import mujoco
import numpy as np
import pinocchio as pin

from control.self_righting.rgc_mpc_cpp.hpipm_mode import HpipmSolverMixin
from control.self_righting.rgc_mpc_cpp.hybrid_scheduler import SchedulerRGCMPCHybrid
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from fsm.self_righting_fsm import RobotStatus, SelfRightingFSM
from sim.engine.pinocchio_engine import PinocchioEngine
from sim.go2_sim import Go2Sim

RECORDS = defaultdict(list)


class ShadowMixin(HpipmSolverMixin):
    """Solve with OSQP (and use it), solve the same problem with HPIPM (and compare)."""

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)
        self.warm_start_enabled = True  # OSQP exactly as in production

    def build_cost_function(self):
        BaseRGCController.build_cost_function(self)

    def build_constraint_problem(self):
        BaseRGCController.build_constraint_problem(self)

    def setup_solver(self):
        BaseRGCController.setup_solver(self)
        HpipmSolverMixin.setup_solver(self)

    def solve_qp(self):
        t0 = time.perf_counter()
        r_os = BaseRGCController.solve_qp(self)
        t_os = time.perf_counter() - t0
        r_hp = HpipmSolverMixin.solve_qp(self)

        nu = self.nu
        rec = {"osqp_status": r_os.info.status, "hpipm_status": r_hp.info.status, "t_osqp": t_os,
               "t_hpipm": r_hp.info.solve_time, "iter_hpipm": r_hp.info.iter}
        if r_os.x is not None and np.isfinite(r_os.x).all():
            u_os, u_hp = np.asarray(r_os.x[:nu]), r_hp.x[:nu]
            rec["du0"] = float(np.max(np.abs(u_os - u_hp)))
            rec["u0_scale"] = float(np.max(np.abs(u_os)))
            rec["dobj_rel"] = abs(r_os.info.obj_val - r_hp.info.obj_val) / max(1.0, abs(r_os.info.obj_val))
        # score both solutions on the same condensed problem (objective z'Hz+Fz, violation of l<=Gz<=u)
        def score(z):
            if z is None or not np.isfinite(z).all():
                return np.nan, np.nan
            z = np.asarray(z, dtype=float).reshape(-1)
            Gz = self.Gc_sparse @ z
            viol = np.max(np.maximum(np.maximum(self.lc.ravel() - Gz, Gz - self.uc.ravel()), 0.0))
            return float(0.5 * z @ (self.H @ z) + self.F.ravel() @ z), float(viol)
        rec["obj_os"], rec["viol_os"] = score(r_os.x)
        rec["obj_hp"], rec["viol_hp"] = score(r_hp.x)
        RECORDS[type(self).python_mode.__name__].append(rec)
        return r_os


class ShadowScheduler(SchedulerRGCMPCHybrid):

    def mode_class(self, cls):
        base = super().mode_class(cls)
        if base is cls:
            return cls
        return type(f"{cls.__name__}Shadow", (ShadowMixin, cls), {"python_mode": cls})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=450)
    args = ap.parse_args()

    pin_engine = PinocchioEngine(pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", pin.JointModelFreeFlyer()))
    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)
    control = ShadowScheduler(pin_engine=pin_engine, cpp_modes="all")
    sim = Go2Sim(mj_model=model, mj_data=data, controller=control, pin_engine=pin_engine, viewer=None)
    sim.reset_robot_pose(b0=[0, 0, 0.3], r0=[np.pi, 0, 0], q0=[0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6])

    status = RobotStatus(sim.state, sim.controller.task_state, sim.torque_limits, sim.joint_limits)
    fsm = SelfRightingFSM('CW')
    tick = 0
    while not fsm.stand_flag and tick < args.ticks:
        action = 0 if tick < 100 else fsm.update(status)
        sim.simulation_loop(action)
        tick += 1
    print(f"FSM (OSQP-driven) {'stood up' if fsm.stand_flag else 'did NOT stand up'} at tick {tick}\n")

    print(f"{'mode':12s} {'calls':>5s} {'status agree':>12s} {'max|du0|':>10s} {'med|du0|':>10s} {'max|u0|':>9s} "
          f"{'max dobj':>9s} {'OSQP ms':>8s} {'HPIPM ms':>9s} {'HP iters':>8s}")
    worst = 0.0
    for name, recs in RECORDS.items():
        agree = np.mean([(r["osqp_status"] in ("solved", "solved inaccurate")) ==
                         (r["hpipm_status"] in ("solved", "solved inaccurate")) for r in recs])
        du = [r["du0"] for r in recs if "du0" in r]
        sc = [r["u0_scale"] for r in recs if "du0" in r]
        do = [r["dobj_rel"] for r in recs if "du0" in r]
        worst = max(worst, max(du) if du else 0.0)
        print(f"{name:12s} {len(recs):5d} {100 * agree:11.0f}% {max(du) if du else np.nan:10.2e} "
              f"{np.median(du) if du else np.nan:10.2e} {max(sc) if sc else np.nan:9.2e} {max(do) if do else np.nan:9.2e} "
              f"{1e3 * np.mean([r['t_osqp'] for r in recs]):8.2f} {1e3 * np.mean([r['t_hpipm'] for r in recs]):9.2f} "
              f"{np.mean([r['iter_hpipm'] for r in recs]):8.1f}")
    print(f"\nworst |u0_osqp - u0_hpipm| over all modes: {worst:.2e}")

    print("\nlargest differences / status disagreements (obj = z'Hz+Fz on the same QP, viol = max constraint violation):")
    for name, recs in RECORDS.items():
        bad = [r for r in recs if r.get("du0", 0) > 1e-2 or
               (r["osqp_status"] in ("solved", "solved inaccurate")) != (r["hpipm_status"] in ("solved", "solved inaccurate"))]
        for r in bad[:6]:
            print(f"  {name:11s} OSQP {r['osqp_status']:22s} obj {r['obj_os']:12.5e} viol {r['viol_os']:8.1e} | "
                  f"HPIPM {r['hpipm_status']:18s} obj {r['obj_hp']:12.5e} viol {r['viol_hp']:8.1e} | du0 {r.get('du0', np.nan):.1e}")


if __name__ == "__main__":
    main()
