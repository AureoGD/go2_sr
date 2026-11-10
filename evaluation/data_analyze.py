#!/usr/bin/env python3
"""
Analyze RGC-MPC logs saved in an .npz file.

Examples
--------
python analyze_rgc_mpc.py rgc_mpc_data.npz --controller 3
python analyze_rgc_mpc.py rgc_mpc_data.npz --controller 3 --start 150 --end 180
python analyze_rgc_mpc.py rgc_mpc_data.npz --controller 3 --plot

Notes
-----
- Controller filtering uses the logged `controller_index` array.
- Each controller has its own hard-constraint and soft-constraint mapping in
  `CONTROLLER_CONFIG`, keyed by controller_index/phase.
- `constraint_violation` columns are grouped using that controller's slices.
  The script warns if the logged violation vector length differs from nch.
- Solver status values follow OSQP conventions:
    1 solved, 2 solved inaccurate, 3 primal infeasible,
    4 primal infeasible inaccurate, 5 dual infeasible,
    6 dual infeasible inaccurate, 7 maximum iterations reached.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

# Constraint validation depends on the controller: each controller can have
# a different hard-constraint layout and different soft-constraint rows.
# All slices are zero-based and end-exclusive.
#
# IMPORTANT:
#   - The Roll mapping below is the mapping previously shared for RollCW.
#   - StandUp's GRF family was reported to start at global slot 24, with
#     20 rows (5 per leg). Other StandUp groups are intentionally left unmapped
#     until their exact logger indices are confirmed.
#   - Add/adjust entries here for each controller using the exact layout used
#     when `constraint_violation` is logged. Do not reuse one controller's
#     layout for another controller.
CONTROLLER_CONFIG = {
    # phase/controller_index 1: GoSafe
    1: {
        "name": "GoSafe",
        "nch": 24,
        "ncs": 6,
        "violation_groups": {
            "joint_position": slice(0, 12),
            "torque":         slice(12, 24),
        },
        "soft_constraint_names": {
            0: "self-collision FR-FL",
            1: "self-collision FR-RR",
            2: "self-collision FR-RL",
            3: "self-collision FL-RR",
            4: "self-collision FL-RL",
            5: "self-collision RR-RL",
        },
    },

    # phase/controller_index 2: PrepareCW
    2: {
        "name": "PrepareCW",
        "nch": 24,
        "ncs": 9,
        "violation_groups": {
            "joint_position": slice(0, 12),
            "torque":         slice(12, 24),
        },
        "soft_constraint_names": {
            0: "self-collision FR-FL",
            1: "self-collision FR-RR",
            2: "self-collision FR-RL",
            3: "self-collision FL-RR",
            4: "self-collision FL-RL",
            5: "self-collision RR-RL",
            6: "RL calf-plane clearance",
            7: "FR thigh normal-force condition",
            8: "RR thigh normal-force condition",
        },
    },

    # phase/controller_index 3: RollCW
    3: {
        "name": "RollCW",
        "nch": 34,
        "ncs": 8,
        "violation_groups": {
            "joint_position": slice(0, 12),
            "torque":         slice(12, 24),
            "CoM":             slice(24, 29),
            "capture_point":   slice(29, 34),
        },
        "soft_constraint_names": {
            0: "RL-foot friction-pyramid row 0",
            1: "RL-foot friction-pyramid row 1",
            2: "RL-foot friction-pyramid row 2",
            3: "RL-foot friction-pyramid row 3",
            4: "RL-foot normal-force row",
            5: "FR-thigh normal-force row",
            6: "RR-thigh normal-force row",
            7: "capture-point / CSI condition",
        },
    },

    # phase/controller_index 4: SwingLegCW (the landing/swing-foot phase)
    4: {
        "name": "SwingLegCW",
        "nch": 24,
        "ncs": 9,
        "violation_groups": {
            "joint_position": slice(0, 12),
            "torque":         slice(12, 24),
        },
        "soft_constraint_names": {
            0: "capture-point / CSI condition",
            1: "CoM/ 1",
            2: "CoM/ 2",
            3: "CoM/ 3",
            4: "CoM/ 4",
            5: "csi/ 5",
            6: "csi/ 6",
            7: "csi/ 7",
            8: "csi/ 8",
        },
    },

    # phase/controller_index 5: SettleCW
    5: {
        "name": "SettleCW",
        "nch": 20,
        "ncs": 20,
        "violation_groups": {
            "joint_reference": slice(0, 12),
            "CoM":             slice(12, 16),
            "capture_point":   slice(16, 20),
        },
        "soft_constraint_names": {
            **{i: f"GRF friction-pyramid row {i}" for i in range(20)},
        },
    },

    # phase/controller_index 6: ProneCW
    6: {
        "name": "ProneCW",
        "nch": 24,
        "ncs": 0,
        "violation_groups": {
            "joint_position": slice(0, 12),
            "torque":         slice(12, 24),
        },
        "soft_constraint_names": {},
    },

    # phase/controller_index 7: StandUp
    7: {
        "name": "StandUp",
        "nch": 44,
        "ncs": 0,
        "violation_groups": {
            "joint_reference": slice(0, 12),
            "CoM":             slice(12, 18),
            "capture_point":   slice(18, 24),
            "GRF":             slice(24, 44),
        },
        "soft_constraint_names": {},
    },
}

STATUS_NAMES = {
    1: "solved",
    2: "solved inaccurate",
    3: "primal infeasible",
    4: "primal infeasible inaccurate",
    5: "dual infeasible",
    6: "dual infeasible inaccurate",
    7: "maximum iterations reached",
}


def finite_values(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def describe(name, values, sample_ids):
    vals = np.asarray(values, dtype=float).ravel()
    good = np.isfinite(vals)
    if not np.any(good):
        print(f"  {name}: no finite values")
        return
    v = vals[good]
    ids = np.asarray(sample_ids)[good]
    imax = int(np.argmax(v))
    imin = int(np.argmin(v))
    print(
        f"  {name}: max={v[imax]:.6g} (sample {ids[imax]}), "
        f"min={v[imin]:.6g} (sample {ids[imin]}), mean={np.mean(v):.6g}"
    )


def load_required(data, key):
    if key not in data:
        print(f"Missing expected key: {key!r}")
        return None
    return data[key]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize RGC-MPC solver and constraint metrics for one controller."
    )
    parser.add_argument("file", help="Path to the RGC-MPC .npz log")
    parser.add_argument("--controller", "-c", type=int, required=True,
                        help="controller_index value to analyze")
    parser.add_argument("--start", type=int, default=None,
                        help="Optional inclusive global sample/tick lower bound")
    parser.add_argument("--end", type=int, default=None,
                        help="Optional inclusive global sample/tick upper bound")
    parser.add_argument("--plot", action="store_true",
                        help="Plot solver residuals, slack, and grouped violations")
    parser.add_argument("--save-csv", default=None,
                        help="Optional path to save selected per-sample metrics as CSV")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        controller = load_required(data, "controller_index")
        if controller is None:
            sys.exit(1)

        controller = np.asarray(controller).ravel()
        global_samples = np.arange(controller.size)
        mask = controller == args.controller
        if args.start is not None:
            mask &= global_samples >= args.start
        if args.end is not None:
            mask &= global_samples <= args.end
        ids = global_samples[mask]

        if ids.size == 0:
            available = np.unique(controller)
            sys.exit(
                f"No samples found for controller_index={args.controller} "
                f"within the selected range. Available indices: {available.tolist()}"
            )

        print("=" * 72)
        print(f"File: {path}")
        config = CONTROLLER_CONFIG.get(args.controller, {})
        controller_name = config.get("name", "Unconfigured controller")
        violation_groups = config.get("violation_groups", {})
        soft_names = config.get("soft_constraint_names", {})
        print(f"Controller index: {args.controller} ({controller_name})")
        if not config:
            print("WARNING: No controller-specific validation config is defined.")
            print("         Solver metrics will be reported, but constraint groups")
            print("         and soft-row labels must be configured for this controller.")
        print(f"Selected samples: {ids.size}")
        print(f"Global sample range: {ids[0]}..{ids[-1]}")
        print("=" * 72)

        status = np.asarray(data["solver_status"]).ravel()[ids] if "solver_status" in keys else None
        if status is not None:
            print("\nSOLVER STATUS")
            rounded = np.rint(status).astype(int)
            for code in sorted(np.unique(rounded)):
                count = int(np.sum(rounded == code))
                name = STATUS_NAMES.get(code, "unknown / custom status")
                print(f"  {code}: {name}: {count}/{ids.size} ({100*count/ids.size:.1f}%)")
            strict = int(np.sum(rounded == 1))
            print(f"  Strict solved rate: {strict}/{ids.size} ({100*strict/ids.size:.1f}%)")
            print(f"  Non-strict-solved count: {ids.size - strict}")

        print("\nSOLVER DIAGNOSTICS")
        for key in ("mpc_obj_val", "primal_res", "dual_res", "lambda_max", "slack_max"):
            if key in keys:
                describe(key, data[key][ids], ids)

        if "mpc_fail" in keys:
            fail = np.asarray(data["mpc_fail"]).ravel()[ids].astype(bool)
            print(f"  mpc_fail: {int(fail.sum())}/{ids.size}")
        if "mpc_critical_fail" in keys:
            fail = np.asarray(data["mpc_critical_fail"]).ravel()[ids].astype(bool)
            print(f"  mpc_critical_fail: {int(fail.sum())}/{ids.size}")

        print("\nSOFT-CONSTRAINT SLACK BY ROW")
        if "slack_max_per_row" in keys:
            slack = np.asarray(data["slack_max_per_row"])[ids]
            if slack.ndim == 1:
                slack = slack[:, None]
            for row in range(slack.shape[1]):
                vals = slack[:, row]
                good = np.isfinite(vals)
                if not np.any(good):
                    continue
                valid_ids = ids[good]
                vv = vals[good]
                imax = int(np.argmax(vv))
                print(
                    f"  row {row} ({soft_names.get(row, 'unlabeled soft constraint')}): "
                    f"max={vv[imax]:.6g} (sample {valid_ids[imax]}), "
                    f"mean={np.mean(vv):.6g}, "
                    f"nonzero={(np.abs(vv) > 1e-9).sum()}/{len(vv)}"
                )
        else:
            slack = None
            print("  slack_max_per_row not present")

        print("\nCONSTRAINT VIOLATIONS")
        if "constraint_violation" in keys:
            violations = np.asarray(data["constraint_violation"])[ids]
            if violations.ndim == 1:
                violations = violations[:, None]
            print(f"  Logged violation array shape per sample: {violations.shape[1:]}")
            expected_nch = config.get("nch")
            if expected_nch is not None and violations.shape[1] != expected_nch:
                print(
                    f"  WARNING: configured nch={expected_nch}, but logged violation vector "
                    f"has {violations.shape[1]} columns. Check the logger's indexing before "
                    "interpreting grouped maxima."
                )
            for name, sl in violation_groups.items():
                if sl.stop is not None and sl.stop <= violations.shape[1]:
                    group = violations[:, sl]
                    finite = group[np.isfinite(group)]
                    if finite.size:
                        flat_index = np.nanargmax(group)
                        row_idx, col_idx = np.unravel_index(flat_index, group.shape)
                        print(
                            f"  {name}: max={group[row_idx, col_idx]:.6g} "
                            f"(sample {ids[row_idx]}, violation column {sl.start + col_idx})"
                        )
                    else:
                        print(f"  {name}: no finite values")
            if not violation_groups:
                print("  No controller-specific groups configured; reporting raw columns only.")
            print("  Per-column maxima:")
            for col in range(violations.shape[1]):
                vals = violations[:, col]
                good = np.isfinite(vals)
                if np.any(good):
                    valid_ids = ids[good]
                    vv = vals[good]
                    imax = int(np.argmax(vv))
                    print(f"    col {col}: {vv[imax]:.6g} (sample {valid_ids[imax]})")
        else:
            violations = None
            print("  constraint_violation not present")

        print("\nCONTACT / TRANSITION CONTEXT")
        for key in ("robot.foot_touching", "robot.force_mj", "cp_trig_signal",
                    "com_trig_signal", "controller_evolution", "last_controller"):
            if key not in keys:
                continue
            arr = np.asarray(data[key])
            # Controller-level arrays have one entry per logged MPC interaction;
            # robot arrays may have a higher-rate simulation sampling frequency.
            if arr.ndim >= 1 and arr.shape[0] == controller.size:
                selected = arr[ids]
                if key == "robot.foot_touching":
                    print(f"  {key}: first={selected[0]}, last={selected[-1]}")
                elif key == "robot.force_mj":
                    print(f"  {key}: selected shape={selected.shape}; inspect individual feet/axes as needed")
                elif key in ("cp_trig_signal", "com_trig_signal"):
                    print(f"  {key}: true={int(np.sum(selected.astype(bool)))}/{len(selected)}")
                elif key == "controller_evolution":
                    describe(key, selected, ids)
                else:
                    print(f"  {key}: first={selected[0]}, last={selected[-1]}")
            elif key.startswith("robot."):
                print(f"  {key}: stored at a different sampling rate; not aligned automatically")

        if args.save_csv:
            import csv
            columns = {"sample": ids, "controller_index": controller[ids]}
            for key in ("solver_status", "mpc_obj_val", "mpc_fail", "mpc_critical_fail",
                        "primal_res", "dual_res", "lambda_max", "slack_max",
                        "controller_evolution", "cp_trig_signal", "com_trig_signal"):
                if key in keys:
                    columns[key] = np.asarray(data[key])[ids]
            if "slack_max_per_row" in keys:
                ss = np.asarray(data["slack_max_per_row"])[ids]
                for j in range(ss.shape[1]):
                    columns[f"slack_row_{j}"] = ss[:, j]
            if "constraint_violation" in keys:
                vv = np.asarray(data["constraint_violation"])[ids]
                for j in range(vv.shape[1]):
                    columns[f"violation_col_{j}"] = vv[:, j]
            with open(args.save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns.keys())
                writer.writerows(zip(*columns.values()))
            print(f"\nSaved per-sample metrics to: {args.save_csv}")

        if args.plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                sys.exit("Install matplotlib to use --plot (python -m pip install matplotlib)")
            panels = []
            if "primal_res" in keys:
                panels.append(("primal_res", np.asarray(data["primal_res"])[ids]))
            if "dual_res" in keys:
                panels.append(("dual_res", np.asarray(data["dual_res"])[ids]))
            if "slack_max" in keys:
                panels.append(("slack_max", np.asarray(data["slack_max"])[ids]))
            if "constraint_violation" in keys:
                vv = np.asarray(data["constraint_violation"])[ids]
                finite = vv[np.isfinite(vv)]
                if finite.size:
                    # Ignore non-finite sentinels; plot maximum logged value per sample.
                    panels.append(("max constraint_violation", np.nanmax(vv, axis=1)))
            if panels:
                fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.7 * len(panels)), sharex=True)
                if len(panels) == 1:
                    axes = [axes]
                for ax, (name, values) in zip(axes, panels):
                    ax.plot(ids, values, marker=".")
                    ax.set_ylabel(name)
                    ax.grid(True)
                axes[-1].set_xlabel("Global sample")
                fig.suptitle(f"RGC-MPC metrics: controller_index={args.controller}")
                fig.tight_layout()
                plt.show()


if __name__ == "__main__":
    main()
