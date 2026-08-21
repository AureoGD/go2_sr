"""
Load a RollCW debug .npz log and plot the RL foot contact-force comparison
(MuJoCo ground truth `gn` vs. measured-torque `force_est`), per axis.

Usage:
    python plot_grf.py path/to/run.npz
    python plot_grf.py path/to/run.npz --foot 3           # foot index into gn (default 3 = RL)
    python plot_grf.py path/to/run.npz --out my_fig.png   # custom output name
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_grf(npz_path, foot=3, out_path=None):
    d = np.load(npz_path)

    k = d["k"]
    rt = d["rt"]
    re = d["re"]
    bt = d["bt"]
    dq_r = d["dq_r"]
    dq_e = d["dq_e"]
    dq_er = d["dq_er"]
    # bt = d["bt"]  # (N, 3)
    # be = d["be"]  # (N, 3)
    # rt = d["rt"]
    # re = d['re']
    # pp = d['pp']
    # pf = d['pf']
    # diff = bt - be
    # mad = np.mean(np.abs(diff), axis=0)
    # corr = [np.corrcoef(bt[:, i], be[:, i])[0, 1] for i in range(3)]
    # re *= 0.1
    # e *= 3.3
    # r = re[:, 0] / np.clip(rt[:, 0], 1e-3, None)  # plateau ticks only
    # print(r.mean())
    # r = re[:, 1] / np.clip(rt[:, 1], 1e-3, None)  # plateau ticks only
    # print(r.mean())
    # r = re[:, 2] / np.clip(rt[:, 2], 1e-3, None)  # plateau ticks only
    # print(r.mean())

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ["1", "2", "3"]
    for i, ax in enumerate(axes):
        # ax.plot(k, rt[:, i], label="dr true", color="tab:green", linewidth=1.8)
        # ax.plot(k, re[:, i], label="dr estimado", color="tab:orange", linewidth=1.8, linestyle="--")
        # ax.plot(k, bt[:, i], label="db true", color="tab:red", linewidth=1.8, linestyle="--")

        ax.plot(k, dq_r[:, i], label="dq real", color="tab:red", linewidth=1.8)
        ax.plot(k, dq_e[:, i], label="dq estimado corrigido", color="tab:blue", linewidth=1.8, linestyle="--")
        ax.plot(k, dq_er[:, i], label="dq estimado raw", color="tab:green", linewidth=1.8, linestyle="--")
        ax.set_ylabel(f"Joint {labels[i]} [rad/s]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("MPC tick k")
    fig.suptitle(f"Joint velocitt \n{Path(npz_path).name}")
    plt.tight_layout()

    if out_path is None:
        out_path = Path(npz_path).with_suffix("").as_posix() + f"_joint_vel.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

    # print(f"saved {out_path}")
    # print(f"corr (x,y,z) = {corr}")
    # print(f"mean|diff| (x,y,z) = {mad}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_path",
                        type=str,
                        default="evaluation/results/validation.npz",
                        help="path to the debug .npz log")
    parser.add_argument("--foot", type=int, default=3, help="foot index into gn (default 3 = RL)")
    parser.add_argument("--out", type=str, default=None, help="output PNG path")
    args = parser.parse_args()

    plot_grf(args.npz_path, foot=args.foot, out_path=args.out)
