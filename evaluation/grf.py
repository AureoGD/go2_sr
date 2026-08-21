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
    f_com = d["f_com"]
    f_t1 = d["f_t1"]
    f_t2 = d["f_t2"]
    f_t3 = d["f_t3"]
    f_t4 = d["f_t4"]
    f_c1 = d["f_c1"]
    f_c2 = d["f_c2"]
    f_c3 = d["f_c3"]

    # diff = f_com - f_pres
    # mad = np.mean(np.abs(diff), axis=0)
    f_sum = f_t1 + f_t3 + f_t4
    f_sum_c = f_c2 + f_c1 + f_c3
    # corr = [np.corrcoef(f_com[:, i], f_sum[:, i])[0, 1] for i in range(3)]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ["x (lateral 1)", "y (lateral 2)", "z (normal)"]
    for i, ax in enumerate(axes):
        ax.plot(k, f_com[:, i], label="Estimate via jacobian ", color="tab:red", linewidth=1.8)
        # ax.plot(k, f_t4[:, i] + f_t2[:, i], label="Estimate via jacobian + G ", color="tab:orange", linewidth=1.8)
        # ax.plot(k, f_t4[:, i] - f_t2[:, i], label="Estimate via jacobian - G ", color="tab:green", linewidth=1.8)
        # ax.plot(k, f_c3[:, i], label="MuJoCo contacts force", color="tab:blue", linewidth=1.8)
        ax.set_ylabel(f"F_{labels[i]} [N]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        # ax.set_title(f"corr={corr[i]:.3f}  mean|diff|={mad[i]:.2f} N", fontsize=9, loc="right")

    axes[-1].set_xlabel("MPC tick k")
    fig.suptitle(f"Foot {foot} contact force: MuJoCo ground truth vs. estimate\n{Path(npz_path).name}")
    plt.tight_layout()

    if out_path is None:
        out_path = Path(npz_path).with_suffix("").as_posix() + f"{foot}_grf.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

    print(f"saved {out_path}")
    # print(f"corr (x,y,z) = {corr}")
    # print(f"mean|diff| (x,y,z) = {mad}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_path",
                        type=str,
                        default="evaluation/results/rgc_data.npz",
                        help="path to the debug .npz log")
    parser.add_argument("--foot", type=int, default=3, help="foot index into gn (default 3 = RL)")
    parser.add_argument("--out", type=str, default=None, help="output PNG path")
    args = parser.parse_args()

    plot_grf(args.npz_path, foot=args.foot, out_path=args.out)
