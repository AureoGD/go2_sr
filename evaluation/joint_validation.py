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
    q = d["q"]
    qr = d["qr"]

    diff = q - qr
    mad = np.mean(np.abs(diff), axis=0)
    corr = [np.corrcoef(q[:, i], qr[:, i])[0, 1] for i in range(3)]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ["x (lateral 1)", "y (lateral 2)", "z (normal)"]
    for i, ax in enumerate(axes):
        ax.plot(k[:], q[:, i], label="q", color="tab:blue", linewidth=1.8)
        ax.plot(k[:], qr[:, i], label="qr", color="tab:orange", linewidth=1.8, linestyle="--")

        ax.set_ylabel(f"F_{labels[i]} [N]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"corr={corr[i]:.3f}  mean|diff|={mad[i]:.2f} N", fontsize=9, loc="right")

    axes[-1].set_xlabel("MPC tick k")
    fig.suptitle(f"Foot {foot} contact force: MuJoCo ground truth vs. estimate\n{Path(npz_path).name}")
    plt.tight_layout()

    if out_path is None:
        out_path = Path(npz_path).with_suffix("").as_posix() + f"joint_pos.png"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

    print(f"saved {out_path}")
    print(f"corr (x,y,z) = {corr}")
    print(f"mean|diff| (x,y,z) = {mad}")
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
