import os
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = os.path.join("tpe", "data", "raw")

EPS = 1e-8


def compute_alpha(rpy):

    rpy = rpy.reshape(-1, 3)

    roll = rpy[:, 0]
    pitch = rpy[:, 1]

    alpha = np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1))
    alpha = 1 - alpha / np.pi

    return alpha


def compute_features(data):

    v = data["com_vel"].reshape(-1, 3)
    w = data["ang_vel"].reshape(-1, 3)
    rpy = data["rpy"].reshape(-1, 3)

    alpha = compute_alpha(rpy)

    v_norm = np.linalg.norm(v, axis=1)
    dir_v = v / (v_norm[:, None] + EPS)

    v_abs = np.tanh(v_norm)

    w_norm = np.linalg.norm(w, axis=1)
    dir_w = w / (w_norm[:, None] + EPS)

    wx = w[:, 0]
    wy = w[:, 1]
    wz = w[:, 2]

    return alpha, wx, wy, dir_w[:, 2], v_abs, dir_v[:, 2]


def plot_episode(data):

    controller = data["controller"].flatten()

    alpha, wx, wy, wz, v_abs, dir_x = compute_features(data)

    t = np.arange(len(alpha))

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(12, 10))

    signals = [alpha, wz, v_abs, dir_x]
    titles = ["alpha", "omega_x", "v_abs", "dir_v_x"]

    actions = np.unique(controller)

    cmap = plt.cm.get_cmap("tab10", len(actions))
    color_map = {a: cmap(i) for i, a in enumerate(actions)}

    # ---- main signals ----
    for ax, signal, title in zip(axs[:4], signals, titles):

        ax.plot(t, signal, color="black", linewidth=1)

        start = 0
        current = controller[0]

        for i in range(1, len(controller)):

            if controller[i] != current:

                ax.axvspan(start, i, color=color_map[current], alpha=0.25)

                start = i
                current = controller[i]

        ax.axvspan(start, len(controller), color=color_map[current], alpha=0.25)

        ax.set_ylabel(title)

    # ---- angular velocity plot ----
    axs[4].plot(t, wx, label="ω_x")
    # axs[4].plot(t, wy, label="ω_y")
    # axs[4].plot(t, wz, label="ω_z")

    axs[4].set_ylabel("angular vel")
    axs[4].legend()

    axs[-1].set_xlabel("time")

    # controller legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[a], alpha=0.5) for a in actions]

    labels = [f"action {a}" for a in actions]

    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()


def main():

    files = sorted([os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".npz")])

    print("Available episodes:")

    for i, f in enumerate(files):
        print(i, os.path.basename(f))

    idx = int(input("Select episode: "))

    data = np.load(files[idx])

    plot_episode(data)


if __name__ == "__main__":
    main()
