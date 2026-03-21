import os
import numpy as np

RAW_DIR = os.path.join("tpe", "data", "raw")
FEATURE_DIR = os.path.join("tpe", "data", "processed")


# --------------------------------------------
# LOAD CONTROLLERS (concatenado)
# --------------------------------------------
def load_controllers():

    files = sorted([os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".npz")])

    controllers = []

    for f in files:
        data = np.load(f)
        ctrl = data["controller"].reshape(-1)
        controllers.append(ctrl)

    return np.concatenate(controllers)


# --------------------------------------------
# LABELING POR EPISÓDIO
# --------------------------------------------
def assign_labels_episode(states, controllers):

    alpha = states[:, 0]

    labels = np.full(len(alpha), -1)

    prepare_ctrl = [1, 2, 7]
    roll_ctrl = [3, 4, 8, 9]
    stand_ctrl = [5, 6, 10]

    i = 0
    N = len(alpha)

    while i < N:

        c = controllers[i]

        # -------------------------
        # PREPARING
        # -------------------------
        if c in prepare_ctrl:
            labels[i] = 0
            i += 1

        # -------------------------
        # ROLLING (segment-based)
        # -------------------------
        elif c in roll_ctrl:

            start = i

            while i < N and controllers[i] in roll_ctrl:
                i += 1

            end = i

            alpha_seg = alpha[start:end]

            # derivada
            diff = np.diff(alpha_seg)

            decay_idx = None
            eps = 0.01

            for k in range(len(diff)):
                if diff[k] < -eps:
                    decay_idx = k + 1
                    break

            # -------- caso 1: nunca caiu --------
            if decay_idx is None:

                if alpha_seg[-1] > alpha_seg[0] + 0.05:
                    labels[start:end] = 1  # rolling
                else:
                    labels[start:end] = 3  # fail

            # -------- caso 2: caiu --------
            else:

                labels[start:start + decay_idx] = 1
                labels[start + decay_idx:end] = 3

        # -------------------------
        # STANDING
        # -------------------------
        elif c in stand_ctrl:

            if alpha[i] <= 0.3:
                labels[i] = 3
            else:
                labels[i] = 2

            i += 1

        else:
            i += 1

    return labels


# --------------------------------------------
# MAIN
# --------------------------------------------
def main():

    print("Loading features...")

    data = np.load(os.path.join(FEATURE_DIR, "dataset_features.npz"))

    states = data["states"]
    episode_lengths = data["episode_lengths"]

    print("Loading controllers...")

    controllers = load_controllers()

    assert len(states) == len(controllers), "Mismatch states/controllers!"

    print("Assigning labels per episode...")

    labels = np.full(len(states), -1)

    start = 0

    for ep_idx, L in enumerate(episode_lengths):

        end = start + L

        print(f"Episode {ep_idx}: {start} -> {end}")

        labels[start:end] = assign_labels_episode(states[start:end], controllers[start:end])

        start = end

    # remover inválidos
    valid = labels >= 0

    states = states[valid]
    labels = labels[valid]

    print("Final dataset size:", states.shape)

    np.savez(os.path.join(FEATURE_DIR, "dataset_labeled.npz"), states=states, labels=labels)

    print("Dataset saved.")


if __name__ == "__main__":
    main()
