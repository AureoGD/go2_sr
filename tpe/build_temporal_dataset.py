import os
import numpy as np

DATA_DIR = os.path.join("tpe", "data", "processed")

WINDOW = 15


def build_windows(states, labels, window):

    X = []
    y = []

    N = len(states)

    for i in range(N - window):

        X.append(states[i:i + window])
        y.append(labels[i + window - 1])

    X = np.array(X)
    y = np.array(y)

    return X, y


def main():

    print("Loading labeled dataset...")

    data = np.load(os.path.join(DATA_DIR, "dataset_labeled.npz"))

    states = data["states"]
    labels = data["labels"]

    print("States shape:", states.shape)

    print("Building temporal windows...")

    X, y = build_windows(states, labels, WINDOW)

    print("Temporal dataset shape:", X.shape)

    print("Labels shape:", y.shape)

    np.savez(os.path.join(DATA_DIR, "dataset_temporal.npz"), X=X, y=y)

    print("Temporal dataset saved.")


if __name__ == "__main__":
    main()
