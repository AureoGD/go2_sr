import numpy as np


def build_temporal_dataset(features, labels, lengths, window=20):

    X, y = [], []
    start = 0

    for L in lengths:

        end = start + L

        f = features[start:end]
        l = labels[start:end]

        for i in range(window, L):

            if l[i] == -1:
                continue  # 🔥 filtro correto

            X.append(f[i - window:i])
            y.append(l[i])

        start = end

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("\n Temporal dataset:", X.shape)

    return X, y
