import numpy as np


def assign_labels_episode(states, controllers):

    alpha = states[:, 0]
    labels = np.full(len(alpha), -1)

    prepare = [0, 1, 2, 7]
    roll = [3, 4, 8, 9]
    stand = [5, 6, 10]

    i = 0
    N = len(alpha)

    while i < N:

        c = controllers[i]

        if c in prepare:
            labels[i] = 0
            i += 1

        elif c in roll:

            start = i
            while i < N and controllers[i] in roll:
                i += 1
            end = i

            alpha_seg = alpha[start:end]
            diff = np.diff(alpha_seg)

            decay = None
            for k in range(len(diff)):
                if diff[k] < -0.01:
                    decay = k + 1
                    break

            if decay is None:
                labels[start:end] = 1 if alpha_seg[-1] > alpha_seg[0] + 0.05 else 3
            else:
                labels[start:start + decay] = 1
                labels[start + decay:end] = 3

        elif c in stand:
            labels[i] = 2 if alpha[i] > 0.3 else 3
            i += 1

        else:
            i += 1

    return labels


def generate_labels(features, lengths, controllers):

    labels = np.full(len(features), -1)

    start = 0

    for L in lengths:
        end = start + L

        labels[start:end] = assign_labels_episode(features[start:end], controllers[start:end])

        start = end

    return labels
