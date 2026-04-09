import os
import numpy as np

OUT_DIR = "tpe/data/processed"


def save_datasets(features, labels, X, y, lengths):

    np.savez_compressed(os.path.join(OUT_DIR, "dataset.npz"), features=features, labels=labels, lengths=lengths)

    np.savez_compressed(os.path.join(OUT_DIR, "dataset_temporal.npz"), X=X, y=y)

    print("\n Datasets salvos!")
