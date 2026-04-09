from tpe.pipeline.load import load_raw_data
from tpe.pipeline.stats import compute_stats
from tpe.pipeline.normalizer import create_normalizer
from tpe.pipeline.features import build_features
from tpe.pipeline.labels import generate_labels
from tpe.pipeline.temporal import build_temporal_dataset
from tpe.pipeline.save import save_datasets
import numpy as np
import gc


def run_pipeline():

    episodes = load_raw_data()

    stats = compute_stats(episodes)
    normalizer = create_normalizer(stats)

    features, lengths = build_features(episodes, normalizer)

    controllers = np.concatenate([ep["controller"] for ep in episodes])
    labels = generate_labels(features, lengths, controllers)

    X, y = build_temporal_dataset(features, labels, lengths)

    save_datasets(features, labels, X, y, lengths)

    del episodes

    gc.collect()


if __name__ == "__main__":
    run_pipeline()
