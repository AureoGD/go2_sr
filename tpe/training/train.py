import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

from tpe.core.model import TPE

DATA_PATH = "tpe/data/processed/dataset.npz"
MODEL_DIR = "tpe/models/tpe_cnn"


def train():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(" Loading base dataset...")
    data = np.load(DATA_PATH)

    features = data["features"]
    labels = data["labels"]
    lengths = data["lengths"]

    # -------------------------
    # SPLIT POR EPISÓDIO
    # -------------------------
    train_idx, val_idx = split_by_episodes(features, labels, lengths)

    # -------------------------
    # TEMPORAL DATASET
    # -------------------------
    X_train, y_train = build_temporal_from_indices(features, labels, train_idx)
    X_val, y_val = build_temporal_from_indices(features, labels, val_idx)

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)

    # -------------------------
    # DATA LOADERS
    # -------------------------
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # -------------------------
    # MODEL
    # -------------------------
    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[2]

    model = TPE(input_dim=input_dim, num_classes=num_classes, window_size=20)

    # -------------------------
    # CLASS WEIGHTS
    # -------------------------
    counts = np.bincount(y_train)
    weights = 1.0 / (counts + 1e-8)
    weights /= weights.sum()

    weights = torch.tensor(weights, dtype=torch.float32)

    print(" Train distribution:", counts)

    # -------------------------
    # TRAIN
    # -------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val = float("inf")

    for epoch in range(10):

        # TRAIN
        model.train()
        train_loss = 0

        for X, y in train_loader:

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in val_loader:

                pred = model(X)
                loss = criterion(pred, y)

                val_loss += loss.item() * X.size(0)

                _, predicted = torch.max(pred, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader.dataset)
        acc = correct / total

        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss

            model_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)

            config = {"input_dim": input_dim, "num_classes": num_classes, "window_size": 20}

            config_path = os.path.join(MODEL_DIR, "config.json")

            import json
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

    print("\n Done!")


# ======================================================
# HELPERS (coloque no mesmo arquivo)
# ======================================================
def split_by_episodes(features, labels, lengths, train_ratio=0.8):

    num_episodes = len(lengths)
    split_ep = int(train_ratio * num_episodes)

    train_idx, val_idx = [], []

    start = 0
    for i, L in enumerate(lengths):

        end = start + L

        if i < split_ep:
            train_idx.append((start, end))
        else:
            val_idx.append((start, end))

        start = end

    return train_idx, val_idx


def build_temporal_from_indices(features, labels, indices, window=20):

    X, y = [], []

    for (start, end) in indices:

        f = features[start:end]
        l = labels[start:end]

        for i in range(window, len(f)):

            if l[i] == -1:
                continue

            X.append(f[i - window:i])
            y.append(l[i])

    return np.array(X, dtype=np.float32), np.array(y)


if __name__ == "__main__":
    train()
