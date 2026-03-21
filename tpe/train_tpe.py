import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from tpe.model import TPE

DATA_PATH = "tpe/data/processed/dataset_temporal.npz"

BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3

WINDOW = 15
INPUT_DIM = 19
NUM_CLASSES = 4


def load_dataset():

    print("Loading dataset...")

    data = np.load(DATA_PATH)

    X = data["X"]
    y = data["y"]

    print("Dataset shape:", X.shape)

    return X, y


def subsample(X, y, max_samples=500000):

    if len(X) > max_samples:

        print("Subsampling dataset...")

        idx = np.random.choice(len(X), max_samples, replace=False)

        X = X[idx]
        y = y[idx]

    return X, y


def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.1,
                                                      stratify=y_train,
                                                      random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, X_val, y_train, y_val):

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


def train(model, train_loader, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"loss={total_loss:.3f} "
              f"val_acc={val_acc:.3f}")


def main():

    X, y = load_dataset()

    X, y = subsample(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    train_loader, val_loader = create_dataloaders(X_train, X_val, y_train, y_val)

    print("Creating model...")

    model = TPE(input_dim=INPUT_DIM, window_size=WINDOW, num_classes=NUM_CLASSES)

    train(model, train_loader, val_loader)

    torch.save(model.state_dict(), "tpe_model.pt")

    print("Model saved.")


if __name__ == "__main__":
    main()
