import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from tpe.model import TPE

DATA_PATH = "tpe/data/processed/dataset_temporal.npz"
MODEL_PATH = "tpe_model.pt"

WINDOW = 15
INPUT_DIM = 19
NUM_CLASSES = 3


def load_dataset():

    print("Loading dataset...")

    data = np.load(DATA_PATH)

    X = data["X"]
    y = data["y"]

    print("Dataset:", X.shape)

    return X, y


def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_test, y_test


def load_model(device):

    model = TPE(input_dim=INPUT_DIM, window_size=WINDOW, num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    print("Model loaded.")

    return model


def evaluate(model, X_test, y_test, device):

    X = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X)

    preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = np.mean(preds == y_test)

    print("\nTest Accuracy:", acc)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    return preds


def plot_confusion(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Preparing", "Rolling", "Standing"],
                yticklabels=["Preparing", "Rolling", "Standing"])

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_dataset()

    X_test, y_test = split_dataset(X, y)

    model = load_model(device)

    preds = evaluate(model, X_test, y_test, device)

    plot_confusion(y_test, preds)


if __name__ == "__main__":
    main()
