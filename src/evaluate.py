"""Final evaluation of the best model on the held-out TEST set.

Run once, after training is finished (from the project root):
    python src/evaluate.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader

from dataset import CLASSES, SkinLesionDataset
from model import build_model

MODEL_NAME = "efficientnet_b0"
CKPT_PATH = Path("checkpoints/best_model.pth")
RESULTS_DIR = Path("results")


@torch.no_grad()  # no learning here, so no gradients needed (faster, less memory)
def predict(model, loader, device):
    """Run the model on every test image; return class probabilities and true labels."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())  # scores -> probabilities
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(exist_ok=True)

    # Same architecture as training, then load YOUR trained weights into it
    model = build_model(MODEL_NAME, pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    test_loader = DataLoader(SkinLesionDataset("test"), batch_size=64, shuffle=False, num_workers=4)
    probs, labels = predict(model, test_loader, device)
    preds = probs.argmax(1)  # the class with the highest probability

    # ---- Overall scores ----
    mel = CLASSES.index("mel")
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_roc_auc": roc_auc_score(labels, probs, multi_class="ovr", average="macro"),
        "melanoma_recall": float((preds[labels == mel] == mel).mean()),
    }
    print("\n=== TEST SET RESULTS ===")
    for name, value in metrics.items():
        print(f"{name:18s}: {value:.3f}")

    # ---- Per-class scores ----
    report = classification_report(labels, preds, target_names=CLASSES, digits=3)
    print("\n=== PER-CLASS REPORT ===")
    print(report)

    # ---- Confusion matrix (raw counts in terminal, normalized image saved) ----
    print("=== CONFUSION MATRIX (rows = true class, columns = predicted) ===")
    print("       " + " ".join(f"{c:>6s}" for c in CLASSES))
    for c, row in zip(CLASSES, confusion_matrix(labels, preds)):
        print(f"{c:>6s} " + " ".join(f"{v:6d}" for v in row))

    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title("Test set confusion matrix (each row sums to 1)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)

    # ---- Save everything for the README ----
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "classification_report.txt").write_text(report)
    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()