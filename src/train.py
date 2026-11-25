"""Two-stage transfer learning training for HAM10000.

Stage 1: backbone frozen, only the new 7-class head learns.
Stage 2: everything unfrozen, fine-tuned with a small learning rate.

Run from the project root:
    python src/train.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import CLASSES, SPLITS_PATH, SkinLesionDataset
from model import build_model, count_trainable, freeze_backbone, unfreeze_all

# ---------------- Settings (change these to experiment) ----------------
CONFIG = {
    "model_name": "efficientnet_b0",
    "img_size": 224,
    "batch_size": 32,       # lower to 16 if you get "CUDA out of memory"
    "num_workers": 4,       # CPU processes loading images in parallel
    "stage1_epochs": 3,     # head only
    "stage1_lr": 1e-3,
    "stage2_epochs": 15,    # full fine-tuning
    "stage2_lr": 1e-4,      # 10x smaller: adapt gently, don't destroy pretrained knowledge
    "weight_decay": 1e-4,
    "seed": 42,
}
CKPT_DIR = Path("checkpoints")
BEST_PATH = CKPT_DIR / "best_model.pth"
LOG_PATH = CKPT_DIR / "training_log.csv"


def get_class_weights(device):
    """Rare classes get bigger weights, so mistakes on them cost more in the loss."""
    df = pd.read_csv(SPLITS_PATH)
    counts = df[df["split"] == "train"]["dx"].value_counts()
    counts = np.array([counts[c] for c in CLASSES], dtype=np.float32)
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One full pass over a loader. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # autocast = mixed precision: faster and uses less GPU memory
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()   # compute gradients
            scaler.step(optimizer)          # update weights
            scaler.update()

        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "bal_acc": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def train_stage(name, model, train_loader, val_loader, criterion, device, epochs, lr, state):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"\n=== {name}: {count_trainable(model):,} trainable weights, lr={lr} ===")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        improved = va["macro_f1"] > state["best_f1"]
        if improved:
            state["best_f1"] = va["macro_f1"]
            torch.save(model.state_dict(), BEST_PATH)

        print(f"[{name}] epoch {epoch}/{epochs} | "
              f"train loss {tr['loss']:.3f} bal_acc {tr['bal_acc']:.3f} | "
              f"val loss {va['loss']:.3f} bal_acc {va['bal_acc']:.3f} macro_f1 {va['macro_f1']:.3f} | "
              f"{time.time() - t0:.0f}s" + ("  <- best, saved" if improved else ""))

        state["log"].append({
            "stage": name, "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        })


def main():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    CKPT_DIR.mkdir(exist_ok=True)

    loader_args = dict(batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
                       pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(SkinLesionDataset("train", CONFIG["img_size"]), shuffle=True, **loader_args)
    val_loader = DataLoader(SkinLesionDataset("val", CONFIG["img_size"]), shuffle=False, **loader_args)

    model = build_model(CONFIG["model_name"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(device))
    state = {"best_f1": 0.0, "log": []}

    # Stage 1: train only the new head
    freeze_backbone(model)
    train_stage("stage1", model, train_loader, val_loader, criterion, device,
                CONFIG["stage1_epochs"], CONFIG["stage1_lr"], state)

    # Stage 2: fine-tune everything
    unfreeze_all(model)
    train_stage("stage2", model, train_loader, val_loader, criterion, device,
                CONFIG["stage2_epochs"], CONFIG["stage2_lr"], state)

    pd.DataFrame(state["log"]).to_csv(LOG_PATH, index=False)
    print(f"\nDone. Best val macro F1: {state['best_f1']:.3f}")
    print(f"Best model saved to {BEST_PATH}, log saved to {LOG_PATH}")


# Required on Windows: DataLoader workers re-import this file, and this guard
# stops them from starting training again themselves.
if __name__ == "__main__":
    main()